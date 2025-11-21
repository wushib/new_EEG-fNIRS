# -*- coding: utf-8 -*-
"""
跨模态_v2.py
-----------------------------------------
基于 EEG-fNIRS 对齐窗口，构建跨模态图 + 增强节点特征。

特点 / 可写进论文的点：
1) EEG 节点特征 (每通道 17 维)
   - 5 个频带 DE（delta/theta/alpha/beta/gamma）
   - 5 个频带相对功率 (band power / total power)
   - 2 个频带比值 (theta/alpha, beta/alpha)
   - 3 个 Hjorth 参数 (activity, mobility, complexity)
   - 2 个动态特征：前半 vs 后半 的 alpha 功率差、全频功率差

2) fNIRS 节点特征 (每通道 10 维，建议 HbO；可扩 HbR)
   - mean, slope, var
   - 0.01–0.1 Hz 带功率
   - log(var)
   - peak-to-peak 振幅
   - 一阶差分 std (快速波动)
   - half_mean_diff (后半均值 - 前半均值)
   - half_slope_diff (后半斜率 - 前半斜率)
   - norm_peak2peak = peak2peak / (std+eps)

3) 特征维度对齐：
   - 使用 align_feature_dim() 将 EEG / fNIRS 特征零填充到相同维度，
     既可进行跨模态距离计算，又满足图卷积输入要求。

4) 跨模态图构建：
   - 分别计算 EEG-EEG, fNIRS-fNIRS, EEG-fNIRS 的距离 (cosine)
   - 自适应 σ 的 RBF 相似度
   - 分块 row-wise kNN 稀疏化
   - 跨模态边重标定 + 衰减 γ
   - 对称化 + 自环

5) 结构增强节点特征：
   - modality_id: EEG=1, fNIRS=0
   - deg_total: 节点度
   - deg_within: 模态内度
   -> 最终 xmodal_feat 用于下游 GCN/Co-Attn/Gate 模型。

假设已存在：
  subj_xx_aligned_windows.npz
或原始干净信号文件（作为 fallback）。

标签文件：
  subj_xx_labels.npy
由你的标签脚本单独生成，本文件不修改。
"""

import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from tqdm import tqdm
from scipy.signal import welch
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import seaborn as sns

# ========= 路径 & 全局参数 =========

# 改成你自己的数据目录（保存结果也在这里）
SAVE_DIR = r"D:\GNN\MI运动表现\数据处理"
os.makedirs(SAVE_DIR, exist_ok=True)

# 当没有 aligned_windows.npz 时的兜底采样参数
EEG_FS_FALLBACK = 200.0
FNIRS_FS_FALLBACK = 10.0
EEG_WIN = 3.0     # s
EEG_STEP = 1.0    # s
FNIRS_WIN = 15.0  # s
LAG_SEC = 4.0     # s

# 图构建参数
USE_HBR = False           # 如需加入 HbR, 置 True 且提供 fnirs_hbr_windows
KNN_WITHIN = 12
KNN_CROSS = 8
GAMMA_XMODAL = 0.9
EPS = 1e-8


# ========= 通用小工具 =========

def _protected_log_var(x: float) -> float:
    x = max(float(x), 1e-8)
    return 0.5 * np.log(2 * np.pi * np.e * x)


def _welch_band_power(x: np.ndarray, fs: float, fmin: float, fmax: float) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    nperseg = min(len(x), max(int(fs * 2.0), 64))
    f, pxx = welch(x, fs=fs, nperseg=nperseg)
    m = (f >= fmin) & (f <= fmax)
    if not m.any():
        return float(np.mean(pxx)) if pxx.size > 0 else 0.0
    return float(np.sum(pxx[m]))


def _slope(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=float)
    t = t - t.mean()
    x = x - x.mean()
    denom = float(np.dot(t, t)) + EPS
    return float(np.dot(t, x) / denom)


def _hjorth_params(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        var0 = np.var(x) + EPS
        return float(var0), 0.0, 0.0

    dx = np.diff(x)
    ddx = np.diff(dx)

    var0 = np.var(x) + EPS
    var1 = np.var(dx) + EPS
    var2 = np.var(ddx) + EPS

    activity = var0
    mobility = np.sqrt(var1 / var0)
    complexity = np.sqrt(var2 / var1) / (mobility + EPS)
    return float(activity), float(mobility), float(complexity)


def _half_split(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return x, x
    mid = n // 2
    return x[:mid], x[mid:]


def align_feature_dim(feat_eeg: np.ndarray,
                      feat_fnirs: np.ndarray):
    """
    将 EEG / fNIRS 特征通过零填充对齐到相同维度。
    不改变已有信息，仅在特征空间增加模态专属维度。
    """
    Fe = feat_eeg.shape[1]
    Ff = feat_fnirs.shape[1]
    if Fe == Ff:
        return feat_eeg, feat_fnirs

    D = max(Fe, Ff)
    eeg_pad = np.pad(
        feat_eeg,
        pad_width=((0, 0), (0, D - Fe)),
        mode="constant",
        constant_values=0.0,
    )
    fnirs_pad = np.pad(
        feat_fnirs,
        pad_width=((0, 0), (0, D - Ff)),
        mode="constant",
        constant_values=0.0,
    )
    return eeg_pad, fnirs_pad


# ========= EEG 特征: 17 维 / 通道 =========

def extract_features_eeg(seg: np.ndarray, fs: float) -> np.ndarray:
    """
    EEG 通道特征 (17 维):
      - 5x DE: delta/theta/alpha/beta/gamma
      - 5x 相对功率
      - 2x 频带比值: theta/alpha, beta/alpha
      - 3x Hjorth: activity, mobility, complexity
      - 2x 动态: alpha_half_diff, pow_half_diff
    seg: [C, T]
    返回: [C, 17]
    """
    bands = [
        (0.5, 4),    # delta
        (4, 8),      # theta
        (8, 13),     # alpha
        (13, 30),    # beta
        (30, 45),    # gamma
    ]

    feats_all: List[List[float]] = []

    for ch in seg:
        ch = np.asarray(ch, dtype=float)

        # 整窗谱
        f, pxx = welch(ch, fs=fs, nperseg=min(len(ch), 256))
        mask_full = (f >= 0.5) & (f <= 45)
        full_power = float(np.sum(pxx[mask_full])) + EPS

        band_pows = []
        for fmin, fmax in bands:
            m = (f >= fmin) & (f <= fmax)
            if m.any():
                bp = float(np.sum(pxx[m]))
            else:
                bp = float(np.sum(pxx[mask_full])) / len(bands)
            band_pows.append(max(bp, EPS))

        # 1) DE
        de_feats = [_protected_log_var(bp) for bp in band_pows]

        # 2) 相对功率
        rel_feats = [bp / full_power for bp in band_pows]

        # 3) 频带比值
        theta_alpha = band_pows[1] / (band_pows[2] + EPS)
        beta_alpha = band_pows[3] / (band_pows[2] + EPS)

        # 4) Hjorth
        activity, mobility, complexity = _hjorth_params(ch)

        # 5) 动态特征 (前/后半)
        ch1, ch2 = _half_split(ch)
        alpha1 = _welch_band_power(ch1, fs, 8, 13)
        alpha2 = _welch_band_power(ch2, fs, 8, 13)
        alpha_half_diff = alpha2 - alpha1

        p1 = _welch_band_power(ch1, fs, 0.5, 45)
        p2 = _welch_band_power(ch2, fs, 0.5, 45)
        pow_half_diff = p2 - p1

        feats = (
            de_feats
            + rel_feats
            + [theta_alpha, beta_alpha]
            + [activity, mobility, complexity]
            + [alpha_half_diff, pow_half_diff]
        )
        assert len(feats) == 17
        feats_all.append(feats)

    return np.array(feats_all, dtype=float)


# ========= fNIRS 特征: 10 维 / 通道 =========

def extract_features_fnirs(seg: np.ndarray, fs: float) -> np.ndarray:
    """
    fNIRS 通道特征 (10 维):
      1) mean
      2) slope
      3) var
      4) band_pow (0.01-0.1 Hz)
      5) log(var)
      6) peak2peak
      7) diff_std (一阶差分 std)
      8) half_mean_diff (后半均值 - 前半均值)
      9) half_slope_diff (后半斜率 - 前半斜率)
     10) norm_peak2peak = peak2peak / (std + eps)
    seg: [C, T]
    返回: [C, 10]
    """
    feats_all: List[List[float]] = []

    for ch in seg:
        ch = np.asarray(ch, dtype=float)
        if ch.size == 0:
            feats_all.append([0.0] * 10)
            continue

        mean_val = float(np.mean(ch))
        slope_val = _slope(ch)
        var_val = float(np.var(ch)) + EPS
        bp = _welch_band_power(ch, fs, 0.01, 0.1)
        log_var = _protected_log_var(var_val)

        peak2peak = float(np.max(ch) - np.min(ch))
        diff_std = float(np.std(np.diff(ch))) if ch.size > 1 else 0.0

        ch1, ch2 = _half_split(ch)
        mean1 = float(np.mean(ch1)) if ch1.size > 0 else mean_val
        mean2 = float(np.mean(ch2)) if ch2.size > 0 else mean_val
        slope1 = _slope(ch1) if ch1.size > 1 else slope_val
        slope2 = _slope(ch2) if ch2.size > 1 else slope_val

        half_mean_diff = mean2 - mean1
        half_slope_diff = slope2 - slope1

        std_val = float(np.std(ch)) + EPS
        norm_peak2peak = peak2peak / std_val

        feats = [
            mean_val,
            slope_val,
            var_val,
            bp,
            log_var,
            peak2peak,
            diff_std,
            half_mean_diff,
            half_slope_diff,
            norm_peak2peak,
        ]
        assert len(feats) == 10
        feats_all.append(feats)

    return np.array(feats_all, dtype=float)


# ========= RBF + kNN 构图 =========

def _row_sigma(D: np.ndarray, k0: int) -> np.ndarray:
    if D.size == 0 or D.shape[1] == 0:
        return np.ones(D.shape[0], dtype=np.float32)
    S = np.sort(D, axis=1)
    idx = np.clip(k0, 0, S.shape[1] - 1)
    return S[:, idx] + EPS


def _keep_row_topk(M: np.ndarray, k: int) -> np.ndarray:
    if M.size == 0 or k <= 0:
        return np.zeros_like(M)
    k = min(k, M.shape[1])
    out = np.zeros_like(M)
    idx_part = np.argpartition(-M, kth=k-1, axis=1)[:, :k]
    row = np.arange(M.shape[0])[:, None]
    vals = M[row, idx_part]
    order = np.argsort(-vals, axis=1)
    idx = idx_part[row, order]
    out[row, idx] = M[row, idx]
    return out


def build_adj_knn_rbf(
    feat_eeg: np.ndarray,
    feat_fnirs: np.ndarray,
    k_within: int = KNN_WITHIN,
    k_cross: int = KNN_CROSS,
    gamma_xmodal: float = GAMMA_XMODAL,
    metric: str = "cosine",
    k0_sigma: int = 10,
    rescale_cross: bool = True,
    cross_q_percentile: int = 90,
    cross_p_target: float = 0.70,
) -> np.ndarray:
    """
    输入: 已对齐维度后的 EEG / fNIRS 特征
    输出: 单个窗口的邻接矩阵 A [N, N]
    """
    Ne, D = feat_eeg.shape
    Nf, D2 = feat_fnirs.shape
    assert D == D2, "EEG / fNIRS 特征维度必须一致 (已通过 align_feature_dim 保证)"

    Xe = feat_eeg.astype(np.float32)
    Xf = feat_fnirs.astype(np.float32)

    # 1) 分块距离
    Dee = cdist(Xe, Xe, metric=metric)
    Dff = cdist(Xf, Xf, metric=metric)
    Def = cdist(Xe, Xf, metric=metric)
    Dfe = Def.T

    sig_ee = _row_sigma(Dee, k0_sigma)
    sig_ff = _row_sigma(Dff, k0_sigma)
    sig_ef_e = _row_sigma(Def, k0_sigma)
    sig_ef_f = _row_sigma(Dfe, k0_sigma)

    # 2) RBF 相似
    We = np.exp(-(Dee ** 2) / (sig_ee[:, None] * sig_ee[None, :] + EPS))
    Wf = np.exp(-(Dff ** 2) / (sig_ff[:, None] * sig_ff[None, :] + EPS))
    Wef = np.exp(-(Def ** 2) / (sig_ef_e[:, None] * sig_ef_f[None, :] + EPS))
    Wfe = Wef.T

    # 3) 分块 kNN
    We = _keep_row_topk(We, k_within)
    Wf = _keep_row_topk(Wf, k_within)
    Wef = _keep_row_topk(Wef, k_cross)
    Wfe = _keep_row_topk(Wfe, k_cross)

    # 4) 跨模态重标定 + 衰减
    if rescale_cross:
        nz = Wef[Wef > 0]
        if nz.size > 0:
            p = np.percentile(nz, cross_q_percentile)
            scale = cross_p_target / max(p, 1e-6)
            scale = float(np.clip(scale, 0.5, 5.0))
            Wef = np.clip(Wef * scale, 0.0, 1.0)
            Wfe = np.clip(Wfe * scale, 0.0, 1.0)

    Wef *= float(gamma_xmodal)
    Wfe *= float(gamma_xmodal)

    # 5) 拼接整图
    N = Ne + Nf
    W = np.zeros((N, N), dtype=np.float32)
    W[:Ne, :Ne] = We
    W[Ne:, Ne:] = Wf
    W[:Ne, Ne:] = Wef
    W[Ne:, :Ne] = Wfe

    # 对称化 + 自环
    W = np.maximum(W, W.T)
    np.fill_diagonal(W, 1.0)

    return W


# ========= 读取对齐窗口（或 fallback） =========

def load_aligned_windows(subj_id: int) -> Dict:
    """
    尝试读取 subj_xx_aligned_windows.npz：
      - eeg_windows: [W, Ceeg, Teeg]
      - fnirs_hbo_windows: [W, Cnirs, Tnirs]
      - (可选) fnirs_hbr_windows
      - eeg_fs, fnirs_fs
    若不存在，则使用简易滑窗 + LAG 回退（注意这可能与标签不完全对齐）。
    """
    subj = f"subj_{subj_id:02d}"
    npz_path = os.path.join(SAVE_DIR, f"{subj}_aligned_windows.npz")
    if os.path.exists(npz_path):
        pack = np.load(npz_path, allow_pickle=True)
        return {k: pack[k] for k in pack.files}

    # fallback：从干净数据滑窗
    eeg_path = os.path.join(SAVE_DIR, f"{subj}_eeg_clean.npy")
    hbo_path = os.path.join(SAVE_DIR, f"{subj}_fnirs_clean_hbo.npy")
    if not (os.path.exists(eeg_path) and os.path.exists(hbo_path)):
        raise FileNotFoundError(f"No aligned_windows or clean signals for {subj}")

    eeg = np.load(eeg_path)   # [Ceeg, Teeg]
    hbo = np.load(hbo_path)   # [Cnirs, Tnirs]

    eeg_fs = EEG_FS_FALLBACK
    fnirs_fs = FNIRS_FS_FALLBACK

    eeg_wins, hbo_wins, t0_list = [], [], []

    t0 = 0.0
    while True:
        eeg_s = int(round(t0 * eeg_fs))
        eeg_e = int(round((t0 + EEG_WIN) * eeg_fs))
        fn_s = int(round((t0 + LAG_SEC) * fnirs_fs))
        fn_e = int(round((t0 + LAG_SEC + FNIRS_WIN) * fnirs_fs))

        if eeg_e > eeg.shape[1] or fn_e > hbo.shape[1]:
            break

        eeg_wins.append(eeg[:, eeg_s:eeg_e])
        hbo_wins.append(hbo[:, fn_s:fn_e])
        t0_list.append(t0)

        t0 += EEG_STEP

    if not eeg_wins:
        raise RuntimeError(f"Fallback windowing produced 0 windows for {subj}")

    return dict(
        eeg_windows=np.stack(eeg_wins, axis=0),
        fnirs_hbo_windows=np.stack(hbo_wins, axis=0),
        t0_list=np.array(t0_list, dtype=float),
        eeg_fs=float(eeg_fs),
        fnirs_fs=float(fnirs_fs),
        params=dict(
            EEG_WIN=EEG_WIN,
            EEG_STEP=EEG_STEP,
            FNIRS_WIN=FNIRS_WIN,
            LAG_SEC=LAG_SEC,
        ),
    )


# ========= 主流程：每个被试生成 adj/feat/meta =========

@dataclass
class SubjectSummary:
    subj: str
    W: int
    N: int
    F: int
    note: str


def run_one_subject(subj_id: int) -> SubjectSummary:
    subj = f"subj_{subj_id:02d}"
    pack = load_aligned_windows(subj_id)

    eeg_wins = pack["eeg_windows"]          # [W, Ceeg, Teeg]
    hbo_wins = pack["fnirs_hbo_windows"]    # [W, Cnirs, Tnirs]
    hbr_wins = pack.get("fnirs_hbr_windows", None)

    eeg_fs = float(pack["eeg_fs"])
    fnirs_fs = float(pack["fnirs_fs"])

    use_hbr = USE_HBR and (hbr_wins is not None)

    W = eeg_wins.shape[0]
    Ceeg = eeg_wins.shape[1]
    Cnirs_hbo = hbo_wins.shape[1]
    Cnirs_hbr = hbr_wins.shape[1] if use_hbr else 0

    # 1) 提取未标准化特征
    raw_eeg_list: List[np.ndarray] = []
    raw_fnirs_list: List[np.ndarray] = []

    for w in range(W):
        f_eeg = extract_features_eeg(eeg_wins[w], eeg_fs)       # [Ceeg, 17]
        f_hbo = extract_features_fnirs(hbo_wins[w], fnirs_fs)   # [Cnirs_hbo,10]

        if use_hbr:
            f_hbr = extract_features_fnirs(hbr_wins[w], fnirs_fs)
            f_fnirs = np.vstack([f_hbo, f_hbr])                 # HbO+HbR
        else:
            f_fnirs = f_hbo

        raw_eeg_list.append(f_eeg)
        raw_fnirs_list.append(f_fnirs)

    # 2) 被试内标准化（模态内）
    eeg_stack = np.vstack(raw_eeg_list)
    fnirs_stack = np.vstack(raw_fnirs_list)

    eeg_mean = eeg_stack.mean(axis=0)
    eeg_std = eeg_stack.std(axis=0) + EPS
    fnirs_mean = fnirs_stack.mean(axis=0)
    fnirs_std = fnirs_stack.std(axis=0) + EPS

    adjs = []
    feats = []

    for w in range(W):
        f_eeg = (raw_eeg_list[w] - eeg_mean) / eeg_std
        f_fnirs = (raw_fnirs_list[w] - fnirs_mean) / fnirs_std

        # 3) 特征维度对齐
        f_eeg_aligned, f_fnirs_aligned = align_feature_dim(f_eeg, f_fnirs)

        # 4) 构建邻接矩阵
        A = build_adj_knn_rbf(f_eeg_aligned, f_fnirs_aligned)

        Ne = f_eeg_aligned.shape[0]
        Nf = f_fnirs_aligned.shape[0]
        N = Ne + Nf

        # 5) 结构增强特征
        deg_total = A.sum(axis=1, keepdims=True)                     # [N,1]
        deg_within = np.zeros((N, 1), dtype=np.float32)
        deg_within[:Ne, 0] = A[:Ne, :Ne].sum(axis=1)
        deg_within[Ne:, 0] = A[Ne:, Ne:].sum(axis=1)

        modality_id = np.zeros((N, 1), dtype=np.float32)
        modality_id[:Ne, 0] = 1.0  # EEG=1, fNIRS=0

        feat_nodes = np.vstack([f_eeg_aligned, f_fnirs_aligned])
        feat_aug = np.concatenate(
            [feat_nodes, modality_id, deg_total, deg_within],
            axis=1
        )

        adjs.append(A.astype(np.float32))
        feats.append(feat_aug.astype(np.float32))

    adjs = np.stack(adjs, axis=0)   # [W, N, N]
    feats = np.stack(feats, axis=0) # [W, N, F_aug]

    # 6) 保存
    np.save(os.path.join(SAVE_DIR, f"{subj}_xmodal_adj.npy"), adjs)
    np.save(os.path.join(SAVE_DIR, f"{subj}_xmodal_feat.npy"), feats)

    n_eeg = Ceeg
    n_fnirs = Cnirs_hbo + (Cnirs_hbr if use_hbr else 0)
    np.savez(
        os.path.join(SAVE_DIR, f"{subj}_xmodal_meta.npz"),
        n_eeg=n_eeg,
        n_fnirs=n_fnirs,
    )

    note = "aligned_npz" if "params" in pack else "fallback"
    return SubjectSummary(subj=subj, W=W, N=adjs.shape[1], F=feats.shape[2], note=note)


# ========= 批处理 & 简单可视化 =========

def main():
    summaries: List[SubjectSummary] = []

    for sid in tqdm(range(1, 27), desc="Subjects"):
        try:
            s = run_one_subject(sid)
            summaries.append(s)
        except Exception as e:
            print(f"[ERROR] subj_{sid:02d}: {e}")
            summaries.append(
                SubjectSummary(
                    subj=f"subj_{sid:02d}", W=0, N=0, F=0, note=f"ERROR:{e}"
                )
            )

    # 打印汇总
    print("\n===== Cross-modal Graph Construction v2 Summary =====")
    for s in summaries:
        print(f"{s.subj}: W={s.W}, N={s.N}, F={s.F}, Note={s.note}")

    # 可选：查看一个被试的邻接矩阵结构
    demo = next((s for s in summaries if s.W > 0), None)
    if demo:
        subj = demo.subj
        adj = np.load(os.path.join(SAVE_DIR, f"{subj}_xmodal_adj.npy"))
        meta = np.load(os.path.join(SAVE_DIR, f"{subj}_xmodal_meta.npz"))
        n_eeg = int(meta["n_eeg"])
        N = adj.shape[1]
        n_fnirs = N - n_eeg

        print(f"\nDemo {subj}: adj shape={adj.shape}, n_eeg={n_eeg}, n_fnirs={n_fnirs}")
        if adj.shape[0] > 0:
            A0 = adj[0]
            plt.figure(figsize=(6, 5))
            ax = sns.heatmap(
                A0,
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                xticklabels=False,
                yticklabels=False,
            )
            ax.vlines(n_eeg, *ax.get_ylim(), colors="w", linewidth=2)
            ax.hlines(n_eeg, *ax.get_xlim(), colors="w", linewidth=2)
            ax.set_title(f"{subj} - window 0 adjacency (EEG / fNIRS)")
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()

