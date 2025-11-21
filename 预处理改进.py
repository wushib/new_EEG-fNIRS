import os
import numpy as np
import scipy.io
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from typing import Tuple, Dict

from scipy.signal import butter, filtfilt, sosfiltfilt, iirnotch, resample

# === 路径配置（保持你的原样） ===
EEG_BASE = r"E:\数据\Finally\EEG_01-26_MATLAB"
FNIRS_BASE = r"E:\数据\Finally\NIRS_01-26_MATLAB"
SAVE_DIR  = r"D:\GNN\MI运动表现\数据处理"
os.makedirs(SAVE_DIR, exist_ok=True)


# === 对齐窗口与时延参数（可按需修改） ===
EEG_WIN  = 3.0    # s  EEG 窗长
EEG_STEP = 1.0    # s  EEG 步长
FNIRS_WIN  = 15.0 # s  fNIRS 窗长（>=10 s）
FNIRS_STEP = 1.0  # s  fNIRS 步长
LAG_SEC    = 4.0  # s  EEG -> fNIRS 正向时延

TARGET_EEG_FS   = 200.0   # 统一 EEG 采样率
MAX_NAN_RATIO   = 0.30    # 单通道 NaN 占比阈值；超过则剔除
MIN_VAR_THRESH  = 1e-8    # 低方差（近常值）阈值；低于则剔除
USE_NOTCH_50HZ  = True    # EEG 是否做 50 Hz 陷波

@dataclass
class SubjectResult:
    subj: str
    eeg_shape: Tuple[int, int] = (0, 0)
    hbo_shape: Tuple[int, int] = (0, 0)
    hbr_shape: Tuple[int, int] = (0, 0)
    kept_eeg: int = 0
    kept_hbo: int = 0
    kept_hbr: int = 0
    n_windows: int = 0
    ok: bool = False
    msg: str = ""

# ---------------- 工具函数 ----------------

def interp_nans_1d(x: np.ndarray) -> np.ndarray:
    """对一维向量做线性插值填补 NaN；若全 NaN 则返回 0 向量"""
    idx = np.flatnonzero(np.isfinite(x))
    if idx.size == 0:
        return np.zeros_like(x)
    return np.interp(np.arange(len(x)), idx, x[idx])

def clean_nans_and_prune(data: np.ndarray,
                         max_nan_ratio: float = MAX_NAN_RATIO,
                         min_var_thresh: float = MIN_VAR_THRESH) -> Tuple[np.ndarray, np.ndarray]:
    """
    逐通道处理 NaN 与低方差。
    返回: (clean_data, kept_mask)
    """
    C, T = data.shape
    kept = np.ones(C, dtype=bool)
    out = data.copy()

    for c in range(C):
        x = out[c]
        nan_ratio = np.isnan(x).mean()
        if nan_ratio > 0 and nan_ratio <= max_nan_ratio:
            out[c] = interp_nans_1d(x)
        elif nan_ratio > max_nan_ratio:
            kept[c] = False
            continue  # 先标记剔除，稍后统一裁剪

        # 低方差通道（含“几乎常值”）剔除
        if np.isfinite(out[c]).sum() < int(0.7 * T) or np.nanvar(out[c]) < min_var_thresh:
            kept[c] = False

    out = out[kept]
    return out, kept

def bandpass_eeg(data: np.ndarray, fs: float,
                 l_freq: float = 0.5, h_freq: float = 50.0,
                 use_notch: bool = True) -> np.ndarray:
    """EEG：先可选 50Hz 陷波，再 0.5–50Hz 带通"""
    if use_notch:
        nyq = fs / 2.0
        w0 = 50.0 / nyq
        b, a = iirnotch(w0, Q=30.0)
        data = filtfilt(b, a, data, axis=1)
    nyq = fs / 2.0
    b, a = butter(N=4, Wn=[l_freq/nyq, h_freq/nyq], btype='band')
    return filtfilt(b, a, data, axis=1)

def bandpass_fnirs_sos(data: np.ndarray, fs: float,
                       l_freq: float = 0.01, h_freq: float = 0.1) -> np.ndarray:
    """fNIRS：0.01–0.1 Hz SOS 带通"""
    nyq = fs / 2.0
    sos = butter(N=2, Wn=[l_freq/nyq, h_freq/nyq], btype='band', output='sos')
    return sosfiltfilt(sos, data, axis=1)

def baseline_correction_auto(data: np.ndarray, fs: float,
                             tmin_sec: float = None, tmax_sec: float = None) -> np.ndarray:
    """
    改进版 baseline：若未指定区间，则取“前 10%”。
    若索引非法自动兜底。
    """
    C, T = data.shape
    if tmin_sec is None or tmax_sec is None:
        start, end = 0, max(1, int(0.1 * T))
    else:
        start = int(max(0, tmin_sec * fs))
        end   = int(min(T, tmax_sec * fs))
        if end <= start:
            start, end = 0, max(1, int(0.1 * T))
    base = np.nanmean(data[:, start:end], axis=1, keepdims=True)
    return data - base

def robust_resample(data: np.ndarray, src_fs: float, tgt_fs: float) -> np.ndarray:
    """稳健重采样到目标采样率"""
    if abs(src_fs - tgt_fs) < 1e-6:
        return data
    T_new = int(round(data.shape[1] * tgt_fs / src_fs))
    return resample(data, T_new, axis=1)

def health(msg: str, x: np.ndarray):
    print(f"[{msg}] shape={x.shape}, "
          f"NaN={np.isnan(x).any()}, "
          f"range=[{np.nanmin(x):.4g}, {np.nanmax(x):.4g}], "
          f"mean={np.nanmean(x):.4g}, var={np.nanvar(x):.4g}")

def make_aligned_windows(eeg: np.ndarray, eeg_fs: float,
                         hbo: np.ndarray, hbr: np.ndarray, fnirs_fs: float,
                         eeg_win: float = EEG_WIN, eeg_step: float = EEG_STEP,
                         fnirs_win: float = FNIRS_WIN, lag_sec: float = LAG_SEC) -> Dict[str, np.ndarray]:
    """
    生成“同一时间锚点”下的对齐滑窗：
      EEG 窗 [t0, t0+eeg_win]
      fNIRS 窗 [t0+lag_sec, t0+lag_sec+fnirs_win]
    只输出两个窗口都在范围内的时间锚点。
    """
    T_eeg = eeg.shape[1] / eeg_fs
    T_fn  = hbo.shape[1] / fnirs_fs  # 与 hbr 等长
    t0_list = []
    eeg_wins, hbo_wins, hbr_wins = [], [], []

    t0 = 0.0
    while True:
        eeg_s = int(round(t0 * eeg_fs))
        eeg_e = int(round((t0 + eeg_win) * eeg_fs))
        fn_s  = int(round((t0 + lag_sec) * fnirs_fs))
        fn_e  = int(round((t0 + lag_sec + fnirs_win) * fnirs_fs))

        if eeg_e > eeg.shape[1] or fn_e > hbo.shape[1]:
            break  # 超界，停止

        eeg_wins.append(eeg[:, eeg_s:eeg_e])
        hbo_wins.append(hbo[:, fn_s:fn_e])
        hbr_wins.append(hbr[:, fn_s:fn_e])
        t0_list.append(t0)

        t0 += EEG_STEP  # 以 EEG 的步长为节拍

    if len(t0_list) == 0:
        return dict(eeg_windows=np.empty((0,)), fnirs_hbo_windows=np.empty((0,)),
                    fnirs_hbr_windows=np.empty((0,)), t0_list=np.array([]))

    return dict(
        eeg_windows=np.stack(eeg_wins, axis=0),
        fnirs_hbo_windows=np.stack(hbo_wins, axis=0),
        fnirs_hbr_windows=np.stack(hbr_wins, axis=0),
        t0_list=np.array(t0_list, dtype=float),
    )

# ---------------- 主流程 ----------------

results = []
for i in tqdm(range(1, 27)):
    subj = f"subj_{i:02d}"
    try:
        eeg_mat_path   = os.path.join(EEG_BASE,  f"VP{i:03d}-EEG",  "cnt_wg.mat")
        fnirs_mat_path = os.path.join(FNIRS_BASE,f"VP{i:03d}-NIRS", "cnt_wg.mat")

        if not (os.path.exists(eeg_mat_path) and os.path.exists(fnirs_mat_path)):
            results.append(SubjectResult(subj=subj, msg="Missing .mat files").__dict__)
            continue

        # ---------- 读取 EEG ----------
        eeg_mat = scipy.io.loadmat(eeg_mat_path, struct_as_record=False, squeeze_me=True)
        eeg_cnt = eeg_mat["cnt_wg"]
        eeg_data = eeg_cnt.x.T           # [C, T]
        eeg_fs   = float(eeg_cnt.fs)

        # 中位数去趋势/均值参考（CAR）
        eeg_data = eeg_data - np.nanmean(eeg_data, axis=0, keepdims=True)

        # 先清 NaN 再滤波
        eeg_data, eeg_keep_mask = clean_nans_and_prune(eeg_data, MAX_NAN_RATIO, MIN_VAR_THRESH)
        health(f"{subj} EEG after clean", eeg_data)

        eeg_data = bandpass_eeg(eeg_data, eeg_fs, use_notch=USE_NOTCH_50HZ)
        eeg_data = robust_resample(eeg_data, eeg_fs, TARGET_EEG_FS)
        eeg_fs_new = TARGET_EEG_FS
        health(f"{subj} EEG filtered+resampled", eeg_data)

        # ---------- 读取 fNIRS ----------
        fnirs_mat = scipy.io.loadmat(fnirs_mat_path, struct_as_record=False, squeeze_me=True)
        fn_cnt = fnirs_mat["cnt_wg"]

        fnirs_fs = float(fn_cnt.oxy.fs)
        hbo_data = fn_cnt.oxy.x.T   # [C, T]
        hbr_data = fn_cnt.deoxy.x.T

        # 清 NaN -> 剔除坏通道 -> 滤波 -> baseline
        hbo_data, hbo_keep = clean_nans_and_prune(hbo_data, MAX_NAN_RATIO, MIN_VAR_THRESH)
        hbr_data, hbr_keep = clean_nans_and_prune(hbr_data, MAX_NAN_RATIO, MIN_VAR_THRESH)
        health(f"{subj} HbO after clean", hbo_data)
        health(f"{subj} HbR after clean", hbr_data)

        hbo_data = bandpass_fnirs_sos(hbo_data, fnirs_fs)
        hbr_data = bandpass_fnirs_sos(hbr_data, fnirs_fs)
        health(f"{subj} HbO filtered", hbo_data)
        health(f"{subj} HbR filtered", hbr_data)

        hbo_data = baseline_correction_auto(hbo_data, fnirs_fs)
        hbr_data = baseline_correction_auto(hbr_data, fnirs_fs)
        health(f"{subj} HbO baseline", hbo_data)
        health(f"{subj} HbR baseline", hbr_data)

        # ---------- 保存清洗后的整段 ----------
        np.save(os.path.join(SAVE_DIR, f"{subj}_eeg_clean.npy"),  eeg_data)
        np.save(os.path.join(SAVE_DIR, f"{subj}_fnirs_clean_hbo.npy"), hbo_data)
        np.save(os.path.join(SAVE_DIR, f"{subj}_fnirs_clean_hbr.npy"), hbr_data)

        # ---------- 生成“带时延”的对齐滑窗 ----------
        pack = make_aligned_windows(
            eeg=eeg_data, eeg_fs=eeg_fs_new,
            hbo=hbo_data, hbr=hbr_data, fnirs_fs=fnirs_fs,
            eeg_win=EEG_WIN, eeg_step=EEG_STEP,
            fnirs_win=FNIRS_WIN, lag_sec=LAG_SEC
        )

        # 可能某些极短记录没有任何对齐窗
        n_windows = 0 if pack["t0_list"].size == 0 else pack["eeg_windows"].shape[0]

        # 附加元信息并保存 .npz（建图脚本直接读取）
        pack.update(dict(
            eeg_fs=eeg_fs_new,
            fnirs_fs=fnirs_fs,
            eeg_channels_kept=np.flatnonzero(eeg_keep_mask).astype(int),
            fnirs_hbo_channels_kept=np.flatnonzero(hbo_keep).astype(int),
            fnirs_hbr_channels_kept=np.flatnonzero(hbr_keep).astype(int),
            params=dict(
                EEG_WIN=EEG_WIN, EEG_STEP=EEG_STEP,
                FNIRS_WIN=FNIRS_WIN, FNIRS_STEP=FNIRS_STEP,
                LAG_SEC=LAG_SEC
            )
        ))

        np.savez_compressed(os.path.join(SAVE_DIR, f"{subj}_aligned_windows.npz"), **pack)

        # ---------- 汇总日志 ----------
        results.append(SubjectResult(
            subj=subj,
            eeg_shape=tuple(eeg_data.shape),
            hbo_shape=tuple(hbo_data.shape),
            hbr_shape=tuple(hbr_data.shape),
            kept_eeg=int(np.sum(eeg_keep_mask)),
            kept_hbo=int(np.sum(hbo_keep)),
            kept_hbr=int(np.sum(hbr_keep)),
            n_windows=int(n_windows),
            ok=True, msg="OK"
        ).__dict__)

    except Exception as e:
        results.append(SubjectResult(subj=subj, ok=False, msg=str(e)).__dict__)
        print(f"[ERROR] {subj}: {e}")

# 保存日志
df = pd.DataFrame(results)
df.to_csv(os.path.join(SAVE_DIR, "preprocess_log.csv"), index=False)
print("✅ 预处理完成。最后 10 条：")
print(df.tail(10))
