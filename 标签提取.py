# -*- coding: utf-8 -*-
"""
根据事件标记生成每窗标签（0=BL, 1=WG），并保存为:
  subj_xx_labels.npy              # [W] int64，一维，0/1
  subj_xx_window_starts_sec.npy   # [W] float，窗起始秒
保持与训练脚本一致的输出文件名，不再产生 object/字符串标签。
"""

import os
import re
import numpy as np
import scipy.io

# ==== 路径（与你的训练/跨模态脚本一致）====
SAVE_DIR = r"D:\GNN\MI运动表现\数据处理"
MRK_BASE = r"E:\数据\Finally\EEG_01-26_MATLAB"

# ==== 回退默认（当没有 aligned_windows.npz 时启用）====
DEF_EEG_STEP   = 1.0   # s
DEF_EEG_WIN    = 3.0   # s
DEF_FNIRS_WIN  = 15.0  # s
DEF_LAG_SEC    = 4.0   # s

# 两类名称仅用于日志，不参与保存
CLASS_NAMES = ['BL', 'WG']  # 0->BL, 1->WG


# ---------------- 辅助函数 ----------------
def _load_markers(mrk_path):
    """读取 mrk_wg.mat，返回 (times_sec[N], class_idx[N])，class_idx∈{0(BL),1(WG)}"""
    m = scipy.io.loadmat(mrk_path, struct_as_record=False, squeeze_me=True)

    # 兼容多种字段命名
    if 'mrk' in m:
        s = m['mrk']
        times = getattr(s, 'time', getattr(s, 'pos', None))
        y = getattr(s, 'y', None)
    elif 'mrk_wg' in m:
        s = m['mrk_wg']
        times = getattr(s, 'time', getattr(s, 'pos', None))
        y = getattr(s, 'y', None)
    else:
        times = m.get('time', m.get('pos', None))
        y = m.get('y', None)

    if times is None or y is None:
        raise KeyError(f"未在 {mrk_path} 找到事件 time/pos 和 y")

    times = np.asarray(times).squeeze()
    # 毫秒→秒 自动判断
    times_sec = times.astype(float) / 1000.0 if times.max() > 1e3 else times.astype(float)

    y = np.asarray(y)
    # 统一为 2×N one-hot（0 行=BL, 1 行=WG）
    if y.ndim == 1:
        y = y.astype(int).ravel()
        cls = np.clip(y, 0, 1)
        onehot = np.zeros((2, y.size), dtype=int)
        for i, c in enumerate(cls):
            onehot[c, i] = 1
        y = onehot
    elif y.shape[0] != 2 and y.shape[1] == 2:
        y = y.T
    elif y.shape[0] != 2 and y.shape[1] != 2:
        # 多类：仅取前两类作为 BL/WG
        if y.shape[0] < y.shape[1]:
            y = y[:2, :]
        else:
            y = y[:, :2].T

    if y.shape[0] != 2:
        raise ValueError(f"y 形状异常：{y.shape} (期望 2×N)")

    N = min(times_sec.size, y.shape[1])
    times_sec = times_sec[:N]
    y = y[:, :N]

    cls_idx = np.argmax(y, axis=0).astype(int)  # 0→BL, 1→WG
    return times_sec, cls_idx


def _dedup_events(times_sec, cls_idx, min_sep=0.5):
    """同类事件在 min_sep 秒内仅保留第一个。"""
    order = np.argsort(times_sec)
    t = times_sec[order]
    c = cls_idx[order]
    keep = [0]
    for i in range(1, len(t)):
        if c[i] != c[keep[-1]] or (t[i] - t[keep[-1]]) >= min_sep:
            keep.append(i)
    keep = np.array(keep, dtype=int)
    return t[keep], c[keep]


def _load_window_anchors(subj_id: int):
    """
    优先读取 aligned_windows 的 t0_list 与参数；否则回退到旧图数量与默认参数。
    返回: t0_list, EEG_STEP, EEG_WIN, FNIRS_WIN, LAG_SEC
    """
    sid = f"{subj_id:02d}"
    tag = f"subj_{sid}"

    npz_path = os.path.join(SAVE_DIR, f"{tag}_aligned_windows.npz")
    if os.path.exists(npz_path):
        pack = np.load(npz_path, allow_pickle=True)
        t0_list = pack["t0_list"].astype(float)
        par = pack.get("params", None)
        if par is not None:
            par = dict(par.tolist()) if isinstance(par, np.ndarray) else dict(par)
            EEG_WIN   = float(par.get("EEG_WIN", DEF_EEG_WIN))
            EEG_STEP  = float(par.get("EEG_STEP", DEF_EEG_STEP))
            FNIRS_WIN = float(par.get("FNIRS_WIN", DEF_FNIRS_WIN))
            LAG_SEC   = float(par.get("LAG_SEC", DEF_LAG_SEC))
        else:
            EEG_WIN, EEG_STEP, FNIRS_WIN, LAG_SEC = DEF_EEG_WIN, DEF_EEG_STEP, DEF_FNIRS_WIN, DEF_LAG_SEC
        return t0_list, EEG_STEP, EEG_WIN, FNIRS_WIN, LAG_SEC

    # 回退：用图文件的 W 来生成锚点（步长默认 1s）
    p_adj_new = os.path.join(SAVE_DIR, f"{tag}_xmodal_adj.npy")
    p_adj_old = os.path.join(SAVE_DIR, f"{tag}_feature_fused_adj.npy")
    if os.path.exists(p_adj_new):
        W = np.load(p_adj_new).shape[0]
    elif os.path.exists(p_adj_old):
        W = np.load(p_adj_old).shape[0]
    else:
        raise FileNotFoundError(f"{tag}: 找不到 aligned_windows.npz 或 *_adj.npy")

    t0_list = np.arange(W) * DEF_EEG_STEP
    return t0_list, DEF_EEG_STEP, DEF_EEG_WIN, DEF_FNIRS_WIN, DEF_LAG_SEC


def _overlap(a0, a1, b0, b1):
    """两区间重叠长度"""
    return max(0.0, min(a1, b1) - max(a0, b0))


def _label_for_window(
    t0, EEG_WIN, FNIRS_WIN, LAG_SEC, evt_times, evt_cls,
    # 任务历时：EEG 常用 0.5~4.5 s；fNIRS 常用 2~12 s
    eeg_epoch=(0.5, 4.5),      # 相对事件时刻 t 的 EEG 有效段
    fnirs_epoch=(2.0, 12.0),   # 相对事件时刻 t 的 fNIRS 有效段
    eeg_thr=0.5,               # 与 EEG 滑窗的重叠占比阈值
    fnirs_thr=0.5,             # 与 fNIRS 滑窗的重叠占比阈值
    rule='or'                  # 'or' 或 'and'
):
    """
    返回整数标签：1=WG，0=BL
    """
    EEG_INT   = (t0, t0 + EEG_WIN)
    FNIRS_INT = (t0 + LAG_SEC, t0 + LAG_SEC + FNIRS_WIN)

    eeg_need   = eeg_thr   * EEG_WIN
    fnirs_need = fnirs_thr * FNIRS_WIN

    is_wg = False
    for t, c in zip(evt_times, evt_cls):
        if c != 1:  # 只匹配 WG 事件
            continue
        eeg_evt   = (t + eeg_epoch[0],   t + eeg_epoch[1])
        fnirs_evt = (t + fnirs_epoch[0], t + fnirs_epoch[1])
        ok_eeg   = _overlap(*EEG_INT,   *eeg_evt)   >= eeg_need
        ok_fnirs = _overlap(*FNIRS_INT, *fnirs_evt) >= fnirs_need
        if (rule == 'or' and (ok_eeg or ok_fnirs)) or (rule == 'and' and (ok_eeg and ok_fnirs)):
            is_wg = True
            break
    return 1 if is_wg else 0


# ---------------- 主逻辑 ----------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    for subj_id in range(1, 27):
        sid = f"{subj_id:02d}"
        tag = f"subj_{sid}"

        # 1) 读取窗口锚点与参数（优先 aligned_windows）
        try:
            t0_list, EEG_STEP, EEG_WIN, FNIRS_WIN, LAG_SEC = _load_window_anchors(subj_id)
        except Exception as e:
            print(f"❌ {tag}: 无窗口信息，跳过 | {e}")
            continue
        W = len(t0_list)

        # 2) 读取事件标记
        mrk_path = os.path.join(MRK_BASE, f"VP{subj_id:03d}-EEG", "mrk_wg.mat")
        if not os.path.exists(mrk_path):
            print(f"❌ {tag}: 缺少 {mrk_path}，跳过")
            continue

        try:
            evt_times, evt_cls = _load_markers(mrk_path)  # times(sec), class_idx(0/1)
        except Exception as e:
            print(f"❌ {tag}: 读取标记失败：{e}")
            continue
        evt_times, evt_cls = _dedup_events(evt_times, evt_cls, min_sep=0.5)

        # 3) 逐窗打标（输出整数 0/1）
        labels = np.zeros(W, dtype=np.int64)
        for i, t0 in enumerate(t0_list):
            labels[i] = _label_for_window(
                t0, EEG_WIN, FNIRS_WIN, LAG_SEC, evt_times, evt_cls,
                eeg_epoch=(0.5, 4.5), fnirs_epoch=(2.0, 12.0),
                eeg_thr=0.5, fnirs_thr=0.5, rule='or'
            )

        # 4) 保存（文件名保持不变；labels 为纯 int64；starts 为 float）
        label_path = os.path.join(SAVE_DIR, f"{tag}_labels.npy")
        np.save(label_path, labels.astype(np.int64))  # 不再使用 dtype=object

        starts_path = os.path.join(SAVE_DIR, f"{tag}_window_starts_sec.npy")
        np.save(starts_path, t0_list.astype(float))

        wg_cnt = int((labels == 1).sum())
        print(f"✅ {tag}: 保存 {W} 个标签（WG={wg_cnt}, BL={W-wg_cnt}） | "
              f"EEG_WIN={EEG_WIN}, STEP={EEG_STEP}, FNIRS_WIN={FNIRS_WIN}, LAG={LAG_SEC}")

if __name__ == "__main__":
    main()
