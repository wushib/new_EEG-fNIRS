# dataset_xmodal.py
# 负责从磁盘加载跨模态图数据，构建 Dataset 和 collate_fn

import os
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from config import config


class CrossModalGraphDataset(Dataset):
    """
    从以下文件加载数据：
      subj_xx_xmodal_adj.npy   : [W, N, N]
      subj_xx_xmodal_feat.npy  : [W, N, F]
      subj_xx_labels.npy       : [W]
      subj_xx_xmodal_meta.npz  : {n_eeg}
    每个时间窗 -> 一个图样本。
    """
    def __init__(self,
                 data_dir: str = None,
                 subject_ids=None):
        self.data_dir = data_dir or config.data_dir
        if subject_ids is None:
            subject_ids = range(config.subject_start, config.subject_end + 1)
        self.subject_ids = list(subject_ids)

        self.graphs = []
        self.labels = []

        self._load_all()

    def _load_all(self):
        print("Loading cross-modal graphs from:", self.data_dir)

        for sid in tqdm(self.subject_ids, desc="Subjects"):
            subj = f"subj_{sid:02d}"
            adj_path = os.path.join(self.data_dir, f"{subj}_xmodal_adj.npy")
            feat_path = os.path.join(self.data_dir, f"{subj}_xmodal_feat.npy")
            label_path = os.path.join(self.data_dir, f"{subj}_labels.npy")
            meta_path = os.path.join(self.data_dir, f"{subj}_xmodal_meta.npz")

            if not (os.path.exists(adj_path)
                    and os.path.exists(feat_path)
                    and os.path.exists(label_path)
                    and os.path.exists(meta_path)):
                # print(f"[INFO] Missing files for {subj}, skip.")
                continue

            try:
                adjs = np.load(adj_path)      # [W, N, N]
                feats = np.load(feat_path)    # [W, N, F]
                labels = np.load(label_path)  # [W]
                meta = np.load(meta_path)
            except Exception as e:
                print(f"[WARN] {subj} load error: {e}")
                continue

            if adjs.ndim != 3 or feats.ndim != 3 or labels.ndim != 1:
                print(f"[WARN] {subj} shape mismatch, skip")
                continue
            if not (adjs.shape[0] == feats.shape[0] == labels.shape[0]):
                print(f"[WARN] {subj} W mismatch, skip")
                continue

            n_eeg = int(meta.get("n_eeg", 0))
            if n_eeg <= 0 or n_eeg > adjs.shape[1]:
                print(f"[WARN] {subj} invalid n_eeg={n_eeg}, skip")
                continue

            num_windows, num_nodes, feat_dim = feats.shape

            for w in range(num_windows):
                y = int(labels[w])
                if y not in (0, 1):
                    continue

                adj = adjs[w]
                x = feats[w]

                if not np.isfinite(adj).all() or not np.isfinite(x).all():
                    continue

                edge_index, edge_weight = self._adj_to_edge(adj)

                self.graphs.append({
                    "x": torch.from_numpy(x).float(),           # [N, F]
                    "edge_index": edge_index,                   # [2, E]
                    "edge_weight": edge_weight,                 # [E]
                    "y": torch.tensor(y, dtype=torch.long),     # []
                    "num_nodes": num_nodes,
                    "n_eeg": n_eeg
                })
                self.labels.append(y)

        if not self.graphs:
            raise RuntimeError("No valid graphs loaded. Check data_dir & files.")

        label_counts = Counter(self.labels)
        print(f"Loaded {len(self.graphs)} graphs total.")
        print(f"Class distribution: {dict(label_counts)}")

    @staticmethod
    def _adj_to_edge(adj: np.ndarray, tol: float = 1e-6):
        """
        邻接矩阵 -> 稀疏 edge_index/edge_weight
        """
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError("Adjacency must be square [N,N].")

        N = adj.shape[0]
        rows, cols = np.nonzero(np.abs(adj) > tol)
        if len(rows) == 0:
            # 无边时退化为自环
            rows = np.arange(N)
            cols = np.arange(N)
            weights = np.ones(N, dtype=np.float32)
        else:
            weights = adj[rows, cols].astype(np.float32)

        edge_index = torch.from_numpy(
            np.stack([rows, cols], axis=0).astype(np.int64)
        )
        edge_weight = torch.from_numpy(weights)
        return edge_index, edge_weight

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def collate_fn(batch):
    """
    将多个图打包成一个 batch:
      x        : [sum_N, F]
      edge_index, edge_weight
      batch    : [sum_N] 图索引
      y        : [B]
      eeg_mask : [sum_N] True=EEG节点
    """
    x_list, ei_list, ew_list = [], [], []
    y_list, batch_idx_list, eeg_mask_list = [], [], []

    node_offset = 0
    for i, g in enumerate(batch):
        x = g["x"]
        ei = g["edge_index"]
        ew = g["edge_weight"]
        y = g["y"]
        num_nodes = g["num_nodes"]
        n_eeg = g["n_eeg"]

        x_list.append(x)
        ei_list.append(ei + node_offset)
        ew_list.append(ew)
        y_list.append(y)

        batch_idx_list.append(torch.full((num_nodes,), i, dtype=torch.long))

        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[:n_eeg] = True
        eeg_mask_list.append(mask)

        node_offset += num_nodes

    x = torch.cat(x_list, dim=0)
    edge_index = torch.cat(ei_list, dim=1)
    edge_weight = torch.cat(ew_list, dim=0)
    batch_idx = torch.cat(batch_idx_list, dim=0)
    y = torch.stack(y_list, dim=0)
    eeg_mask = torch.cat(eeg_mask_list, dim=0)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "batch": batch_idx,
        "y": y,
        "eeg_mask": eeg_mask,
    }
