# config.py
# 全局配置与随机种子设置

import os
import torch
import numpy as np


# config.py
class Config:
    def __init__(self):
        # ……你已有的配置……
        self.seed = 42
        self.data_dir = r"D:\GNN\MI运动表现\数据处理"
        self.batch_size = 32
        self.hidden_dim = 128
        self.num_layers = 2
        self.dropout = 0.3
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.epochs = 200
        self.patience = 25

        # —— 训练/采样策略 ——
        self.use_sampler = True          # 训练集 1:1 过采样（只作用在 train dataloader）

        # —— 优化细节 ——
        self.label_smoothing = 0.05
        self.warmup_epochs   = 5
        self.min_lr          = 1e-5

        # —— 跨模态对比学习（CMC）——
        self.cmc_lambda = 0.2            # 联合损失权重（0 关闭）
        self.cmc_tau    = 0.2            # 温度

        # —— 轻量增广（可先开 0.10）——
        self.modality_drop_p = 0.10      # 训练时随机屏蔽少量 EEG/fNIRS 节点

        # —— 可选 FocalLoss（默认关闭）——
        self.use_focal   = False
        self.focal_gamma = 2.0

        # —— 数据划分（若你已有可保持）——
        self.train_ratio = 0.70
        self.val_ratio   = 0.15
        self.test_ratio  = 0.15

        self.subject_start = 1
        self.subject_end   = 26

# 单例
config = Config()




config = Config()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 初始化随机种子（导入 config 时自动执行）
set_seed(config.seed)

# 确保输出目录存在
os.makedirs(config.data_dir, exist_ok=True)
