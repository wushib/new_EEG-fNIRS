import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if src.numel() == 0:
        return torch.zeros(dim_size, src.size(1), device=src.device)
    out = torch.zeros(dim_size, src.size(1), device=src.device)
    cnt = torch.zeros(dim_size, device=src.device)
    out.index_add_(0, index, src)
    ones = torch.ones_like(index, dtype=torch.float32, device=src.device)
    cnt.index_add_(0, index, ones)
    cnt = cnt.clamp(min=1.0).unsqueeze(-1)
    return out / cnt

class CoAttnGateGCN(nn.Module):
    """
    Modality-aware MLP -> Edge Modulation -> 2-Relation GCN(intra/cross) -> Co-Attn -> Gate -> Classifier
    + 可选 CMC 投影头（z_e, z_f）供 InfoNCE
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
        proj_dim: int = 128,           # CMC 投影维度
    ):
        super().__init__()
        assert num_layers >= 1
        self.dropout = dropout
        self.hidden_dim = hidden_dim

        # 1) 模态专属编码
        self.eeg_proj   = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.fnirs_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())

        # 2) 边权调制：依据 |h_i - h_j| 与原 edge_weight 出一个门控
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

        # 3) 2-关系 GCN（模态内 / 跨模态 两路）
        self.convs_intra = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.convs_cross = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.rel_gate    = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(num_layers)])
        self.bns         = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])

        # 4) 共注意 & 门控融合
        self.co_attn_fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2)
        )
        self.gate_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

        # 5) 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # 6) CMC 投影（训练时若 lambda>0 会用到）
        self.proj_e = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, proj_dim))
        self.proj_f = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, proj_dim))

    def _modulate_edges(self, h, edge_index, edge_weight):
        src, dst = edge_index
        diff = torch.abs(h[src] - h[dst])            # [E,H]
        ew = edge_weight.unsqueeze(-1)               # [E,1]
        g = torch.sigmoid(self.edge_mlp(torch.cat([diff, ew], dim=1))).squeeze(-1)  # [E]
        return edge_weight * g.clamp(0.2, 1.2)       # 适度约束，防止全抹或爆

    def forward(self, x, edge_index, edge_weight, batch, eeg_mask):
        eeg_mask = eeg_mask.bool()
        non_eeg_mask = ~eeg_mask

        # 1) 模态专属编码
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        if eeg_mask.any():     h[eeg_mask] = self.eeg_proj(x[eeg_mask])
        if non_eeg_mask.any(): h[non_eeg_mask] = self.fnirs_proj(x[non_eeg_mask])

        # 2) 边权调制
        ew_hat = self._modulate_edges(h, edge_index, edge_weight)

        # 2-关系拆边
        src, dst = edge_index
        intra_mask = (eeg_mask[src] == eeg_mask[dst])
        cross_mask = ~intra_mask
        ei_intra, ew_intra = edge_index[:, intra_mask], ew_hat[intra_mask]
        ei_cross, ew_cross = edge_index[:, cross_mask], ew_hat[cross_mask]

        # 3) 逐层 2-关系卷积 + 门控融合
        for l in range(len(self.convs_intra)):
            h_intra = self.convs_intra[l](h, ei_intra, ew_intra)
            h_cross = self.convs_cross[l](h, ei_cross, ew_cross)
            alpha   = torch.sigmoid(self.rel_gate[l])      # ∈(0,1)
            h = h_intra + alpha * h_cross
            h = self.bns[l](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # 4) 池化：模态池化 + 全图池化
        h_global = global_mean_pool(h, batch)  # [B,H]
        if eeg_mask.any():
            h_e = scatter_mean(h[eeg_mask], batch[eeg_mask], h_global.size(0))
        else:
            h_e = h_global
        if non_eeg_mask.any():
            h_f = scatter_mean(h[non_eeg_mask], batch[non_eeg_mask], h_global.size(0))
        else:
            h_f = h_global

        # 共注意
        co_in = torch.cat([h_e, h_f, h_e * h_f], dim=-1)
        alpha = torch.sigmoid(self.co_attn_fc(co_in))
        alpha = alpha / (alpha.sum(dim=-1, keepdim=True) + 1e-6)
        h_cross = alpha[:, :1] * h_e + alpha[:, 1:] * h_f

        # 门控融合
        g = torch.sigmoid(self.gate_fc(torch.cat([h_cross, h_global], dim=-1)))
        h_final = g * h_cross + (1 - g) * h_global

        # 分类 & CMC 投影
        logits = self.classifier(h_final)
        z_e = F.normalize(self.proj_e(h_e), dim=-1)
        z_f = F.normalize(self.proj_f(h_f), dim=-1)

        return {"logits": logits, "z_e": z_e, "z_f": z_f}
