import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

# 可选绘图
try:
    import matplotlib.pyplot as plt, seaborn as sns
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False

from config import config
from dataset_xmodal import CrossModalGraphDataset, collate_fn
from model_coattn_gate_gcn import CoAttnGateGCN

# --------- 对比学习（InfoNCE） ----------
def info_nce(z1, z2, tau=0.2):
    # z1,z2: [B,D]，同索引为正对，其余为负样本
    sim = torch.matmul(z1, z2.t()) / tau  # [B,B]
    pos = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(sim, pos) + F.cross_entropy(sim.t(), pos))

# --------- 可选 FocalLoss ----------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha  # tensor([a0,a1]) or None
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction="none")
        pt = torch.softmax(logits, dim=1).gather(1, target.view(-1,1)).squeeze(1)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, all_labels):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # ============ 损失：过采样→等权；否则→轻量类权 ============
        self.use_sampler = bool(getattr(config, "use_sampler", True))
        label_smoothing = float(getattr(config, "label_smoothing", 0.05))
        use_focal = bool(getattr(config, "use_focal", False))
        focal_gamma = float(getattr(config, "focal_gamma", 2.0))

        if self.use_sampler:
            base_ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self.ce_or_focal = base_ce if not use_focal else FocalLoss(gamma=focal_gamma)
            print("[Train] sampler=1:1, CE(label_smoothing=%.2f)%s" %
                  (label_smoothing, "" if not use_focal else " + Focal"))
        else:
            all_labels = np.asarray(all_labels)
            counts = np.bincount(all_labels, minlength=2)
            eff_ratio = float(min(counts[0] / (counts[1] + 1e-6), 2.0))
            alpha = torch.tensor([1.0, eff_ratio], dtype=torch.float32, device=self.device)
            base_ce = nn.CrossEntropyLoss(weight=alpha, label_smoothing=label_smoothing)
            self.ce_or_focal = base_ce if not use_focal else FocalLoss(alpha=alpha, gamma=focal_gamma)
            print("[Train] light class weights:", alpha.detach().cpu().numpy(),
                  "| CE(label_smoothing=%.2f)%s" % (label_smoothing, "" if not use_focal else " + Focal"))

        # CMC 超参
        self.cmc_lambda = float(getattr(config, "cmc_lambda", 0.2))
        self.cmc_tau    = float(getattr(config, "cmc_tau", 0.2))
        # Modality Dropout
        self.modality_drop_p = float(getattr(config, "modality_drop_p", 0.10))

        # 优化器 & 调度
        lr = getattr(config, "learning_rate", getattr(config, "lr", 1e-3))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=config.weight_decay)
        warmup_epochs = int(getattr(config, "warmup_epochs", 5))
        min_lr = float(getattr(config, "min_lr", 1e-5))
        if warmup_epochs > 0 and warmup_epochs < config.epochs:
            self._sched = SequentialLR(
                self.optimizer,
                schedulers=[
                    LambdaLR(self.optimizer, lr_lambda=lambda e: (e + 1) / warmup_epochs),
                    CosineAnnealingLR(self.optimizer, T_max=config.epochs - warmup_epochs, eta_min=min_lr),
                ],
                milestones=[warmup_epochs],
            )
        else:
            self._sched = CosineAnnealingLR(self.optimizer, T_max=config.epochs, eta_min=min_lr)

        # 记录
        self.best_val_metric = 0.0
        self.best_state = None
        self.best_thresh = 0.5
        self.patience_counter = 0
        self.train_losses, self.val_losses, self.val_accs, self.val_f1s = [], [], [], []

    # ------------ 训练一个 epoch -------------
    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss, steps = 0.0, 0
        pbar = tqdm(self.train_loader, ncols=100, desc=f"Epoch {epoch+1}/{config.epochs}")

        for batch in pbar:
            x = batch["x"].to(self.device)
            ei = batch["edge_index"].to(self.device)
            ew = batch["edge_weight"].to(self.device)
            b  = batch["batch"].to(self.device)
            y  = batch["y"].to(self.device)
            mask = batch["eeg_mask"].to(self.device)

            # Modality Dropout（训练期）：随机遮蔽小部分 EEG 或 fNIRS 节点
            p = self.modality_drop_p
            if p > 0:
                with torch.no_grad():
                    drop_eeg = ((torch.rand_like(mask.float()) < p) & mask.bool())
                    drop_fn  = ((torch.rand_like(mask.float()) < p) & (~mask.bool()))
                if drop_eeg.any() or drop_fn.any():
                    x = x.clone()
                    x[drop_eeg | drop_fn] = 0.0

            self.optimizer.zero_grad()
            out = self.model(x, ei, ew, b, mask)
            logits = out["logits"]
            loss = self.ce_or_focal(logits, y)

            if self.cmc_lambda > 0:
                loss = loss + self.cmc_lambda * info_nce(out["z_e"], out["z_f"], tau=self.cmc_tau)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            steps += 1
            pbar.set_postfix({"Train Loss": f"{total_loss / steps:.4f}"})

        return total_loss / max(steps, 1)

    # ------------ 验证 -------------
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        all_probs, all_preds05, all_labels = [], [], []

        for batch in self.val_loader:
            x = batch["x"].to(self.device)
            ei = batch["edge_index"].to(self.device)
            ew = batch["edge_weight"].to(self.device)
            b  = batch["batch"].to(self.device)
            y  = batch["y"].to(self.device)
            mask = batch["eeg_mask"].to(self.device)

            out = self.model(x, ei, ew, b, mask)
            logits = out["logits"]
            loss = self.ce_or_focal(logits, y)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.cpu().numpy())
            all_preds05.append((probs >= 0.5).astype(int))

        all_probs  = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        preds05    = np.concatenate(all_preds05)

        acc = accuracy_score(all_labels, preds05)
        f1m = f1_score(all_labels, preds05, average="macro")

        # 阈值搜索优化 Macro-F1
        ts = np.linspace(0.1, 0.9, 81)
        best_t, best_score = 0.5, -1.0
        for t in ts:
            pred = (all_probs >= t).astype(int)
            score = f1_score(all_labels, pred, average="macro")
            if score > best_score:
                best_score, best_t = score, t

        avg_loss = total_loss / max(len(self.val_loader), 1)
        return avg_loss, acc, f1m, best_t, best_score

    # ------------ 主训练循环 -------------
    def fit(self):
        print("Starting training (use_sampler =", self.use_sampler, ")")
        for epoch in range(config.epochs):
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_acc, val_f1, best_t, sel_f1 = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            self.val_f1s.append(val_f1)

            improved = sel_f1 > self.best_val_metric
            if improved:
                self.best_val_metric = sel_f1
                self.best_state = self.model.state_dict()
                self.best_thresh = best_t
                self.patience_counter = 0
                flag = "★"
            else:
                self.patience_counter += 1
                flag = ""

            print(f"Epoch {epoch+1:03d}: "
                  f"Train {train_loss:.4f}, Val {val_loss:.4f}, "
                  f"Val Acc {val_acc:.4f}, Val F1 {val_f1:.4f} "
                  f"| BestT {best_t:.2f}, SelF1 {sel_f1:.4f} {flag}")

            # lr 调度
            self._sched.step()

            if self.patience_counter >= config.patience:
                print("Early stopping.")
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        print(f"Best Val Macro-F1: {self.best_val_metric:.4f} @ Thresh={self.best_thresh:.2f}")

    # ------------ 测试 -------------
    @torch.no_grad()
    def test(self):
        self.model.eval()
        total_loss = 0.0
        all_probs, all_labels = [], []

        for batch in self.test_loader:
            x = batch["x"].to(self.device)
            ei = batch["edge_index"].to(self.device)
            ew = batch["edge_weight"].to(self.device)
            b  = batch["batch"].to(self.device)
            y  = batch["y"].to(self.device)
            mask = batch["eeg_mask"].to(self.device)

            out = self.model(x, ei, ew, b, mask)
            logits = out["logits"]
            loss = self.ce_or_focal(logits, y)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.cpu().numpy())

        all_probs  = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        thr = float(getattr(self, "best_thresh", 0.5))
        preds = (all_probs >= thr).astype(int)

        test_loss = total_loss / max(len(self.test_loader), 1)
        acc = accuracy_score(all_labels, preds)
        f1m = f1_score(all_labels, preds, average="macro")
        try: auc = roc_auc_score(all_labels, all_probs)
        except Exception: auc = float("nan")
        cm = confusion_matrix(all_labels, preds, labels=[0,1])

        print("\n====== Test (using best threshold from Val) ======")
        print(f"Best Thresh : {thr:.2f}")
        print(f"Test Loss   : {test_loss:.4f}")
        print(f"Test Acc    : {acc:.4f}")
        print(f"Test MacroF1: {f1m:.4f}")
        print(f"Test AUC    : {auc:.4f}")
        print("\nClassification report:\n",
              classification_report(all_labels, preds, labels=[0,1], digits=4))

        if HAS_PLOT:
            plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=[0,1], yticklabels=[0,1])
            plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
            plt.tight_layout(); plt.show()

    def plot_curves(self):
        if not HAS_PLOT or not self.train_losses:
            return
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.legend(); plt.grid(alpha=0.3); plt.title("Loss")

        plt.subplot(1,2,2)
        plt.plot(self.val_accs, label="Val Acc")
        plt.plot(self.val_f1s, label="Val Macro-F1")
        plt.legend(); plt.grid(alpha=0.3); plt.title("Val Metrics")
        plt.tight_layout(); plt.show()

# ---------- DataLoader 构建（保留 use_sampler 开关） ----------
def build_loaders(dataset, train_idx, val_idx, test_idx):
    if getattr(config, "use_sampler", True):
        train_labels = np.array([dataset.labels[i] for i in train_idx])
        n0, n1 = np.bincount(train_labels, minlength=2)
        weights = np.where(train_labels == 0, 1.0 / max(n0,1), 1.0 / max(n1,1)).astype(np.float64)
        num_samples = int(2 * max(n0, n1))
        sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples, replacement=True)
        shuffle_flag = False
        print(f"[Sampler] train n0={n0}, n1={n1}, num_samples={num_samples}")
    else:
        sampler, shuffle_flag = None, True

    train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=config.batch_size,
                              shuffle=shuffle_flag, sampler=sampler, collate_fn=collate_fn)
    val_loader   = DataLoader([dataset[i] for i in val_idx], batch_size=config.batch_size,
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader([dataset[i] for i in test_idx], batch_size=config.batch_size,
                              shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader

# --------------- main ---------------
def main():
    subject_ids = range(config.subject_start, config.subject_end + 1)
    dataset = CrossModalGraphDataset(config.data_dir, subject_ids)
    all_labels = np.array(dataset.labels)

    # 样本级分层划分
    idx_all = np.arange(len(dataset))
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        idx_all, all_labels, test_size=(1.0 - config.train_ratio),
        random_state=config.seed, stratify=all_labels)
    val_ratio_rel = config.val_ratio / (config.val_ratio + config.test_ratio)
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx, y_temp, test_size=(1.0 - val_ratio_rel),
        random_state=config.seed, stratify=y_temp)

    print("\nData split:", f"Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # DataLoader
    train_loader, val_loader, test_loader = build_loaders(dataset, train_idx, val_idx, test_idx)

    # 模型
    in_dim = dataset[0]["x"].shape[1]
    print(f"Input dim: {in_dim}")
    model = CoAttnGateGCN(input_dim=in_dim,
                          hidden_dim=config.hidden_dim,
                          num_layers=config.num_layers,
                          dropout=config.dropout,
                          num_classes=2)

    # 训练 & 测试
    trainer = Trainer(model, train_loader, val_loader, test_loader, all_labels)
    trainer.fit()
    trainer.test()
    trainer.plot_curves()

if __name__ == "__main__":
    main()
