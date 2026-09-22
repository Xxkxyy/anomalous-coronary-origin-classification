"""Step 3: 5-fold 交叉验证训练 (多模型支持)

用法:
  python step3_train.py --model resnet50          # 2.5D ResNet-50 (默认)
  python step3_train.py --model densenet121       # 2.5D DenseNet-121
  python step3_train.py --model single_slice      # 单层 2D ResNet-50
  python step3_train.py --model resnet18_3d       # 3D ResNet-18

输出: checkpoints/{model}/fold_{i}/, cv_results_{model}.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights, DenseNet121_Weights, ViT_B_16_Weights
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NIFTI_DIR = os.path.join(PROJECT_ROOT, "nifti_norm")
SPLIT_CSV = os.path.join(PROJECT_ROOT, "data_split.csv")
TEST_CSV = os.path.join(PROJECT_ROOT, "data_split_test.csv")

BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 15
T_MAX = 100
NUM_WORKERS = 0
N_FOLDS = 5

MODEL_CONFIGS = {
    "resnet50":       {"channels": 8,  "mode": "2.5d", "input_size": 110},
    "densenet121":    {"channels": 8,  "mode": "2.5d", "input_size": 110},
    "single_slice":   {"channels": 1,  "mode": "2d",   "input_size": 110},
    "resnet18_3d":    {"channels": 1,  "mode": "3d",   "input_size": 110},
    "vit_b_16":       {"channels": 8,  "mode": "2.5d", "input_size": 224},
}

# ============================================================
# 数据加载
# ============================================================
def load_nifti_with_temp(filepath):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, "temp.nii.gz")
        shutil.copy2(filepath, tmp_path)
        return nib.load(tmp_path).get_fdata().astype(np.float32)


class NiftiDataset(Dataset):
    def __init__(self, nifti_dir, split_df, fold, split_name, augment=False,
                 channels=8, mode="2.5d", input_size=110):
        self.augment = augment
        self.channels = channels
        self.mode = mode
        self.input_size = input_size
        df = split_df[(split_df["fold"] == fold) & (split_df["split"] == split_name)]
        self.samples = []
        for _, row in df.iterrows():
            pid = str(row["patient_id"]).strip()
            label = int(row["label"])
            filepath = os.path.join(nifti_dir, str(label), pid + ".nii.gz")
            self.samples.append((filepath, label))
        if augment:
            base_transforms = [
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
            if input_size != 110:
                base_transforms.append(transforms.Resize((input_size, input_size)))
            self.transform = transforms.Compose(base_transforms)
        else:
            if input_size != 110:
                self.transform = transforms.Resize((input_size, input_size))
            else:
                self.transform = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        data = load_nifti_with_temp(filepath)  # [110, 110, 8]

        if self.mode == "2.5d":
            # [110, 110, 8] → [8, 110, 110]
            data = np.transpose(data, (2, 0, 1))
            if self.augment and data.shape[0] > 1:
                shift = np.random.randint(-1, 2)
                if shift != 0:
                    data = np.roll(data, shift, axis=0)
        elif self.mode == "2d":
            # 取中间 1 层，复制为 3 通道（标准 ResNet 输入）
            mid = data.shape[2] // 2
            data = data[:, :, mid]  # [110, 110]
            data = np.stack([data] * 3, axis=0)  # [3, 110, 110]
        elif self.mode == "3d":
            # [110, 110, 8] → [1, 8, 110, 110] (C, D, H, W)
            data = np.transpose(data, (2, 0, 1))  # [8, 110, 110]
            data = data[np.newaxis, :, :, :]  # [1, 8, 110, 110]

        tensor = torch.from_numpy(data).float()
        if self.transform is not None and self.mode != "3d":
            tensor = self.transform(tensor)
        return tensor, torch.tensor(float(label), dtype=torch.float32)


class TestDataset(Dataset):
    def __init__(self, nifti_dir, test_df, channels=8, mode="2.5d", input_size=110):
        self.channels = channels
        self.mode = mode
        self.input_size = input_size
        self.samples = []
        for _, row in test_df.iterrows():
            pid = str(row["patient_id"]).strip()
            label = int(row["label"])
            filepath = os.path.join(nifti_dir, str(label), pid + ".nii.gz")
            self.samples.append((filepath, label))
        if input_size != 110:
            self.transform = transforms.Resize((input_size, input_size))
        else:
            self.transform = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        data = load_nifti_with_temp(filepath)
        if self.mode == "2.5d":
            data = np.transpose(data, (2, 0, 1))
        elif self.mode == "2d":
            mid = data.shape[2] // 2
            data = data[:, :, mid]
            data = np.stack([data] * 3, axis=0)
        elif self.mode == "3d":
            data = np.transpose(data, (2, 0, 1))
            data = data[np.newaxis, :, :, :]
        tensor = torch.from_numpy(data).float()
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, torch.tensor(float(label), dtype=torch.float32)


# ============================================================
# 模型构建
# ============================================================
def _expand_conv1_weights(old_weight, target_channels):
    """将预训练 RGB conv1 权重扩展到 target_channels"""
    avg = old_weight.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]
    return avg.repeat(1, target_channels, 1, 1)


def build_resnet50():
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    old = model.conv1.weight.data.clone()
    model.conv1 = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        model.conv1.weight.copy_(_expand_conv1_weights(old, 8))
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def build_densenet121():
    model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
    old = model.features.conv0.weight.data.clone()
    model.features.conv0 = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        model.features.conv0.weight.copy_(_expand_conv1_weights(old, 8))
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model


def build_single_slice():
    """标准 ResNet-50，3 通道输入（单层复制为 RGB）"""
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def build_resnet18_3d():
    """3D ResNet-18 (torchvision video model)"""
    from torchvision.models.video import r3d_18, R3D_18_Weights
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    # 替换第一个 conv3d 以接受 1 通道输入（灰度）
    old_conv = model.stem[0]
    old_weight = old_conv.weight.data  # [64, 3, 3, 7, 7]
    new_conv = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                         padding=(1, 3, 3), bias=False)
    with torch.no_grad():
        new_conv.weight.copy_(old_weight.mean(dim=1, keepdim=True))
    model.stem[0] = new_conv
    # 替换最后的 fc 层
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def build_vit_b_16():
    """ViT-B/16，8 通道输入（2.5D）"""
    model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    old = model.conv_proj.weight.data.clone()  # [768, 3, 16, 16]
    new_conv = nn.Conv2d(8, 768, kernel_size=16, stride=16, bias=False)
    with torch.no_grad():
        new_conv.weight.copy_(old.mean(dim=1, keepdim=True).repeat(1, 8, 1, 1))
    model.conv_proj = new_conv
    model.heads = nn.Linear(model.heads.head.in_features, 1)
    return model


def build_model(model_name):
    builders = {
        "resnet50": build_resnet50,
        "densenet121": build_densenet121,
        "single_slice": build_single_slice,
        "resnet18_3d": build_resnet18_3d,
        "vit_b_16": build_vit_b_16,
    }
    if model_name not in builders:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(builders.keys())}")
    return builders[model_name]()


# ============================================================
# 训练
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for tensors, labels in loader:
        tensors = tensors.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).view(-1, 1)
        optimizer.zero_grad()
        loss = criterion(model(tensors), labels)
        loss.backward()
        optimizer.step()
        bs = tensors.size(0)
        total_loss += loss.item() * bs
        n += bs
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_labels, all_probs = [], []
    for tensors, labels in loader:
        tensors = tensors.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).view(-1, 1)
        logits = model(tensors)
        loss = criterion(logits, labels)
        bs = tensors.size(0)
        total_loss += loss.item() * bs
        n += bs
        all_labels.extend(labels.cpu().numpy().reshape(-1).tolist())
        all_probs.extend(torch.sigmoid(logits).cpu().numpy().reshape(-1).tolist())
    avg_loss = total_loss / max(n, 1)
    auc = float(roc_auc_score(all_labels, all_probs)) if len(set(all_labels)) > 1 else 0.0
    return avg_loss, auc, all_labels, all_probs


def run_fold(fold_id, train_df, test_df, device, model_name, config):
    """训练单个 fold"""
    fold_name = f"fold_{fold_id}"
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", model_name)
    fold_dir = os.path.join(ckpt_dir, fold_name)
    os.makedirs(fold_dir, exist_ok=True)
    best_path = os.path.join(fold_dir, "best_model.pth")

    channels = config["channels"]
    mode = config["mode"]

    train_set = NiftiDataset(NIFTI_DIR, train_df, fold_name, "train", augment=True,
                             channels=channels, mode=mode, input_size=config["input_size"])
    val_set = NiftiDataset(NIFTI_DIR, train_df, fold_name, "val", augment=False,
                           channels=channels, mode=mode, input_size=config["input_size"])
    test_set = TestDataset(NIFTI_DIR, test_df, channels=channels, mode=mode, input_size=config["input_size"])

    print(f"  Fold {fold_id}: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    train_labels = [s[1] for s in train_set.samples]
    n_neg = train_labels.count(0)
    n_pos = train_labels.count(1)
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"    pos_weight = {n_neg}/{n_pos} = {pos_weight:.4f}")

    model = build_model(model_name).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX)

    best_val_auc = -1.0
    no_improve = 0
    epoch_logs = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc, _, _ = evaluate(model, val_loader, criterion, device)
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        epoch_logs.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_auc": val_auc, "lr": cur_lr,
        })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            no_improve = 0
            torch.save(model.state_dict(), best_path)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"    Early stop at epoch {epoch}")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_auc, test_labels, test_probs = evaluate(model, test_loader, criterion, device)
    return best_val_auc, test_auc, test_loss, test_labels, test_probs, epoch_logs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet50",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model architecture")
    args = parser.parse_args()

    model_name = args.model
    config = MODEL_CONFIGS[model_name]

    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", model_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    results_csv = os.path.join(PROJECT_ROOT, f"cv_results_{model_name}.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {model_name} (mode={config['mode']}, channels={config['channels']})")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    train_df = pd.read_csv(SPLIT_CSV)
    test_df = pd.read_csv(TEST_CSV)
    print(f"CV 划分: {len(train_df)} 条记录, Holdout: {len(test_df)} 例")

    all_results = []
    for fold_id in range(N_FOLDS):
        print(f"\n{'='*50}")
        print(f"Fold {fold_id}/{N_FOLDS}")
        print(f"{'='*50}")

        best_val_auc, test_auc, test_loss, test_labels, test_probs, epoch_logs = \
            run_fold(fold_id, train_df, test_df, device, model_name, config)

        all_results.append({
            "fold": fold_id, "best_val_auc": best_val_auc,
            "test_auc": test_auc, "test_loss": test_loss,
            "best_epoch": max(l["epoch"] for l in epoch_logs),
        })

        # 保存日志
        log_path = os.path.join(ckpt_dir, f"fold_{fold_id}", "epoch_log.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_auc", "lr"])
            w.writeheader()
            w.writerows(epoch_logs)

        # 保存预测
        pred_path = os.path.join(ckpt_dir, f"fold_{fold_id}", "test_predictions.csv")
        test_pids = test_df["patient_id"].tolist()
        with open(pred_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["patient_id", "true_label", "prob"])
            for pid, tl, pr in zip(test_pids, test_labels, test_probs):
                w.writerow([pid, int(tl), pr])

        print(f"  Fold {fold_id}: best_val_auc={best_val_auc:.4f}, test_auc={test_auc:.4f}")

    # 汇总
    print(f"\n{'='*50}")
    print(f"{model_name} 5-Fold CV 汇总")
    print(f"{'='*50}")
    val_aucs = [r["best_val_auc"] for r in all_results]
    test_aucs = [r["test_auc"] for r in all_results]
    print(f"  Val AUC: {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}")
    print(f"  Test AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")

    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "best_val_auc", "test_auc", "test_loss", "best_epoch"])
        w.writeheader()
        w.writerows(all_results)
    print(f"\n汇总结果 → {results_csv}")


if __name__ == "__main__":
    main()