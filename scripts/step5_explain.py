"""Step 5: 可解释性可视化（最佳模型 3D ResNet-18）

为最佳模型每个 test 样本的每折预测生成：
- Grad-CAM 热力图（叠加在原始图像上）
- Saliency Map（输入梯度）
- 输出到 explainability/{fold}/{patient_id}.png
"""
from __future__ import annotations

import os, csv, shutil, tempfile
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NIFTI_DIR = os.path.join(PROJECT_ROOT, "nifti_norm")
TEST_CSV = os.path.join(PROJECT_ROOT, "data_split_test.csv")
CKPT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "resnet18_3d")
OUT_DIR = os.path.join(PROJECT_ROOT, "explainability")
N_FOLDS = 5

# ============================================================
# 模型构建
# ============================================================
def build_model(device):
    from torchvision.models.video import r3d_18, R3D_18_Weights
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    old_conv = model.stem[0]
    old_weight = old_conv.weight.data
    new_conv = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                         padding=(1, 3, 3), bias=False)
    with torch.no_grad():
        new_conv.weight.copy_(old_weight.mean(dim=1, keepdim=True))
    model.stem[0] = new_conv
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.to(device)


# ============================================================
# 数据加载
# ============================================================
def load_volume(filepath):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, "temp.nii.gz")
        shutil.copy2(filepath, tmp_path)
        return nib.load(tmp_path).get_fdata().astype(np.float32)


def load_test_samples():
    df = pd.read_csv(TEST_CSV)
    samples = []
    for _, row in df.iterrows():
        pid = str(row["patient_id"]).strip()
        label = int(row["label"])
        filepath = os.path.join(NIFTI_DIR, str(label), f"{pid}.nii.gz")
        samples.append((pid, filepath, label))
    return samples


# ============================================================
# Grad-CAM (3D)
# ============================================================
class GradCAM3D:
    """3D Grad-CAM：hook 在 layer3[-1].conv2

    r3d_18 各层特征图尺寸变化（输入 [1,1,8,110,110]）：
      stem   → [1, 64,  8, 55, 55]   (stride=1,2,2)
      layer1 → [1, 64,  8, 55, 55]   (stride=1)
      layer2 → [1, 128, 4, 28, 28]   (stride=2)
      layer3 → [1, 256, 2, 14, 14]   (stride=2)  ← hook 这里，D=2
      layer4 → [1, 512, 1,  7,  7]   (stride=2)  D=1，所有切片会塌缩为同一张热力图

    选择 layer3 兼顾语义高层特征与深度分辨率，通过三线性插值还原到原始尺寸。
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.features = None
        self.gradients = None

        def forward_hook(m, i, o):
            self.features = o
            o.register_hook(lambda g: setattr(self, 'gradients', g))

        # layer3 输出 D=2，比 layer4 (D=1) 有更好的深度分辨率
        model.layer3[-1].conv2.register_forward_hook(forward_hook)

    def generate(self, x):
        """x: [1, 1, 8, 110, 110] → 返回 cam [8, 110, 110]"""
        logits = self.model(x)
        prob = torch.sigmoid(logits).item()
        self.model.zero_grad()
        logits.backward()

        grads = self.gradients[0].detach()  # [C, D_cam, H_cam, W_cam]
        feats = self.features[0].detach()   # [C, D_cam, H_cam, W_cam]

        weights = grads.mean(dim=(1, 2, 3))  # [C]
        cam = torch.zeros(feats.shape[1:], dtype=feats.dtype, device=feats.device)
        for c, w in enumerate(weights):
            cam += w * feats[c]
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # 三线性插值还原到输入尺寸 [D_in=8, H=110, W=110]
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),  # [1, 1, D_cam, H_cam, W_cam]
            size=(x.shape[2], x.shape[3], x.shape[4]),
            mode='trilinear', align_corners=False
        ).squeeze()  # [8, 110, 110]

        return cam.cpu().numpy(), prob


# ============================================================
# 可视化
# ============================================================
def make_overlay(slice_2d, cam_2d, alpha=0.5):
    """将热力图叠加到灰度图上"""
    slice_norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(slice_norm, cmap='gray')
    ax.imshow(cam_2d, cmap='jet', alpha=alpha)
    ax.axis('off')
    plt.tight_layout(pad=0)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close()
    return data[:, :, :3]


def generate_for_sample(model, filepath, patient_id, label, fold_id, device):
    """生成一个样本的可解释性图"""
    volume = load_volume(filepath)  # [110, 110, 8]
    # [110, 110, 8] -> [1, 1, 8, 110, 110]
    tensor = torch.from_numpy(volume).float()
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,8,110,110]

    # --- Grad-CAM ---
    gradcam = GradCAM3D(model)
    cam_3d, prob = gradcam.generate(tensor)  # cam_3d: [8, 110, 110]

    # --- Saliency Map（在全量 8 层输入上计算梯度，再逐层提取）---
    tensor_sal = tensor.clone().detach().requires_grad_(True)
    logits_sal = model(tensor_sal)
    model.zero_grad()
    logits_sal.backward()
    saliency_3d = tensor_sal.grad.abs().squeeze().cpu().numpy()  # [8, 110, 110]

    # 取所有层做可视化
    n_slices = min(8, volume.shape[2])
    mid_start = max(0, (volume.shape[2] - n_slices) // 2)
    slices_to_show = list(range(mid_start, mid_start + n_slices))

    # 创建 n_slices 行 × 4 列的子图（每行：原始 + CAM + 叠加 + Saliency）
    fig, axes = plt.subplots(n_slices, 4, figsize=(16, 4 * n_slices))
    if n_slices == 1:
        axes = axes.reshape(1, -1)

    for row, z in enumerate(slices_to_show):
        slice_2d = volume[:, :, z]

        # Grad-CAM：cam_3d 已插值到 [8, 110, 110]，直接索引
        cam_2d = cam_3d[z]

        # Saliency：逐层归一化
        sal = saliency_3d[z]
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)

        # 1. 原始图像
        axes[row, 0].imshow(slice_2d, cmap='gray')
        axes[row, 0].set_title(f"Slice {z}" if row == 0 else "")
        axes[row, 0].axis('off')

        # 2. Grad-CAM 热力图
        axes[row, 1].imshow(cam_2d, cmap='jet')
        axes[row, 1].set_title("Grad-CAM" if row == 0 else "")
        axes[row, 1].axis('off')

        # 3. 叠加图
        slice_norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
        axes[row, 2].imshow(slice_norm, cmap='gray')
        axes[row, 2].imshow(cam_2d, cmap='jet', alpha=0.45)
        axes[row, 2].set_title("Overlay" if row == 0 else "")
        axes[row, 2].axis('off')

        # 4. Saliency Map
        axes[row, 3].imshow(sal, cmap='hot')
        axes[row, 3].set_title("Saliency" if row == 0 else "")
        axes[row, 3].axis('off')

    fig.suptitle(f"{patient_id}  |  True: {label}  |  Pred: {prob:.4f}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, prob


# ============================================================
# Main
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    samples = load_test_samples()
    print(f"Test samples: {len(samples)}")

    all_results = []

    for fold_id in range(N_FOLDS):
        ckpt_path = os.path.join(CKPT_DIR, f"fold_{fold_id}", "best_model.pth")
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: Fold {fold_id} checkpoint not found, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"Fold {fold_id}: Loading model from {ckpt_path}")
        model = build_model(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        fold_out = os.path.join(OUT_DIR, f"fold_{fold_id}")
        os.makedirs(fold_out, exist_ok=True)

        for pid, filepath, label in samples:
            save_path = os.path.join(fold_out, f"{pid}.png")
            fig, prob = generate_for_sample(model, filepath, pid, label, fold_id, device)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            all_results.append({
                "fold": fold_id, "patient_id": pid, "true_label": label,
                "prob": prob, "pred": int(prob >= 0.5),
            })

        print(f"  Fold {fold_id}: {len(samples)} images saved")

    # 保存汇总
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)

    # 统计
    for fold_id in range(N_FOLDS):
        fdf = df[df["fold"] == fold_id]
        if len(fdf) == 0:
            continue
        from sklearn.metrics import roc_auc_score, accuracy_score
        auc = roc_auc_score(fdf["true_label"], fdf["prob"])
        acc = accuracy_score(fdf["true_label"], fdf["pred"])
        print(f"  Fold {fold_id}: AUC={auc:.4f}, Acc={acc:.4f}")

    print(f"\nDone! Results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()