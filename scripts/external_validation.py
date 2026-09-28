"""外部验证脚本：对 data/外部验证 数据集进行预处理 + 推理 + 评估

用法:
  python scripts/external_validation.py [--model resnet50]

流程:
  1. DICOM → NIfTI (适配外部验证目录结构: 无 dicom/ 子目录)
  2. Center crop 160×160 → min-max 归一化 → 110×110×8
  3. 5-fold ensemble 推理 → 预测概率
  4. 评估指标输出
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

try:
    from scripts.preprocessing import (
        read_dicom_series, build_dicom_affine, center_crop_160, min_max_norm,
        resize_110x110x8, CROP_SIZE, TARGET_SIZE, PREPROCESS_VERSION,
        DEFAULT_WINDOW_CENTER, DEFAULT_WINDOW_WIDTH,
    )
except ImportError:  # 直接以 python scripts/external_validation.py 运行时 sys.path[0]=scripts
    from preprocessing import (
        read_dicom_series, build_dicom_affine, center_crop_160, min_max_norm,
        resize_110x110x8, CROP_SIZE, TARGET_SIZE, PREPROCESS_VERSION,
        DEFAULT_WINDOW_CENTER, DEFAULT_WINDOW_WIDTH,
    )

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTERNAL_DICOM_DIR = os.path.join(PROJECT_ROOT, 'data', '外部验证未标注')
EXTERNAL_NIFTI_RAW = os.path.join(PROJECT_ROOT, 'external_nifti')
EXTERNAL_NIFTI_ROI = os.path.join(PROJECT_ROOT, 'external_roi')
EXTERNAL_NIFTI_NORM = os.path.join(PROJECT_ROOT, 'external_nifti_v3')
EXTERNAL_RESULTS = os.path.join(PROJECT_ROOT, 'external_results')

BATCH_SIZE = 16
N_FOLDS = 5

MODEL_CONFIGS = {
    "resnet50":       {"channels": 8,  "mode": "2.5d", "input_size": 110},
    "densenet121":    {"channels": 8,  "mode": "2.5d", "input_size": 110},
    "single_slice":   {"channels": 1,  "mode": "2d",   "input_size": 110},
    "resnet18_3d":    {"channels": 1,  "mode": "3d",   "input_size": 110},
    "vit_b_16":       {"channels": 8,  "mode": "2.5d", "input_size": 224},
    # enhanced 3D ResNet-18 variants
    "mixed":          {"channels": 1,  "mode": "3d",   "input_size": 110, "exp": "mixed"},
    "mixed_filtered": {"channels": 1,  "mode": "3d",   "input_size": 110, "exp": "mixed_filtered"},
    "mixed_filtered2":{"channels": 1,  "mode": "3d",   "input_size": 110, "exp": "mixed_filtered2"},
    "augment_strong": {"channels": 1,  "mode": "3d",   "input_size": 110, "exp": "augment_strong"},
}


# ============================================================
# Step 1: DICOM → NIfTI (适配外部验证目录结构)
# ============================================================
def convert_external_dicom():
    """将外部验证 DICOM → NIfTI（无 dicom/ 子目录，带真实 affine）

    预处理逻辑与 step1_preprocess.py 完全一致（共享 scripts/preprocessing.py）:
    Rescale→HU + 窗宽窗位（默认 WC=40, WW=400）→ [0,1]。
    返回 {label: {pid: meta}} 供 normalize 阶段写元数据 sidecar。
    """
    print("=" * 60)
    print("Step 1: 外部验证 DICOM → NIfTI (Rescale + 窗宽窗位 WC="
          f"{DEFAULT_WINDOW_CENTER}, WW={DEFAULT_WINDOW_WIDTH})")
    print("=" * 60)

    patient_metas = {}
    for label in ['0', '1']:
        src = os.path.join(EXTERNAL_DICOM_DIR, label)
        if not os.path.exists(src):
            print(f"  跳过 {label}/ (目录不存在)")
            continue
        dst = os.path.join(EXTERNAL_NIFTI_RAW, label)
        os.makedirs(dst, exist_ok=True)
        patient_metas[label] = {}

        subs = sorted([d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))])
        success, fail = 0, 0
        for sub in subs:
            patient_dir = os.path.join(src, sub)
            try:
                vol, meta = read_dicom_series(patient_dir)
                affine = build_dicom_affine(meta)
                out_path = os.path.join(dst, f'{sub}.nii.gz')
                nib.save(nib.Nifti1Image(vol, affine), out_path)
                meta['label'] = label
                patient_metas[label][sub] = meta
                success += 1
            except Exception as e:
                fail += 1
                print(f"  [失败] {sub}: {e}")
        print(f"  label={label}: 成功 {success}, 失败 {fail}")
    return patient_metas


# ============================================================
# Step 2: 中心裁剪 160×160（affine 平移同步调整）
# ============================================================
def crop_external():
    print("\n" + "=" * 60)
    print("Step 2: 中心裁剪 160×160")
    print("=" * 60)
    for label in ['0', '1']:
        src = os.path.join(EXTERNAL_NIFTI_RAW, label)
        if not os.path.exists(src):
            continue
        dst = os.path.join(EXTERNAL_NIFTI_ROI, label)
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            if not f.endswith('.nii.gz'):
                continue
            nii = nib.load(os.path.join(src, f))
            data = nii.get_fdata().astype(np.float32)
            affine = np.array(nii.affine, dtype=np.float64)
            cropped, _offsets, affine = center_crop_160(data, affine)
            nib.save(nib.Nifti1Image(cropped, affine), os.path.join(dst, f))
    print(f"  完成")


# ============================================================
# Step 3: per-image min-max 归一化 → 110×110×8（与训练数据一致）
# ============================================================
def normalize_external(patient_metas):
    print("\n" + "=" * 60)
    print("Step 3: min-max 归一化 → 110×110×8")
    print("=" * 60)
    for label in ['0', '1']:
        src = os.path.join(EXTERNAL_NIFTI_ROI, label)
        if not os.path.exists(src):
            continue
        dst = os.path.join(EXTERNAL_NIFTI_NORM, label)
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            if not f.endswith('.nii.gz'):
                continue
            nii = nib.load(os.path.join(src, f))
            data = nii.get_fdata().astype(np.float32)
            affine = np.array(nii.affine, dtype=np.float64)
            data = min_max_norm(data)
            data, affine = resize_110x110x8(data, affine)
            nib.save(nib.Nifti1Image(data, affine), os.path.join(dst, f))
            pid = f.replace('.nii.gz', '')
            meta = patient_metas.get(label, {}).get(pid)
            if meta is not None:
                meta['affine'] = affine.tolist()
    print(f"  完成")

    meta_path = os.path.join(EXTERNAL_NIFTI_NORM, 'preprocess_meta.json')
    payload = {
        "preprocess_version": PREPROCESS_VERSION,
        "default_window_center": DEFAULT_WINDOW_CENTER,
        "default_window_width": DEFAULT_WINDOW_WIDTH,
        "crop_size": CROP_SIZE,
        "target_size": list(TARGET_SIZE),
        "pipeline": "read_dicom_series(Rescale+window[0,1]) -> center_crop_160 -> min_max_norm -> resize_110x110x8",
        "patients": {label: pmetas for label, pmetas in patient_metas.items()},
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  预处理元数据 → {meta_path}")


# ============================================================
# 模型构建（复用 step3 逻辑）
# ============================================================
def _expand_conv1_weights(old_weight, target_channels):
    avg = old_weight.mean(dim=1, keepdim=True)
    return avg.repeat(1, target_channels, 1, 1)


def build_model(model_name):
    from torchvision import models
    from torchvision.models import ResNet50_Weights, DenseNet121_Weights, ViT_B_16_Weights

    # enhanced models use resnet18_3d architecture
    arch = model_name
    if model_name in ("mixed", "mixed_filtered", "mixed_filtered2", "augment_strong"):
        arch = "resnet18_3d"

    if arch == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        old = model.conv1.weight.data.clone()
        model.conv1 = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            model.conv1.weight.copy_(_expand_conv1_weights(old, 8))
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif arch == "densenet121":
        model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        old = model.features.conv0.weight.data.clone()
        model.features.conv0 = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            model.features.conv0.weight.copy_(_expand_conv1_weights(old, 8))
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    elif arch == "single_slice":
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif arch == "resnet18_3d":
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
    elif arch == "vit_b_16":
        model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        old = model.conv_proj.weight.data.clone()
        new_conv = nn.Conv2d(8, 768, kernel_size=16, stride=16, bias=False)
        with torch.no_grad():
            new_conv.weight.copy_(old.mean(dim=1, keepdim=True).repeat(1, 8, 1, 1))
        model.conv_proj = new_conv
        model.heads = nn.Linear(model.heads.head.in_features, 1)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


# ============================================================
# 数据加载（外部验证用）
# ============================================================
def load_nifti_with_temp(filepath):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, "temp.nii.gz")
        shutil.copy2(filepath, tmp_path)
        return nib.load(tmp_path).get_fdata().astype(np.float32)


class ExternalDataset(Dataset):
    """外部验证数据集，支持无标签模式"""
    def __init__(self, nifti_norm_dir, patient_ids, label=None, channels=8, mode="2.5d", input_size=110):
        self.channels = channels
        self.mode = mode
        self.input_size = input_size
        self.samples = []
        for pid in patient_ids:
            filepath = os.path.join(nifti_norm_dir, pid + ".nii.gz")
            if os.path.exists(filepath):
                self.samples.append((filepath, pid))
        if input_size != 110:
            self.transform = transforms.Resize((input_size, input_size))
        else:
            self.transform = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, pid = self.samples[idx]
        data = load_nifti_with_temp(filepath)  # [110, 110, 8]

        if self.mode == "2.5d":
            data = np.transpose(data, (2, 0, 1))  # [8, 110, 110]
        elif self.mode == "2d":
            mid = data.shape[2] // 2
            data = data[:, :, mid]
            data = np.stack([data] * 3, axis=0)
        elif self.mode == "3d":
            data = np.transpose(data, (2, 0, 1))  # [8, 110, 110]
            data = data[np.newaxis, :, :, :]  # [1, 8, 110, 110]

        tensor = torch.from_numpy(data).float()
        if self.transform is not None and self.mode != "3d":
            tensor = self.transform(tensor)
        return tensor, pid


# ============================================================
# Step 4: 推理
# ============================================================
@torch.no_grad()
def inference(model_name, device, data_dir=None):
    """使用 5-fold ensemble 对外部验证集进行推理"""
    if data_dir is None:
        data_dir = EXTERNAL_NIFTI_NORM
    config = MODEL_CONFIGS[model_name]
    exp_name = config.get("exp", model_name)
    if "exp" in config:
        ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints_enhanced", "resnet18_3d", exp_name)
    else:
        ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", model_name)

    # 收集所有预处理后的患者
    all_patients = []
    for label in ['0', '1']:
        src = os.path.join(data_dir, label)
        if not os.path.exists(src):
            continue
        for f in os.listdir(src):
            if f.endswith('.nii.gz'):
                all_patients.append((f.replace('.nii.gz', ''), int(label)))

    if not all_patients:
        print("ERROR: 未找到预处理后的外部验证数据，请先运行预处理步骤")
        return None

    print(f"\n外部验证病例数: {len(all_patients)}")
    for label in sorted(set(l for _, l in all_patients)):
        count = sum(1 for _, l in all_patients if l == label)
        print(f"  label={label}: {count} 例")

    # 为所有病例创建统一的数据集（合并所有 label 文件夹）
    # 构建一个统一的 nifti 目录结构
    unified_pids = []
    for pid, label in all_patients:
        unified_pids.append((pid, label))

    # 按 label 分组，分别创建 dataset
    all_probs = {}  # pid -> list of probs from each fold
    all_pids = []

    for pid, label in unified_pids:
        all_pids.append(pid)
        all_probs[pid] = []

    # 按 label 分组推理
    for label in ['0', '1']:
        label_dir = os.path.join(data_dir, label)
        if not os.path.exists(label_dir):
            continue
        label_pids = [pid for pid, l in unified_pids if l == int(label)]

        if not label_pids:
            continue

        dataset = ExternalDataset(label_dir, label_pids,
                                   channels=config["channels"],
                                   mode=config["mode"],
                                   input_size=config["input_size"])
        loader = DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=0)

        # 对每个 fold 的模型进行推理
        for fold_id in range(N_FOLDS):
            ckpt_path = os.path.join(ckpt_dir, f"fold_{fold_id}", "best_model.pth")
            if not os.path.exists(ckpt_path):
                print(f"  WARNING: 未找到 {ckpt_path}，跳过")
                continue

            model = build_model(model_name).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()

            for tensors, pids in loader:
                tensors = tensors.to(device)
                probs = torch.sigmoid(model(tensors)).cpu().numpy().reshape(-1)
                for pid, prob in zip(pids, probs):
                    all_probs[pid].append(float(prob))

            print(f"  Fold {fold_id} ({model_name}) 推理完成")

    # 计算 ensemble 平均概率
    results = []
    for pid in all_pids:
        if all_probs[pid]:
            mean_prob = np.mean(all_probs[pid])
            std_prob = np.std(all_probs[pid])
            results.append((pid, mean_prob, std_prob))

    return results, all_patients


# ============================================================
# Step 5: 评估
# ============================================================
def evaluate(results, all_patients, model_name, data_dir=None, no_labels=False, threshold=0.5):
    """评估外部验证结果"""
    if data_dir is None:
        data_dir = EXTERNAL_NIFTI_NORM
    print("\n" + "=" * 60)
    print(f"外部验证评估 ({model_name})" + (" [无标签]" if no_labels else ""))
    print(f"分类阈值: {threshold:.3f}")
    print("=" * 60)

    # 构建 pid -> label 映射
    # 文件夹标签与目标标签一致：0=阴性(正常), 1=阳性(异常)
    pid2label = {pid: label for pid, label in all_patients}

    probs = []
    labels = []
    pids_out = []

    for pid, mean_prob, std_prob in results:
        pids_out.append(pid)
        probs.append(mean_prob)
        labels.append(pid2label.get(pid, 0))

    probs = np.array(probs)
    labels = np.array(labels)
    preds = (probs >= threshold).astype(int)

    # 汇总统计
    print(f"\n总计: {len(results)} 例")
    print(f"  label=0 (正常): {sum(labels == 0)} 例")
    print(f"  label=1 (异常): {sum(labels == 1)} 例")

    # 概率分布
    print(f"\n预测概率分布:")
    print(f"  Mean ± Std: {probs.mean():.4f} ± {probs.std():.4f}")
    print(f"  Min: {probs.min():.4f}, Max: {probs.max():.4f}")
    print(f"  Median: {np.median(probs):.4f}")
    print(f"  预测为异常 (>={threshold:.3f}): {sum(preds == 1)} / {len(preds)} ({sum(preds == 1)/len(preds)*100:.1f}%)")
    print(f"  预测为正常 (<{threshold:.3f}): {sum(preds == 0)} / {len(preds)} ({sum(preds == 0)/len(preds)*100:.1f}%)")

    # 按标签统计
    for label_val in sorted(set(labels)):
        mask = labels == label_val
        label_probs = probs[mask]
        label_preds = preds[mask]
        label_name = "正常" if label_val == 0 else "异常"
        print(f"\n  [label={label_val} ({label_name})] {sum(mask)} 例:")
        print(f"    预测概率: Mean={label_probs.mean():.4f}, Std={label_probs.std():.4f}")
        print(f"    预测为异常: {sum(label_preds == 1)} / {sum(mask)} ({sum(label_preds == 1)/sum(mask)*100:.1f}%)")

    # 二分类指标（如果两个类别都有）
    if not no_labels and len(set(labels)) > 1:
        print(f"\n{'='*60}")
        print(f"二分类指标")
        print(f"{'='*60}")
        try:
            auc = roc_auc_score(labels, probs)
            print(f"  AUC: {auc:.4f}")
        except Exception:
            auc = None
            print(f"  AUC: N/A (单类别)")

        acc = accuracy_score(labels, preds)
        sens = recall_score(labels, preds, pos_label=1)  # label=1 为异常（阳性）
        spec = recall_score(labels, preds, pos_label=0)  # label=0 为正常（阴性）
        prec = precision_score(labels, preds, pos_label=1, zero_division=0)
        f1 = f1_score(labels, preds, pos_label=1, zero_division=0)

        print(f"  Accuracy: {acc:.4f}")
        print(f"  Sensitivity (检出率): {sens:.4f}")  # 异常被正确识别
        print(f"  Specificity: {spec:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  F1: {f1:.4f}")

        cm = confusion_matrix(labels, preds)
        print(f"\n  混淆矩阵:")
        print(f"                Pred Neg  Pred Pos")
        print(f"  Actual Neg    {cm[0, 0]:>8d}  {cm[0, 1]:>8d}")
        print(f"  Actual Pos    {cm[1, 0]:>8d}  {cm[1, 1]:>8d}")
    elif no_labels:
        print(f"\n  [无标签模式] 仅输出预测概率，不计算评估指标")
        auc = acc = sens = spec = prec = f1 = None

    # 保存详细结果
    os.makedirs(EXTERNAL_RESULTS, exist_ok=True)
    data_tag = os.path.basename(data_dir.rstrip('/\\'))
    results_path = os.path.join(EXTERNAL_RESULTS, f"external_validation_{model_name}_{data_tag}.csv")
    with open(results_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['patient_id', 'label', 'mean_prob', 'std_prob', 'prediction'])
        for pid, mean_prob, std_prob in results:
            label = pid2label.get(pid, 0) if not no_labels else -1
            pred = 1 if mean_prob >= threshold else 0
            w.writerow([pid, label, f"{mean_prob:.6f}", f"{std_prob:.6f}", pred])

    print(f"\n详细结果保存至: {results_path}")

    # 保存摘要
    summary_path = os.path.join(EXTERNAL_RESULTS, f"external_summary_{model_name}_{data_tag}.csv")
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Metric', 'Value'])
        w.writerow(['Total Cases', len(results)])
        w.writerow(['Label=0 (正常)', sum(labels == 0) if not no_labels else 'N/A'])
        w.writerow(['Label=1 (异常)', sum(labels == 1) if not no_labels else 'N/A'])
        w.writerow(['Mean Prob', f"{probs.mean():.4f}"])
        w.writerow(['Std Prob', f"{probs.std():.4f}"])
        w.writerow(['Predicted Anomalous', f"{sum(preds == 1)} ({sum(preds == 1)/len(preds)*100:.1f}%)"])
        w.writerow(['Predicted Normal', f"{sum(preds == 0)} ({sum(preds == 0)/len(preds)*100:.1f}%)"])
        if not no_labels and len(set(labels)) > 1:
            w.writerow(['AUC', f"{auc:.4f}" if auc else "N/A"])
            w.writerow(['Accuracy', f"{acc:.4f}"])
            w.writerow(['Sensitivity', f"{sens:.4f}"])
            w.writerow(['Specificity', f"{spec:.4f}"])
            w.writerow(['Precision', f"{prec:.4f}"])
            w.writerow(['F1', f"{f1:.4f}"])

    print(f"摘要保存至: {summary_path}")

    return results_path


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="外部验证")
    parser.add_argument("--model", type=str, default="resnet50",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model architecture (default: resnet50)")
    parser.add_argument("--skip_preprocess", action="store_true",
                        help="跳过预处理（已有预处理数据）")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="预处理后的 NIfTI 目录 (如 external_nifti/labeled)")
    parser.add_argument("--no_labels", action="store_true",
                        help="无标签模式（仅输出预测，不计算指标）")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="分类阈值 (default: 0.5)")
    args = parser.parse_args()

    model_name = args.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Step 1-3: 预处理
    if not args.skip_preprocess and not args.data_dir:
        # 清理旧数据
        for d in [EXTERNAL_NIFTI_RAW, EXTERNAL_NIFTI_ROI, EXTERNAL_NIFTI_NORM]:
            if os.path.exists(d):
                shutil.rmtree(d)

        patient_metas = convert_external_dicom()
        crop_external()
        normalize_external(patient_metas)
    elif args.data_dir:
        print(f"使用已有预处理数据: {args.data_dir}")
    else:
        print("跳过预处理，使用已有数据")

    # Step 4: 推理
    print("\n" + "=" * 60)
    print(f"Step 4: {model_name} 5-fold Ensemble 推理")
    print("=" * 60)

    result = inference(model_name, device, data_dir=args.data_dir)
    if result is None:
        return
    results, all_patients = result

    # Step 5: 评估
    evaluate(results, all_patients, model_name, data_dir=args.data_dir, no_labels=args.no_labels, threshold=args.threshold)

    print("\n外部验证完成!")


if __name__ == "__main__":
    main()