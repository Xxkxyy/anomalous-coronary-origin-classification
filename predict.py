"""推理脚本：给定 DICOM/NIfTI 文件夹，输出预测结果

用法:
  # 单个患者 DICOM 目录
  python predict.py --input ./case_dicom/ --output ./results.csv

  # 批量患者（每个子目录是一个患者）
  python predict.py --input ./all_cases/ --output ./results.csv

  # 批量患者（每个子目录下有 dicom/ 子目录）
  python predict.py --input ./all_cases/ --output ./results.csv --dicom-subdir

  # 指定模型 + TTA
  python predict.py --input ./cases/ --model densenet121 --tta --output ./results.csv

  # 多模型集成（推荐，提升召回率）
  python predict.py --input ./cases/ --ensemble --output ./results.csv

  # 自定义阈值（提高召回率用低阈值 0.3，降低假阳性用高阈值 0.7）
  python predict.py --input ./cases/ --ensemble --threshold 0.3 --output ./results.csv

输入支持:
  1. DICOM 序列：目录下直接是 .dcm 文件
  2. DICOM 序列（含 dicom/ 子目录）：每个患者目录下有 dicom/ 子目录
  3. 已预处理的 NIfTI：目录下是 .nii.gz 文件

输出:
  CSV 文件: patient_id, probability, prediction (异常概率≥threshold为阳性)
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import tempfile
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights, DenseNet121_Weights, ViT_B_16_Weights
from scipy.ndimage import zoom, rotate

warnings.filterwarnings("ignore")

# ============================================================
# 配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CROP_SIZE = 160
TARGET_SIZE = (110, 110, 8)
BATCH_SIZE = 16
N_FOLDS = 5

MODEL_CONFIGS = {
    "resnet50":    {"channels": 8, "mode": "2.5d", "input_size": 110},
    "densenet121": {"channels": 8, "mode": "2.5d", "input_size": 110},
    "resnet18_3d": {"channels": 1, "mode": "3d",   "input_size": 110},
    "vit_b_16":    {"channels": 8, "mode": "2.5d", "input_size": 224},
}

# 默认集成模型（排除 ViT，其在召回率上表现较差）
ENSEMBLE_MODELS = ["resnet50", "densenet121", "resnet18_3d"]


# ============================================================
# DICOM → NIfTI 预处理
# ============================================================
def _read_dicom_series(dicom_dir: str, wc_override: float = None,
                       ww_override: float = None) -> np.ndarray:
    """读取 DICOM 序列，按 InstanceNumber 排序，统一处理为灰度 [0, 65535]

    自动检测 DICOM 类型:
      - RGB 二级图像 (DERIVED): 直接 RGB→灰度
      - 原始 CT DICOM: HU 转换 + 窗宽窗位 → 映射到 [0, 65535]

    Args:
        wc_override: 指定 WindowCenter (None=使用 DICOM 头中的值)
        ww_override: 指定 WindowWidth (None=使用 DICOM 头中的值)
    """
    import pydicom
    slices = []
    for f in sorted(os.listdir(dicom_dir)):
        if not f.endswith('.dcm'):
            continue
        d = pydicom.dcmread(os.path.join(dicom_dir, f))
        pix = d.pixel_array.astype(np.float32)

        if pix.ndim == 3 and pix.shape[2] == 3:
            # RGB 二级图像 (如 PACS 截图): RGB → 灰度
            pix = 0.299 * pix[:, :, 0] + 0.587 * pix[:, :, 1] + 0.114 * pix[:, :, 2]
        else:
            # 原始 CT DICOM: 应用 Rescale → HU, 再窗宽窗位 → 映射到 [0, 65535]
            rescale_slope = float(getattr(d, 'RescaleSlope', 1))
            rescale_intercept = float(getattr(d, 'RescaleIntercept', 0))
            if wc_override is not None and ww_override is not None:
                wc, ww = wc_override, ww_override
            else:
                wc = getattr(d, 'WindowCenter', 40)
                ww = getattr(d, 'WindowWidth', 400)
                if hasattr(wc, '__iter__') and not isinstance(wc, str):
                    wc = float(wc[0])
                else:
                    wc = float(wc)
                if hasattr(ww, '__iter__') and not isinstance(ww, str):
                    ww = float(ww[0])
                else:
                    ww = float(ww)

            # HU = stored_value * slope + intercept
            pix = pix * rescale_slope + rescale_intercept

            # 窗宽窗位映射到 [0, 65535]
            low = wc - ww / 2
            pix = np.clip((pix - low) / ww, 0.0, 1.0) * 65535.0

        slices.append((d.InstanceNumber, pix))
    slices.sort(key=lambda x: x[0])
    return np.stack([s[1] for s in slices], axis=2)


def preprocess_dicom_volume(dicom_dir: str, wc: float = None,
                            ww: float = None) -> np.ndarray:
    """DICOM 目录 → 预处理后的 3D 体积 [110, 110, 8]

    步骤: DICOM 序列读取 → 中心裁剪 160×160 → z-score 归一化 → 分布缩放 → resize 110×110×8
    """
    vol = _read_dicom_series(dicom_dir, wc_override=wc, ww_override=ww)

    # 中心裁剪 160×160
    h, w = vol.shape[0], vol.shape[1]
    cy, cx = h // 2, w // 2
    half = CROP_SIZE // 2
    vol = vol[cy - half:cy + half, cx - half:cx + half, :]

    # per-image min-max 归一化到 [0, 1]（与训练数据一致）
    dmin, dmax = vol.min(), vol.max()
    if dmax > dmin:
        vol = (vol - dmin) / (dmax - dmin)

    # resize 到 [110, 110, 8]
    zoom_factors = (TARGET_SIZE[0] / vol.shape[0],
                    TARGET_SIZE[1] / vol.shape[1],
                    TARGET_SIZE[2] / vol.shape[2])
    vol = zoom(vol, zoom_factors, order=1)

    return vol.astype(np.float32)


# ============================================================
# 模型构建
# ============================================================
def _expand_conv1_weights(old_weight, target_channels):
    avg = old_weight.mean(dim=1, keepdim=True)
    return avg.repeat(1, target_channels, 1, 1)


def build_model(model_name: str) -> nn.Module:
    if model_name == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        old = model.conv1.weight.data.clone()
        model.conv1 = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            model.conv1.weight.copy_(_expand_conv1_weights(old, 8))
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif model_name == "densenet121":
        model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        old = model.features.conv0.weight.data.clone()
        model.features.conv0 = nn.Conv2d(8, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            model.features.conv0.weight.copy_(_expand_conv1_weights(old, 8))
        model.classifier = nn.Linear(model.classifier.in_features, 1)
    elif model_name == "resnet18_3d":
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
    elif model_name == "vit_b_16":
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
# TTA (Test-Time Augmentation)
# ============================================================
def generate_tta_views(volume: np.ndarray, mode: str) -> list:
    """生成 TTA 增强视图列表

    Args:
        volume: 原始体积 [110, 110, 8]
        mode: "2.5d" 或 "3d"

    Returns:
        list of (view_data, is_flipped) tuples
    """
    views = [volume]  # 原始视图

    if mode == "2.5d":
        # 水平翻转 (沿 H 轴)
        flipped = np.flip(volume, axis=1).copy()
        views.append(flipped)

        # 小角度旋转 ±5°
        for angle in [5, -5]:
            rotated = np.zeros_like(volume)
            for z in range(volume.shape[2]):
                rotated[:, :, z] = rotate(volume[:, :, z], angle, reshape=False, order=1)
            views.append(rotated)

    elif mode == "3d":
        # 水平翻转 (沿 H 轴)
        flipped = np.flip(volume, axis=1).copy()
        views.append(flipped)

    return views


def volume_to_tensor(volume: np.ndarray, channels: int, mode: str,
                     input_size: int) -> torch.Tensor:
    """将体积数组转换为模型输入 tensor"""
    data = volume.copy()

    if mode == "2.5d":
        data = np.transpose(data, (2, 0, 1))  # [8, 110, 110]
    elif mode == "3d":
        data = np.transpose(data, (2, 0, 1))  # [8, 110, 110]
        data = data[np.newaxis, :, :, :]  # [1, 8, 110, 110]

    tensor = torch.from_numpy(data).float()

    if input_size != 110 and mode != "3d":
        tensor = F.interpolate(tensor.unsqueeze(0), size=(input_size, input_size),
                               mode='bilinear', align_corners=False).squeeze(0)

    return tensor


# ============================================================
# 数据加载
# ============================================================
class VolumeDataset(Dataset):
    """推理数据集：从预处理好的 numpy 数组列表加载"""
    def __init__(self, volumes, ids, channels=8, mode="2.5d", input_size=110,
                 use_tta=False):
        self.volumes = volumes
        self.ids = ids
        self.channels = channels
        self.mode = mode
        self.use_tta = use_tta
        if use_tta:
            self._all_views = []
            for vol in volumes:
                self._all_views.append(generate_tta_views(vol, mode))
            self._tta_count = len(self._all_views[0])
        if input_size != 110:
            self.transform = transforms.Resize((input_size, input_size))
        else:
            self.transform = None

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        if self.use_tta:
            # TTA 模式：返回所有视图的 tensor 列表
            views = self._all_views[idx]
            tensors = []
            for v in views:
                t = volume_to_tensor(v, self.channels, self.mode, 110)
                if self.transform is not None and self.mode != "3d":
                    t = self.transform(t)
                tensors.append(t)
            return torch.stack(tensors), self.ids[idx]
        else:
            data = self.volumes[idx]  # [110, 110, 8]
            tensor = volume_to_tensor(data, self.channels, self.mode, 110)
            if self.transform is not None and self.mode != "3d":
                tensor = self.transform(tensor)
            return tensor, self.ids[idx]


# ============================================================
# 推理
# ============================================================
@torch.no_grad()
def run_inference(volumes, patient_ids, model_name, device, use_tta=False):
    """5-fold ensemble 推理（可选 TTA）"""
    config = MODEL_CONFIGS[model_name]
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", model_name)

    dataset = VolumeDataset(volumes, patient_ids,
                            channels=config["channels"],
                            mode=config["mode"],
                            input_size=config["input_size"],
                            use_tta=use_tta)
    loader = DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=0)

    all_probs = {pid: [] for pid in patient_ids}

    for fold_id in range(N_FOLDS):
        ckpt_path = os.path.join(ckpt_dir, f"fold_{fold_id}", "best_model.pth")
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: Fold {fold_id} checkpoint not found, skipping")
            continue

        model = build_model(model_name).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        for batch in loader:
            if use_tta:
                # TTA: [batch, n_views, C, H, W] 或 [batch, n_views, C, D, H, W]
                tensors, pids = batch
                b, nv = tensors.shape[0], tensors.shape[1]
                flat = tensors.view(b * nv, *tensors.shape[2:]).to(device)
                probs = torch.sigmoid(model(flat)).cpu().numpy().reshape(b, nv)
                # 平均所有视图的预测
                probs = probs.mean(axis=1)
            else:
                tensors, pids = batch
                tensors = tensors.to(device)
                probs = torch.sigmoid(model(tensors)).cpu().numpy().reshape(-1)

            for pid, prob in zip(pids, probs):
                all_probs[pid].append(float(prob))

        print(f"  Fold {fold_id} 推理完成")

    # Ensemble 平均
    results = {}
    for pid in patient_ids:
        if all_probs[pid]:
            results[pid] = {
                "mean_prob": np.mean(all_probs[pid]),
                "std_prob": np.std(all_probs[pid]) if len(all_probs[pid]) > 1 else 0.0,
                "n_folds": len(all_probs[pid]),
            }
        else:
            results[pid] = {"mean_prob": 0.0, "std_prob": 0.0, "n_folds": 0}

    return results


def run_ensemble_inference(volumes, patient_ids, model_names, device,
                           use_tta=False):
    """多模型集成推理：对每个模型做 5-fold 推理，再平均各模型概率"""
    all_model_probs = {pid: [] for pid in patient_ids}

    for model_name in model_names:
        print(f"\n--- {model_name} ---")
        results = run_inference(volumes, patient_ids, model_name, device,
                                use_tta=use_tta)
        for pid in patient_ids:
            all_model_probs[pid].append(results[pid]["mean_prob"])

    # 跨模型平均
    final_results = {}
    for pid in patient_ids:
        probs = all_model_probs[pid]
        final_results[pid] = {
            "mean_prob": np.mean(probs) if probs else 0.0,
            "std_prob": np.std(probs) if len(probs) > 1 else 0.0,
            "n_folds": len(probs),
        }

    return final_results


# ============================================================
# 输入发现
# ============================================================
def discover_patients(input_dir: str, use_dicom_subdir: bool = False) -> dict:
    """自动发现输入目录中的患者

    支持三种结构:
      1. 目录下直接是 .dcm 文件 → 单个患者
      2. 子目录是患者，每个子目录内是 .dcm 文件 → 批量
      3. 子目录是患者，每个子目录下有 dicom/ 子目录 → 批量 (use_dicom_subdir=True)
      4. 目录下是 .nii.gz 文件 → 已预处理
    """
    # 检查是否是 DICOM 目录（直接含 .dcm 文件）
    dcm_files = [f for f in os.listdir(input_dir) if f.endswith('.dcm')]
    if dcm_files:
        name = os.path.basename(input_dir.rstrip('/\\'))
        return {name: {"type": "dicom", "path": input_dir}}

    # 检查是否是 NIfTI 目录
    nii_files = [f for f in os.listdir(input_dir) if f.endswith('.nii.gz')]
    if nii_files:
        patients = {}
        for f in nii_files:
            pid = f.replace('.nii.gz', '')
            patients[pid] = {"type": "nifti", "path": os.path.join(input_dir, f)}
        return patients

    # 子目录模式
    patients = {}
    for sub in sorted(os.listdir(input_dir)):
        sub_path = os.path.join(input_dir, sub)
        if not os.path.isdir(sub_path):
            continue

        if use_dicom_subdir:
            dicom_dir = os.path.join(sub_path, 'dicom')
            if os.path.isdir(dicom_dir):
                dcm_files = [f for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
                if dcm_files:
                    patients[sub] = {"type": "dicom", "path": dicom_dir}
        else:
            dcm_files = [f for f in os.listdir(sub_path) if f.endswith('.dcm')]
            if dcm_files:
                patients[sub] = {"type": "dicom", "path": sub_path}

    return patients


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="冠脉起源异常分类推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python predict.py -i ./case1/ -o results.csv
  python predict.py -i ./all_cases/ -o results.csv
  python predict.py -i ./all_cases/ --dicom-subdir -o results.csv
  python predict.py -i ./cases/ -m densenet121 -o results.csv
  python predict.py -i ./cases/ --ensemble --tta -t 0.3 -o results.csv
        """,
    )
    parser.add_argument("-i", "--input", required=True,
                        help="输入目录（DICOM 或 NIfTI）")
    parser.add_argument("-o", "--output", default="predictions.csv",
                        help="输出 CSV 路径 (默认: predictions.csv)")
    parser.add_argument("-m", "--model", default="resnet50",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="模型架构 (默认: resnet50)")
    parser.add_argument("--ensemble", action="store_true",
                        help="使用多模型集成 (ResNet50 + DenseNet121 + ResNet18_3D)")
    parser.add_argument("--tta", action="store_true",
                        help="启用 Test-Time Augmentation (水平翻转+旋转)")
    parser.add_argument("-t", "--threshold", type=float, default=0.5,
                        help="分类阈值 (默认: 0.5; 提高召回率用 0.3, 降低假阳性用 0.7)")
    parser.add_argument("--dicom-subdir", action="store_true",
                        help="每个患者子目录下有 dicom/ 子目录")
    parser.add_argument("--gpu", type=int, default=None,
                        help="指定 GPU ID (默认: 自动选择)")
    parser.add_argument("--wc", type=float, default=None,
                        help="指定窗位 WindowCenter (默认: 使用 DICOM 头中的值)")
    parser.add_argument("--ww", type=float, default=None,
                        help="指定窗宽 WindowWidth (默认: 使用 DICOM 头中的值)")
    args = parser.parse_args()

    # 设备
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device.index or 0)}")

    # 阈值
    threshold = args.threshold
    print(f"分类阈值: {threshold:.2f}")

    # 发现患者
    patients = discover_patients(args.input, args.dicom_subdir)
    if not patients:
        print(f"ERROR: 在 {args.input} 中未找到 DICOM 或 NIfTI 文件")
        print("  支持的输入格式:")
        print("    1. 目录下直接是 .dcm 文件")
        print("    2. 子目录是患者，每个子目录内有 .dcm 文件")
        print("    3. 子目录是患者，每个子目录下有 dicom/ 子目录 (加 --dicom-subdir)")
        print("    4. 目录下是 .nii.gz 文件")
        return

    print(f"发现 {len(patients)} 例患者")

    # 预处理
    volumes = []
    patient_ids = []
    import nibabel as nib

    for pid, info in patients.items():
        if info["type"] == "dicom":
            print(f"  预处理: {pid} (DICOM)")
            try:
                vol = preprocess_dicom_volume(info["path"], wc=args.wc, ww=args.ww)
            except Exception as e:
                print(f"    ERROR: {e}")
                continue
        elif info["type"] == "nifti":
            print(f"  加载: {pid} (NIfTI)")
            try:
                nii = nib.load(info["path"])
                vol = nii.get_fdata().astype(np.float32)
            except Exception as e:
                print(f"    ERROR: {e}")
                continue
        else:
            continue

        volumes.append(vol)
        patient_ids.append(pid)

    if not volumes:
        print("ERROR: 没有成功预处理的患者")
        return

    print(f"成功预处理 {len(volumes)} 例")

    # 推理
    if args.ensemble:
        model_names = ENSEMBLE_MODELS
        print(f"\n多模型集成: {', '.join(model_names)}")
        if args.tta:
            print("TTA: 启用 (水平翻转 + 旋转)")
        print(f"推理中...")
        results = run_ensemble_inference(volumes, patient_ids, model_names, device,
                                         use_tta=args.tta)
    else:
        print(f"\n模型: {args.model}")
        if args.tta:
            print("TTA: 启用 (水平翻转 + 旋转)")
        print(f"5-fold ensemble 推理中...")
        results = run_inference(volumes, patient_ids, args.model, device,
                                use_tta=args.tta)

    # 输出
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "probability", "std", "n_folds", "prediction"])
        for pid in patient_ids:
            r = results.get(pid, {"mean_prob": 0.0, "std_prob": 0.0, "n_folds": 0})
            pred = "异常" if r["mean_prob"] >= threshold else "正常"
            w.writerow([pid, f"{r['mean_prob']:.6f}", f"{r['std_prob']:.6f}",
                        r["n_folds"], pred])

    print(f"\n{'='*60}")
    print(f"预测结果 (阈值={threshold:.2f})")
    print(f"{'='*60}")
    probs = [results[pid]["mean_prob"] for pid in patient_ids]
    n_anomalous = sum(1 for p in probs if p >= threshold)
    n_normal = len(probs) - n_anomalous
    print(f"  总计: {len(probs)} 例")
    print(f"  预测异常 (概率≥{threshold:.2f}): {n_anomalous} 例 "
          f"({n_anomalous/len(probs)*100:.1f}%)")
    print(f"  预测正常 (概率<{threshold:.2f}): {n_normal} 例 "
          f"({n_normal/len(probs)*100:.1f}%)")
    print(f"  概率范围: [{min(probs):.4f}, {max(probs):.4f}]")
    print(f"  概率均值: {np.mean(probs):.4f} ± {np.std(probs):.4f}")
    print(f"\n结果保存至: {args.output}")

    # 详细列表
    print(f"\n{'PID':<30} {'Prob':>10} {'Std':>10} {'Prediction':>12}")
    print("-" * 65)
    for pid in patient_ids:
        r = results[pid]
        pred = "异常" if r["mean_prob"] >= threshold else "正常"
        print(f"{pid:<30} {r['mean_prob']:>10.4f} {r['std_prob']:>10.4f} {pred:>12}")


if __name__ == "__main__":
    main()