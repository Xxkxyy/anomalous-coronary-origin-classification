"""Step 1: DICOM → NIfTI 预处理
- DICOM 序列 → 3D NIfTI（RGB自动转灰度，原始 CT 做 Rescale+窗宽窗位）
- 160×160 中心裁剪
- per-image min-max 归一化 → 110×110×8
- 使用真实 affine（ImageOrientationPatient/PixelSpacing/ImagePositionPatient），
  不再用 np.eye(4)；每阶段 affine 沿流水线传播
- 输出 nifti_norm/preprocess_meta.json（预处理元数据 sidecar）

预处理逻辑统一由 scripts/preprocessing.py 提供（与 external_validation.py /
predict.py 完全一致），此处仅保留三阶段磁盘结构。
"""
import os, json, shutil, tempfile
import numpy as np
import nibabel as nib
import warnings
warnings.filterwarnings('ignore')

try:
    from scripts.preprocessing import (
        read_dicom_series, build_dicom_affine, center_crop_160, min_max_norm,
        resize_110x110x8, CROP_SIZE, TARGET_SIZE, PREPROCESS_VERSION,
        DEFAULT_WINDOW_CENTER, DEFAULT_WINDOW_WIDTH,
    )
except ImportError:  # 直接以 python scripts/step1_preprocess.py 运行时 sys.path[0]=scripts
    from preprocessing import (
        read_dicom_series, build_dicom_affine, center_crop_160, min_max_norm,
        resize_110x110x8, CROP_SIZE, TARGET_SIZE, PREPROCESS_VERSION,
        DEFAULT_WINDOW_CENTER, DEFAULT_WINDOW_WIDTH,
    )

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
NIFTI_RAW = os.path.join(PROJECT_ROOT, 'nifti')
NIFTI_ROI = os.path.join(PROJECT_ROOT, 'roi')
NIFTI_NORM = os.path.join(PROJECT_ROOT, 'nifti_norm')


# ============================================================
# 1.1 DICOM → NIfTI
# ============================================================
def convert_one(patient_dir, label, out_dir):
    """将单个患者 DICOM 序列转为 NIfTI（带真实 affine）"""
    dicom_dir = os.path.join(patient_dir, 'dicom')
    if not os.path.isdir(dicom_dir):
        return "跳过", f"无 dicom/ 子目录", None
    patient_name = os.path.basename(patient_dir)
    try:
        vol, meta = read_dicom_series(dicom_dir)
        affine = build_dicom_affine(meta)
        nii = nib.Nifti1Image(vol, affine)
        out_path = os.path.join(out_dir, f'{patient_name}.nii.gz')
        os.makedirs(out_dir, exist_ok=True)
        nib.save(nii, out_path)
        return "成功", vol.shape, meta
    except Exception as e:
        return "失败", str(e), None


def convert_all():
    """批量转换 data/{0,1} 下所有患者；返回 {label: {patient: meta}}"""
    print("=" * 50)
    print("Step 1.1: DICOM → NIfTI (Rescale + 窗宽窗位 WC="
          f"{DEFAULT_WINDOW_CENTER}, WW={DEFAULT_WINDOW_WIDTH})")
    print("=" * 50)
    stats = {'成功': 0, '失败': 0, '跳过': 0}
    patient_metas = {}
    for label in ['0', '1']:
        src = os.path.join(DATA_ROOT, label)
        dst = os.path.join(NIFTI_RAW, label)
        os.makedirs(dst, exist_ok=True)
        patient_metas[label] = {}
        for subdir in sorted(os.listdir(src)):
            patient_dir = os.path.join(src, subdir)
            if not os.path.isdir(patient_dir):
                continue
            status, info, meta = convert_one(patient_dir, label, dst)
            if status == '成功':
                stats['成功'] += 1
                meta['label'] = label
                patient_metas[label][subdir] = meta
            else:
                stats[status] += 1
                print(f"  [{status}] {subdir}: {info}")
    print(f"  完成: 成功 {stats['成功']}, 失败 {stats['失败']}, 跳过 {stats['跳过']}")
    return patient_metas


# ============================================================
# 1.2 中心裁剪 160×160（affine 平移同步调整）
# ============================================================
def crop_center_all():
    print("\n" + "=" * 50)
    print("Step 1.2: 中心裁剪 160×160")
    print("=" * 50)
    for label in ['0', '1']:
        src = os.path.join(NIFTI_RAW, label)
        dst = os.path.join(NIFTI_ROI, label)
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            if not f.endswith('.nii.gz'):
                continue
            nii = nib.load(os.path.join(src, f))
            data = nii.get_fdata().astype(np.float32)
            affine = np.array(nii.affine, dtype=np.float64)
            cropped, _offsets, affine = center_crop_160(data, affine)
            nib.save(nib.Nifti1Image(cropped, affine), os.path.join(dst, f))
    print(f"  完成: 全部裁剪为 {CROP_SIZE}×{CROP_SIZE}")


# ============================================================
# 1.3 per-image min-max 归一化 → 110×110×8（affine 缩放同步调整）
# ============================================================
def normalize_one(src_path, dst_path):
    """归一化 + resize 单个文件，返回最终 affine（用于元数据 sidecar）"""
    nii = nib.load(src_path)
    data = nii.get_fdata().astype(np.float32)
    affine = np.array(nii.affine, dtype=np.float64)
    data = min_max_norm(data)
    data, affine = resize_110x110x8(data, affine)
    nib.save(nib.Nifti1Image(data, affine), dst_path)
    return affine


def normalize_all(patient_metas):
    """批量归一化 → 110×110×8，并在 nifti_norm/ 写 preprocess_meta.json"""
    print("\n" + "=" * 50)
    print("Step 1.3: min-max 归一化 → 110×110×8")
    print("=" * 50)
    for label in ['0', '1']:
        src = os.path.join(NIFTI_ROI, label)
        dst = os.path.join(NIFTI_NORM, label)
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            if f.endswith('.nii.gz'):
                pid = f.replace('.nii.gz', '')
                final_affine = normalize_one(os.path.join(src, f), os.path.join(dst, f))
                meta = patient_metas.get(label, {}).get(pid)
                if meta is not None:
                    meta['affine'] = final_affine.tolist()
    print(f"  完成: 全部归一化 → {TARGET_SIZE}")

    meta_path = os.path.join(NIFTI_NORM, 'preprocess_meta.json')
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
# Main
# ============================================================
def main():
    patient_metas = convert_all()
    crop_center_all()
    normalize_all(patient_metas)
    print("\nStep 1 完成!")


if __name__ == '__main__':
    main()
