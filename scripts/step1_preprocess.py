"""Step 1: DICOM → NIfTI 预处理
- DICOM 序列 → 3D NIfTI（RGB自动转灰度）
- 160×160 中心裁剪
- per-image min-max 归一化 → 110×110×8
"""
import os, re, shutil, tempfile
import numpy as np
import pydicom
import nibabel as nib
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data')
NIFTI_RAW = os.path.join(PROJECT_ROOT, 'nifti')
NIFTI_ROI = os.path.join(PROJECT_ROOT, 'roi')
NIFTI_NORM = os.path.join(PROJECT_ROOT, 'nifti_norm')

CROP_SIZE = 160
TARGET_SIZE = (110, 110, 8)


# ============================================================
# 1.1 DICOM → NIfTI
# ============================================================
def _read_dicom_series(dicom_dir):
    """读取整个 DICOM 序列，按 InstanceNumber 排序，RGB→灰度"""
    slices = []
    for f in sorted(os.listdir(dicom_dir)):
        if not f.endswith('.dcm'):
            continue
        d = pydicom.dcmread(os.path.join(dicom_dir, f))
        pix = d.pixel_array.astype(np.float32)
        if pix.ndim == 3 and pix.shape[2] == 3:
            pix = 0.299 * pix[:,:,0] + 0.587 * pix[:,:,1] + 0.114 * pix[:,:,2]
        slices.append((d.InstanceNumber, pix))
    slices.sort(key=lambda x: x[0])
    return np.stack([s[1] for s in slices], axis=2)


def convert_one(patient_dir, label, out_dir):
    """将单个患者 DICOM 序列转为 NIfTI"""
    dicom_dir = os.path.join(patient_dir, 'dicom')
    if not os.path.isdir(dicom_dir):
        return "跳过", f"无 dicom/ 子目录"
    patient_name = os.path.basename(patient_dir)
    try:
        vol = _read_dicom_series(dicom_dir)
        nii = nib.Nifti1Image(vol, np.eye(4))
        out_path = os.path.join(out_dir, f'{patient_name}.nii.gz')
        os.makedirs(out_dir, exist_ok=True)
        nib.save(nii, out_path)
        return "成功", vol.shape
    except Exception as e:
        return "失败", str(e)


def convert_all():
    """批量转换 data/{0,1} 下所有患者"""
    print("=" * 50)
    print("Step 1.1: DICOM → NIfTI")
    print("=" * 50)
    stats = {'成功': 0, '失败': 0, '跳过': 0}
    for label in ['0', '1']:
        src = os.path.join(DATA_ROOT, label)
        dst = os.path.join(NIFTI_RAW, label)
        os.makedirs(dst, exist_ok=True)
        for subdir in sorted(os.listdir(src)):
            patient_dir = os.path.join(src, subdir)
            if not os.path.isdir(patient_dir):
                continue
            status, info = convert_one(patient_dir, label, dst)
            if status == '成功':
                stats['成功'] += 1
            else:
                stats[status] += 1
                print(f"  [{status}] {subdir}: {info}")
    print(f"  完成: 成功 {stats['成功']}, 失败 {stats['失败']}, 跳过 {stats['跳过']}")


# ============================================================
# 1.2 中心裁剪 160×160
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
            h, w = data.shape[0], data.shape[1]
            cy, cx = h // 2, w // 2
            half = CROP_SIZE // 2
            cropped = data[cy - half:cy + half, cx - half:cx + half, :]
            nib.save(nib.Nifti1Image(cropped, np.eye(4)),
                     os.path.join(dst, f))
    print(f"  完成: 全部裁剪为 {CROP_SIZE}×{CROP_SIZE}")


# ============================================================
# 1.3 per-image min-max 归一化 → 110×110×8
# ============================================================
def normalize_one(src_path, dst_path):
    nii = nib.load(src_path)
    data = nii.get_fdata().astype(np.float32)
    # per-image min-max 归一化到 [0, 1]
    dmin, dmax = data.min(), data.max()
    if dmax > dmin:
        data = (data - dmin) / (dmax - dmin)
    # resize
    from scipy.ndimage import zoom
    zoom_factors = (TARGET_SIZE[0] / data.shape[0],
                    TARGET_SIZE[1] / data.shape[1],
                    TARGET_SIZE[2] / data.shape[2])
    data = zoom(data, zoom_factors, order=1)
    nib.save(nib.Nifti1Image(data, np.eye(4)), dst_path)


def normalize_all():
    print("\n" + "=" * 50)
    print("Step 1.3: min-max 归一化 → 110×110×8")
    print("=" * 50)
    for label in ['0', '1']:
        src = os.path.join(NIFTI_ROI, label)
        dst = os.path.join(NIFTI_NORM, label)
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            if f.endswith('.nii.gz'):
                normalize_one(os.path.join(src, f), os.path.join(dst, f))
    print(f"  完成: 全部归一化 → {TARGET_SIZE}")


# ============================================================
# Main
# ============================================================
def main():
    convert_all()
    crop_center_all()
    normalize_all()
    print("\nStep 1 完成!")


if __name__ == '__main__':
    main()