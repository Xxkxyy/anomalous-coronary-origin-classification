"""共享预处理模块：训练/验证/预测三端统一入口

修复两个历史问题：
1. 预处理一致性：所有 DICOM 一律做 Rescale→HU + 窗宽窗位（默认 WC=40, WW=400）
   线性映射到 [0,1]，再 center-crop 160 → per-image min-max → 110×110×8；
   RGB 二级图像直接加权灰度（不做 HU/窗化）。
2. 空间信息缺失：用 ImageOrientationPatient × PixelSpacing + 首末层
   ImagePositionPatient 构建真实 4×4 affine（层间距用 3D 向量 norm 法），
   并按 SeriesInstanceUID 分组、按 InstanceNumber 排序、选层数最多的序列。

统一流水线（三端共用）:
  DICOM: read_dicom_series → build_dicom_affine → center_crop_160
         → min_max_norm → resize_110x110x8
  NIfTI: preprocess_nifti_volume（get_fdata → crop → norm → resize）
"""
from __future__ import annotations

import os

import numpy as np
import pydicom
import nibabel as nib
from scipy.ndimage import zoom

PREPROCESS_VERSION = "v2"

# 共享默认窗宽窗位（训练端/验证端/预测端必须一致）
DEFAULT_WINDOW_CENTER = 40.0
DEFAULT_WINDOW_WIDTH = 400.0

CROP_SIZE = 160
TARGET_SIZE = (110, 110, 8)


# ============================================================
# DICOM 头字段解析辅助
# ============================================================
def _vec3(v, default=None):
    """取前 3 个 float；缺失/非法 → default"""
    if v is None:
        return default
    try:
        return [float(x) for x in v[:3]]
    except (TypeError, ValueError):
        return default


def _vec2(v, default):
    """取前 2 个 float；缺失/非法 → default"""
    if v is None:
        return default
    try:
        return [float(x) for x in v[:2]]
    except (TypeError, ValueError):
        return default


def _vec6(v, default):
    """取前 6 个 float；缺失/非法 → default"""
    if v is None:
        return default
    try:
        return [float(x) for x in v[:6]]
    except (TypeError, ValueError):
        return default


# ============================================================
# DICOM 读取
# ============================================================
def read_dicom_series(dicom_dir: str, wc: float = None,
                      ww: float = None) -> tuple:
    """读取 DICOM 序列 → (volume [H,W,S] float32 已窗化到 [0,1], meta dict)

    处理逻辑:
      - 按 SeriesInstanceUID 分组，按 InstanceNumber 排序，选层数最多的序列
      - 原始 CT DICOM: RescaleSlope/RescaleIntercept → HU，
        窗宽窗位（默认 WC=40, WW=400，可被 wc/ww 覆盖）→ clip 映射到 [0,1]
      - RGB 二级图像: 0.299R+0.587G+0.114B 加权灰度（不做 HU/窗化）

    meta 包含: series_uid / n_slices / rescale_slope / rescale_intercept /
      window_center / window_width / pixel_spacing / orientation /
      position_first / position_last / applied（实际执行了哪些处理）
    """
    if wc is None:
        wc = DEFAULT_WINDOW_CENTER
    if ww is None:
        ww = DEFAULT_WINDOW_WIDTH

    groups = {}  # series_uid -> list of (instance_number, pix, ds, applied_rescale, applied_window, applied_rgb)
    for f in sorted(os.listdir(dicom_dir)):
        if not f.endswith('.dcm'):
            continue
        d = pydicom.dcmread(os.path.join(dicom_dir, f))
        pix = d.pixel_array.astype(np.float32)

        if pix.ndim == 3 and pix.shape[2] == 3:
            # RGB 二级图像（如 PACS 截图）: RGB → 灰度
            pix = 0.299 * pix[:, :, 0] + 0.587 * pix[:, :, 1] + 0.114 * pix[:, :, 2]
            applied_rescale, applied_window, applied_rgb = False, False, True
        else:
            # 原始 CT DICOM: Rescale → HU, 窗宽窗位 → [0, 1]
            slope = float(getattr(d, 'RescaleSlope', 1) or 1)
            intercept = float(getattr(d, 'RescaleIntercept', 0) or 0)
            pix = pix * slope + intercept
            low = wc - ww / 2
            pix = np.clip((pix - low) / ww, 0.0, 1.0)
            applied_rescale, applied_window, applied_rgb = True, True, False

        uid = str(getattr(d, 'SeriesInstanceUID', None) or 'unknown')
        groups.setdefault(uid, []).append(
            (int(getattr(d, 'InstanceNumber', 0) or 0), pix, d,
             applied_rescale, applied_window, applied_rgb))

    if not groups:
        raise FileNotFoundError(f"未在 {dicom_dir} 中找到 .dcm 文件")

    # 选层数最多的序列
    best_uid = max(groups, key=lambda u: len(groups[u]))
    slices = groups[best_uid]
    slices.sort(key=lambda x: x[0])

    vol = np.stack([s[1] for s in slices], axis=2)  # [H, W, S]

    first, last = slices[0][2], slices[-1][2]
    applied = {
        "rescale": slices[0][3],
        "windowing": slices[0][4],
        "rgb_to_gray": slices[0][5],
    }
    meta = {
        "series_uid": best_uid,
        "n_slices": len(slices),
        "rescale_slope": float(getattr(first, 'RescaleSlope', 1) or 1),
        "rescale_intercept": float(getattr(first, 'RescaleIntercept', 0) or 0),
        "window_center": float(wc),
        "window_width": float(ww),
        "pixel_spacing": _vec2(getattr(first, 'PixelSpacing', None), (1.0, 1.0)),
        "orientation": _vec6(getattr(first, 'ImageOrientationPatient', None),
                             (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
        "position_first": _vec3(getattr(first, 'ImagePositionPatient', None)),
        "position_last": _vec3(getattr(last, 'ImagePositionPatient', None)),
        "applied": applied,
    }
    return vol, meta


# ============================================================
# 真实 4×4 affine 构建（不再用 np.eye(4)）
# ============================================================
def build_dicom_affine(meta_or_slices) -> np.ndarray:
    """构建 DICOM 真实 4×4 affine（输入为 meta dict 或 pydicom Dataset 列表）

    affine 约定（nibabel）:
      - 第 0 列 = 列方向余弦 × PixelSpacing[1]（volume 轴 0 = 列索引）
      - 第 1 列 = 行方向余弦 × PixelSpacing[0]（volume 轴 1 = 行索引）
      - 第 2 列 = 层方向 = row×col 叉积 × 层间距
        （层间距 = ||pos_last - pos_first|| / (n-1)，3D 向量 norm 法）
      - 第 3 列 = 首层 ImagePositionPatient
    缺失标签回退默认: IOP=(1,0,0,0,1,0)、PixelSpacing=(1,1)、层间距=1、
    origin=(0,0,0)。
    """
    if isinstance(meta_or_slices, dict):
        meta = meta_or_slices
        n = int(meta.get("n_slices", 1) or 1)
        iop = meta.get("orientation") or (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        ps = meta.get("pixel_spacing") or (1.0, 1.0)
        pos_first = meta.get("position_first")
        pos_last = meta.get("position_last")
    else:
        slices = list(meta_or_slices)
        n = len(slices)
        first, last = slices[0], slices[-1]
        iop = _vec6(getattr(first, 'ImageOrientationPatient', None),
                    (1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        ps = _vec2(getattr(first, 'PixelSpacing', None), (1.0, 1.0))
        pos_first = _vec3(getattr(first, 'ImagePositionPatient', None))
        pos_last = _vec3(getattr(last, 'ImagePositionPatient', None))

    row_cos = np.array(iop[0:3], dtype=np.float64)  # 行方向余弦（沿行索引 j）
    col_cos = np.array(iop[3:6], dtype=np.float64)  # 列方向余弦（沿列索引 i）
    row_spacing = float(ps[0])
    col_spacing = float(ps[1])

    # 层间距: 首末层 ImagePositionPatient 3D 距离 / (n-1)
    gap = 1.0
    if pos_first is not None and pos_last is not None and n > 1:
        d = np.array(pos_last, dtype=np.float64) - np.array(pos_first, dtype=np.float64)
        gap = float(np.linalg.norm(d)) / (n - 1)
        if not np.isfinite(gap) or gap <= 0:
            gap = 1.0
    slice_vec = np.cross(row_cos, col_cos) * gap  # 层方向 = 行×列 法向

    origin = np.array(pos_first, dtype=np.float64) if pos_first is not None else np.zeros(3)

    affine = np.eye(4)
    affine[0:3, 0] = col_cos * col_spacing
    affine[0:3, 1] = row_cos * row_spacing
    affine[0:3, 2] = slice_vec
    affine[0:3, 3] = origin
    return affine


# ============================================================
# affine 传播
# ============================================================
def adjust_affine_crop(affine: np.ndarray, r0: float, c0: float) -> np.ndarray:
    """裁剪后平移调整: origin += A[:,0]*c0 + A[:,1]*r0（r0/c0 为相对原始体数据的偏移）"""
    a = np.array(affine, dtype=np.float64)
    a[0:3, 3] = a[0:3, 3] + a[0:3, 0] * float(c0) + a[0:3, 1] * float(r0)
    return a


def adjust_affine_zoom(affine: np.ndarray, zoom_factors) -> np.ndarray:
    """resize 后 affine 列缩放: 新列 = 旧列 / zoom_factor"""
    a = np.array(affine, dtype=np.float64)
    for i in range(3):
        a[0:3, i] = a[0:3, i] / float(zoom_factors[i])
    return a


# ============================================================
# 流水线各阶段
# ============================================================
def center_crop_160(volume: np.ndarray, affine: np.ndarray = None):
    """中心裁剪 160×160（输入 <160 时先中心 padding 再裁剪）

    Returns:
        (cropped [160,160,S], offsets (r0-pr, c0-pc) 相对原始体数据的偏移,
         adjusted_affine or None)
    """
    h, w = volume.shape[0], volume.shape[1]
    pr = max(0, (CROP_SIZE - h) // 2)
    pc = max(0, (CROP_SIZE - w) // 2)
    if pr or pc:
        volume = np.pad(volume, ((pr, pr), (pc, pc), (0, 0)), mode="constant")
    h2, w2 = volume.shape[0], volume.shape[1]
    cy, cx = h2 // 2, w2 // 2
    half = CROP_SIZE // 2
    r0, c0 = cy - half, cx - half
    cropped = volume[r0:r0 + CROP_SIZE, c0:c0 + CROP_SIZE, :]
    offsets = (r0 - pr, c0 - pc)  # 相对原始体数据的偏移（含 padding 补偿）
    if affine is not None:
        affine = adjust_affine_crop(affine, offsets[0], offsets[1])
    return cropped, offsets, affine


def min_max_norm(volume: np.ndarray) -> np.ndarray:
    """per-image min-max 归一化到 [0,1]（dmax==dmin 时保持不变）"""
    data = np.asarray(volume, dtype=np.float32)
    dmin, dmax = data.min(), data.max()
    if dmax > dmin:
        data = (data - dmin) / (dmax - dmin)
    return data


def resize_110x110x8(volume: np.ndarray, affine: np.ndarray = None):
    """scipy.ndimage.zoom order=1 缩放到 110×110×8

    Returns:
        (resized float32 [110,110,8], adjusted_affine or None)
    """
    data = np.asarray(volume, dtype=np.float32)
    zoom_factors = (TARGET_SIZE[0] / data.shape[0],
                    TARGET_SIZE[1] / data.shape[1],
                    TARGET_SIZE[2] / data.shape[2])
    data = zoom(data, zoom_factors, order=1)
    if affine is not None:
        affine = adjust_affine_zoom(affine, zoom_factors)
    return data.astype(np.float32), affine


# ============================================================
# 完整流水线（三端统一入口）
# ============================================================
def preprocess_dicom_volume(dicom_dir: str, wc: float = None,
                            ww: float = None) -> tuple:
    """DICOM 目录 → (volume [110,110,8] float32, meta, affine)

    流水线: read_dicom_series（Rescale+窗化[0,1]）→ center_crop_160
            → min_max_norm → resize_110x110x8
    """
    vol, meta = read_dicom_series(dicom_dir, wc=wc, ww=ww)
    affine = build_dicom_affine(meta)
    vol, _offsets, affine = center_crop_160(vol, affine)
    vol = min_max_norm(vol)
    vol, affine = resize_110x110x8(vol, affine)
    return vol.astype(np.float32), meta, affine


def preprocess_nifti_volume(nifti_path: str) -> tuple:
    """NIfTI → (volume [110,110,8] float32, affine)

    4D 取第一 3D；保留 NIfTI 自带 affine 并沿流水线传播；
    同样执行 center_crop_160 → min_max_norm → resize_110x110x8。
    """
    nii = nib.load(nifti_path)
    data = nii.get_fdata().astype(np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"NIfTI 必须为 3D/4D，当前 shape={data.shape}")
    affine = np.array(nii.affine, dtype=np.float64)
    data, _offsets, affine = center_crop_160(data, affine)
    data = min_max_norm(data)
    data, affine = resize_110x110x8(data, affine)
    return data.astype(np.float32), affine


def qc_summary(volume: np.ndarray, meta: dict) -> str:
    """单行质控摘要（用于日志/JSON）"""
    return ("; ".join([
        f"preprocess_version={PREPROCESS_VERSION}",
        f"shape={volume.shape}",
        f"range=[{float(volume.min()):.4f}, {float(volume.max()):.4f}]",
        f"series_uid={meta.get('series_uid')}",
        f"n_slices={meta.get('n_slices')}",
        f"window_center={meta.get('window_center')}",
        f"window_width={meta.get('window_width')}",
        f"applied={meta.get('applied')}",
    ]))
