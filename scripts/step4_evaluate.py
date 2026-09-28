"""Step 4: SCI 期刊级评估（多模型对比）

自动检测所有已训练的模型结果，生成：
- Fig 1: 主模型 (resnet50) 患者级集成 ROC（与部署端 predict.py 的五折概率平均一致）
- Fig 2: 患者级集成混淆矩阵（总计数 = 独立患者数）
- Fig 3: 训练曲线
- Fig 4: 患者级集成校准曲线
- Fig 5: 多模型患者级集成 ROC 对比
- Table 1: 多模型指标对比表（患者级）
- DeLong 配对检验 p-values（患者级，单次比较）
"""
from __future__ import annotations

import os, csv, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")
TEST_CSV = os.path.join(PROJECT_ROOT, "data_split_test.csv")
N_FOLDS = 5

# ============================================================
# SCI 样式
# ============================================================
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.linewidth": 1.0, "axes.spines.top": False, "axes.spines.right": False,
})

MODEL_COLORS = {
    "resnet50": "#B2182B",
    "densenet121": "#2166AC",
    "single_slice": "#D6604D",
    "resnet18_3d": "#4DAF4A",
    "vit_b_16": "#8E44AD",
}
MODEL_LABELS = {
    "resnet50": "2.5D ResNet-50",
    "densenet121": "2.5D DenseNet-121",
    "single_slice": "Single-Slice ResNet-50",
    "resnet18_3d": "3D ResNet-18",
    "vit_b_16": "2.5D ViT-B/16",
}


# ============================================================
# 数据加载
# ============================================================
def discover_models():
    """自动发现所有已训练的模型"""
    models = {}
    for f in glob.glob(os.path.join(PROJECT_ROOT, "cv_results_*.csv")):
        name = os.path.basename(f).replace("cv_results_", "").replace(".csv", "")
        if os.path.exists(f):
            models[name] = {"results_csv": f}
    return models


def load_patient_level_preds(model_name):
    """读取模型各折 test_predictions.csv，按 patient_id 对齐生成患者级集成预测。

    与部署端 predict.py 的“五折概率平均”保持一致：每个患者一条记录，
    ensemble_prob = 各折 prob 的均值。

    Returns:
        pd.DataFrame: 列 [patient_id, true_label, ensemble_prob]

    Raises:
        ValueError: 各折患者集不一致、同一患者各折 true_label 不一致，
                    或 CSV 中 patient_id 重复（数据未对齐）。
    """
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", model_name)
    fold_dfs = []
    for fold_id in range(N_FOLDS):
        pred_path = os.path.join(ckpt_dir, f"fold_{fold_id}", "test_predictions.csv")
        if os.path.exists(pred_path):
            df = pd.read_csv(pred_path)
            if not {"patient_id", "true_label", "prob"}.issubset(df.columns):
                raise ValueError(f"{pred_path} 缺少列 patient_id/true_label/prob")
            df["patient_id"] = df["patient_id"].astype(str)
            df["true_label"] = df["true_label"].astype(int)
            df["prob"] = df["prob"].astype(float)
            if df["patient_id"].duplicated().any():
                raise ValueError(f"{pred_path} 存在重复 patient_id，无法对齐")
            fold_dfs.append(df)

    if not fold_dfs:
        return pd.DataFrame(columns=["patient_id", "true_label", "ensemble_prob"])

    # 断言各折患者集完全一致
    patient_sets = [set(df["patient_id"]) for df in fold_dfs]
    if any(s != patient_sets[0] for s in patient_sets[1:]):
        raise ValueError(
            f"模型 {model_name} 各折 test_predictions.csv 患者集不一致，数据未对齐 "
            f"(各折患者数: {[len(s) for s in patient_sets]})")

    # 同一患者各折 true_label 必须一致
    label_by_pid = {}
    for df in fold_dfs:
        for pid, tl in zip(df["patient_id"], df["true_label"]):
            if pid in label_by_pid and label_by_pid[pid] != tl:
                raise ValueError(
                    f"模型 {model_name} 患者 {pid} 各折 true_label 不一致: "
                    f"{label_by_pid[pid]} vs {tl}")
            label_by_pid[pid] = int(tl)

    # 按第一折（即 test_df）顺序输出，每个患者取各折 prob 均值
    probs_by_pid = [df.set_index("patient_id")["prob"] for df in fold_dfs]
    records = []
    for pid in fold_dfs[0]["patient_id"]:
        probs = [s.loc[pid] for s in probs_by_pid]
        records.append({
            "patient_id": pid,
            "true_label": label_by_pid[pid],
            "ensemble_prob": float(np.mean(probs)),
        })
    return pd.DataFrame(records)


def load_logs(model_name):
    """加载训练日志"""
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", model_name)
    logs = {}
    for fold_id in range(N_FOLDS):
        log_path = os.path.join(ckpt_dir, f"fold_{fold_id}", "epoch_log.csv")
        if os.path.exists(log_path):
            logs[fold_id] = pd.read_csv(log_path)
    return logs


# ============================================================
# DeLong 配对检验
# ============================================================
def delong_paired_auc_test(y_true, probs_a, probs_b):
    """DeLong et al. (1988) 配对 AUC 差异检验（结构分量法）。

    严格按 DeLong 结构分量公式：
      - 结构分量 ψ_k(i,j) = I[p_k(i) > p_k(j)] + 0.5·I[p_k(i) == p_k(j)]
        （i 为正样本，j 为负样本）
      - AUC_k = (1/(n1·n0)) Σ_i Σ_j ψ_k(i,j)（等价于排序去重阈值的
        Mann–Whitney 统计量，含平局 0.5 修正）
      - 正样本分量 D10_i = (1/n0) Σ_j (ψ_a(i,j) − ψ_b(i,j))
      - 负样本分量 D01_j = (1/n1) Σ_i (ψ_a(i,j) − ψ_b(i,j))
      - Var(θ_a − θ_b) = sample_var(D10)/n1 + sample_var(D01)/n0 (ddof=1)
      - se = sqrt(Var); z = (θ_a − θ_b)/se; p = 2·(1 − Φ(|z|))

    Args:
        y_true: 真实标签（0/1）
        probs_a / probs_b: 两个模型的预测概率（同一批样本，逐样本配对）

    Returns:
        (auc_a, auc_b, se, z, p)

    退化情形（统计上无法估计差异，语义为“无差异”）：
      - n1 == 0 或 n0 == 0：AUC 无定义
      - se 为 0 或 NaN：两模型结构分量完全一致（预测排序/平局相同），
        或 ddof=1 样本方差无定义（n1==1 或 n0==1）
    此时返回 se=0.0、z=0.0、p=1.0，并打印退化说明（不是静默吞错，
    实现仍严格按公式计算，仅在方差不可估计时给出“无差异”语义）。

    Notes:
        对称性：交换 probs_a/probs_b 后，z 变号、p 精确不变。
    """
    y_true = np.asarray(y_true).ravel()
    pa = np.asarray(probs_a, dtype=float).ravel()
    pb = np.asarray(probs_b, dtype=float).ravel()
    if not (len(y_true) == len(pa) == len(pb)):
        raise ValueError("y_true / probs_a / probs_b 长度不一致")

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n1, n0 = len(pos_idx), len(neg_idx)

    if n1 == 0 or n0 == 0:
        print(f"  [delong] 退化情形: 只有一类样本 (n1={n1}, n0={n0})，"
              f"AUC 无定义，返回 z=0.0, p=1.0")
        return 0.0, 0.0, 0.0, 0.0, 1.0

    # 结构分量矩阵: psi_k[i, j] = I[p_k(i)>p_k(j)] + 0.5·I[p_k(i)==p_k(j)]
    def _psi_matrix(probs):
        psi = np.empty((n1, n0), dtype=float)
        pp = probs[pos_idx]
        pn = probs[neg_idx]
        for i in range(n1):
            psi[i, :] = (pp[i] > pn).astype(float) + 0.5 * (pp[i] == pn).astype(float)
        return psi

    psi_a = _psi_matrix(pa)
    psi_b = _psi_matrix(pb)

    # AUC（结构分量平均，等价于排序去重阈值计算）
    auc_a = float(psi_a.sum() / (n1 * n0))
    auc_b = float(psi_b.sum() / (n1 * n0))

    # 结构分量差 → 正/负样本分量
    delta = psi_a - psi_b
    d10 = delta.mean(axis=1)  # D10_i = (1/n0) Σ_j (ψ_a − ψ_b)
    d01 = delta.mean(axis=0)  # D01_j = (1/n1) Σ_i (ψ_a − ψ_b)

    var = np.var(d10, ddof=1) / n1 + np.var(d01, ddof=1) / n0
    se = float(np.sqrt(var)) if var >= 0 else float("nan")

    if not np.isfinite(se) or se == 0.0:
        reason = ("两模型结构分量完全一致（预测排序/平局相同），方差为 0"
                  if se == 0.0
                  else "样本方差无定义（n1==1 或 n0==1）或方差为 NaN")
        print(f"  [delong] 退化情形: {reason}，返回 z=0.0, p=1.0（无差异）")
        return auc_a, auc_b, 0.0, 0.0, 1.0

    z = (auc_a - auc_b) / se
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return auc_a, auc_b, se, float(z), float(p)


# ============================================================
# Fig 1: 主模型患者级集成 ROC
# ============================================================
def plot_main_roc(patient_df, model_name, save_path):
    """主模型患者级集成 ROC（单条曲线，基于五折概率平均的 ensemble_prob）"""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    y = patient_df["true_label"].values
    p = patient_df["ensemble_prob"].values
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    ax.plot(fpr, tpr, color="#B2182B", lw=2.5,
            label=f"Holdout Ensemble (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.5)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title(f"{MODEL_LABELS.get(model_name, model_name)}\nHoldout Test Ensemble ROC",
                 fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.2, linestyle="--")
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()
    print(f"  Fig 1: ROC → {save_path}")


# ============================================================
# Fig 2: 患者级集成混淆矩阵
# ============================================================
def plot_confusion_matrix(patient_df, save_path):
    """患者级集成概率的混淆矩阵（总计数 == 独立患者数，不再累计 5 倍）"""
    y = patient_df["true_label"].values
    p = patient_df["ensemble_prob"].values
    preds = (p >= 0.5).astype(int)
    cm = confusion_matrix(y, preds)
    total = cm.sum()
    assert total == len(patient_df), \
        f"混淆矩阵总计数 ({total}) 必须等于独立患者数 ({len(patient_df)})"

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Negative", "Predicted Positive"])
    ax.set_yticklabels(["Actual Negative", "Actual Positive"])
    ax.set_title("Holdout Ensemble Confusion Matrix", fontweight="bold")
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=13, fontweight="bold")
    plt.colorbar(ax.imshow(cm, cmap=plt.cm.Blues), ax=ax, fraction=0.046, pad=0.04, label="Count")
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()
    print(f"  Fig 2: CM → {save_path}")


# ============================================================
# Fig 3: 训练曲线
# ============================================================
def plot_training_curves(all_logs, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    max_epochs = min(len(logs) for logs in all_logs.values())
    fold_colors = ["#2166AC", "#D6604D", "#4DAF4A", "#984EA3", "#FF7F00"]

    for i, fold_id in enumerate(sorted(all_logs.keys())):
        logs = all_logs[fold_id].iloc[:max_epochs]
        epochs = logs["epoch"].values
        ax1.plot(epochs, logs["train_loss"].values, color=fold_colors[i], lw=0.8, alpha=0.5)
        ax1.plot(epochs, logs["val_loss"].values, color=fold_colors[i], lw=1.0, alpha=0.8, linestyle="--")
        ax2.plot(epochs, logs["val_auc"].values, color=fold_colors[i], lw=1.0, alpha=0.8,
                 label=f"Fold {fold_id}")

    train_losses = np.array([all_logs[f]["train_loss"].values[:max_epochs] for f in sorted(all_logs.keys())])
    val_losses = np.array([all_logs[f]["val_loss"].values[:max_epochs] for f in sorted(all_logs.keys())])
    val_aucs = np.array([all_logs[f]["val_auc"].values[:max_epochs] for f in sorted(all_logs.keys())])
    epochs = np.arange(1, max_epochs + 1)

    ax1.plot(epochs, train_losses.mean(0), color="black", lw=2.0, label="Train (mean)")
    ax1.fill_between(epochs, train_losses.mean(0) - train_losses.std(0),
                     train_losses.mean(0) + train_losses.std(0), color="black", alpha=0.1)
    ax1.plot(epochs, val_losses.mean(0), color="black", lw=2.0, linestyle="--", label="Val (mean)")
    ax1.fill_between(epochs, val_losses.mean(0) - val_losses.std(0),
                     val_losses.mean(0) + val_losses.std(0), color="black", alpha=0.1)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss", fontweight="bold")
    ax1.legend(fontsize=8, frameon=False); ax1.grid(True, alpha=0.2, linestyle="--")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2.plot(epochs, val_aucs.mean(0), color="black", lw=2.0, label="Mean")
    ax2.fill_between(epochs, val_aucs.mean(0) - val_aucs.std(0),
                     val_aucs.mean(0) + val_aucs.std(0), color="black", alpha=0.1, label="±1 SD")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("AUC")
    ax2.set_title("Validation AUC", fontweight="bold")
    ax2.legend(fontsize=8, frameon=False); ax2.grid(True, alpha=0.2, linestyle="--")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()
    print(f"  Fig 3: Training → {save_path}")


# ============================================================
# Fig 4: 患者级集成校准曲线
# ============================================================
def plot_calibration(patient_df, save_path):
    """患者级集成概率的校准曲线（不拼接各折重复记录）"""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    y = patient_df["true_label"].values
    p = patient_df["ensemble_prob"].values
    prob_true, prob_pred = calibration_curve(y, p, n_bins=10, strategy="uniform")
    brier = brier_score_loss(y, p)
    ax.plot(prob_pred, prob_true, "o-", color="#B2182B", lw=2.5, markersize=8,
            label=f"Ensemble (Brier={brier:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.5)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
    ax.set_title("Holdout Ensemble Calibration Curve", fontweight="bold")
    ax.legend(loc="lower right", frameon=False); ax.grid(True, alpha=0.2, linestyle="--")
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()
    print(f"  Fig 4: Calibration → {save_path}")


# ============================================================
# Fig 5: 多模型患者级集成 ROC 对比
# ============================================================
def plot_multimodel_roc(models_data, save_path):
    """多模型 ROC 对比：各模型都用各自的患者级 ensemble_prob（同一批患者）"""
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # 取所有模型的公共患者集，保证对比基于同一批患者
    pid_sets = [set(df["patient_id"]) for df in models_data.values() if len(df) > 0]
    if not pid_sets:
        print("  Fig 5: 无可用数据，跳过多模型 ROC")
        return
    common_pids = set.intersection(*pid_sets)
    if not common_pids:
        print("  Fig 5: 各模型无公共患者集，跳过多模型 ROC")
        return
    common_pids = sorted(common_pids)

    for model_name in sorted(models_data.keys()):
        df = models_data[model_name]
        if len(df) == 0:
            continue
        sub = df[df["patient_id"].isin(common_pids)].set_index("patient_id")
        y = sub.loc[common_pids, "true_label"].values.astype(int)
        p = sub.loc[common_pids, "ensemble_prob"].values
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        color = MODEL_COLORS.get(model_name, "gray")
        label = MODEL_LABELS.get(model_name, model_name)
        ax.plot(fpr, tpr, color=color, lw=2.0,
                label=f"{label} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.5)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("1 - Specificity (False Positive Rate)")
    ax.set_ylabel("Sensitivity (True Positive Rate)")
    ax.set_title("Model Comparison: Holdout Ensemble ROC", fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.2, linestyle="--")
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()
    print(f"  Fig 5: Multi-Model ROC → {save_path}")


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print("=" * 60)
    print("Step 4: SCI 期刊级评估 (多模型对比)")
    print("=" * 60)

    models = discover_models()
    if not models:
        print("ERROR: 未找到 cv_results_*.csv 文件")
        return

    print(f"\n发现 {len(models)} 个模型: {list(models.keys())}")

    # 加载所有模型的患者级集成预测（与部署端 predict.py 一致）
    models_data = {}
    for model_name in models:
        patient_df = load_patient_level_preds(model_name)
        if len(patient_df) > 0:
            models_data[model_name] = patient_df
            auc = roc_auc_score(patient_df["true_label"], patient_df["ensemble_prob"])
            print(f"  {model_name}: {len(patient_df)} 例患者, ensemble AUC={auc:.4f}")

    # 主模型设为 resnet50 (或第一个)
    main_model = "resnet50" if "resnet50" in models_data else list(models_data.keys())[0]
    main_patient_df = models_data[main_model]

    # 生成单模型图表 (用主模型的患者级集成概率)
    print(f"\n--- 主模型 ({main_model}) 图表 ---")
    plot_main_roc(main_patient_df, main_model,
                  os.path.join(FIG_DIR, "fig1_roc_curves.png"))
    plot_confusion_matrix(main_patient_df,
                          os.path.join(FIG_DIR, "fig2_confusion_matrix.png"))

    main_logs = load_logs(main_model)
    if main_logs:
        plot_training_curves(main_logs, os.path.join(FIG_DIR, "fig3_training_curves.png"))
    plot_calibration(main_patient_df, os.path.join(FIG_DIR, "fig4_calibration.png"))

    # 多模型对比
    if len(models_data) > 1:
        print(f"\n--- 多模型对比 ---")
        plot_multimodel_roc(models_data, os.path.join(FIG_DIR, "fig5_multimodel_roc.png"))

        # 对比表（患者级指标）
        print(f"\n{'='*80}")
        print(f"  {'Model':<25} {'AUC':<12} {'Sens':<10} {'Spec':<10} {'Acc':<10} {'F1':<10}")
        print(f"  {'-'*78}")
        table_rows = []
        for model_name in sorted(models_data.keys()):
            df = models_data[model_name]
            y = df["true_label"].values
            p = df["ensemble_prob"].values
            preds = (p >= 0.5).astype(int)
            row = {
                "Model": MODEL_LABELS.get(model_name, model_name),
                "AUC": f"{roc_auc_score(y, p):.4f}",
                "Sensitivity": f"{recall_score(y, preds, pos_label=1):.4f}",
                "Specificity": f"{recall_score(y, preds, pos_label=0):.4f}",
                "Accuracy": f"{accuracy_score(y, preds):.4f}",
                "F1": f"{f1_score(y, preds, pos_label=1):.4f}",
            }
            print(f"  {row['Model']:<25} {row['AUC']:<12} {row['Sensitivity']:<10} "
                  f"{row['Specificity']:<10} {row['Accuracy']:<10} {row['F1']:<10}")
            table_rows.append(row)

        pd.DataFrame(table_rows).to_csv(os.path.join(FIG_DIR, "model_comparison.csv"), index=False)

        # DeLong 配对检验（患者级集成概率，单次比较；不再对多折 p 值取均值）
        print(f"\n{'='*80}")
        print(f"  DeLong Paired AUC Test (vs {main_model}, patient-level ensemble)")
        print(f"  {'='*80}")
        main_by_pid = main_patient_df.set_index("patient_id")
        for model_name in sorted(models_data.keys()):
            if model_name == main_model:
                continue
            df = models_data[model_name]
            other_by_pid = df.set_index("patient_id")
            common = [pid for pid in main_by_pid.index if pid in other_by_pid.index]
            if len(common) < 2:
                print(f"  {main_model} vs {model_name}: 公共患者不足 2 例，跳过")
                continue
            y = main_by_pid.loc[common, "true_label"].values.astype(int)
            p1 = main_by_pid.loc[common, "ensemble_prob"].values
            p2 = other_by_pid.loc[common, "ensemble_prob"].values
            auc_a, auc_b, se, z, p_val = delong_paired_auc_test(y, p1, p2)
            sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
            print(f"  {main_model} vs {model_name}: AUC={auc_a:.4f} vs {auc_b:.4f}, "
                  f"z={z:.3f}, p={p_val:.4f} {sig}")

    print(f"\n所有图表保存到: {FIG_DIR}/")
    print("Step 4 完成!")


if __name__ == "__main__":
    main()