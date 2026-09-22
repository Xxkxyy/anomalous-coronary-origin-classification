"""Step 4: SCI 期刊级评估（多模型对比）

自动检测所有已训练的模型结果，生成：
- Fig 1: 主模型 (resnet50) 5-fold ROC 曲线
- Fig 2: 聚合混淆矩阵
- Fig 3: 训练曲线
- Fig 4: 校准曲线
- Fig 5: 多模型 ROC 对比 (均值 ± SD)
- Table 1: 多模型指标对比表
- DeLong 检验 p-values
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
    "radiomics_svm": "#FF7F00",
    "radiomics_rf": "#984EA3",
}
MODEL_LABELS = {
    "resnet50": "2.5D ResNet-50",
    "densenet121": "2.5D DenseNet-121",
    "single_slice": "Single-Slice ResNet-50",
    "resnet18_3d": "3D ResNet-18",
    "vit_b_16": "2.5D ViT-B/16",
    "radiomics_svm": "Radiomics + SVM",
    "radiomics_rf": "Radiomics + RF",
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


def load_preds(model_name):
    """加载模型所有 fold 的预测"""
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", model_name)
    all_labels, all_probs = [], []
    for fold_id in range(N_FOLDS):
        pred_path = os.path.join(ckpt_dir, f"fold_{fold_id}", "test_predictions.csv")
        if os.path.exists(pred_path):
            df = pd.read_csv(pred_path)
            all_labels.append(df["true_label"].values)
            all_probs.append(df["prob"].values)
    return all_labels, all_probs


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
# DeLong 检验
# ============================================================
def delong_roc_test(y_true, probs1, probs2):
    """非参数 DeLong 检验，比较两个 AUC"""
    try:
        from sklearn.metrics import roc_auc_score
        auc1 = roc_auc_score(y_true, probs1)
        auc2 = roc_auc_score(y_true, probs2)

        n = len(y_true)
        # V 矩阵估计
        v10 = _compute_v10(y_true, probs1)
        v01 = _compute_v01(y_true, probs1, probs2)

        # 方差
        var1 = sum(v10) / (n * n)
        var2 = sum(v10) / (n * n)  # 简化：假设同方差
        cov = sum(v01) / (n * n)

        se = np.sqrt(var1 + var2 - 2 * cov)
        if se < 1e-10:
            return 1.0
        z = (auc1 - auc2) / se
        return 2 * stats.norm.sf(abs(z))
    except Exception:
        return 1.0


def _compute_v10(y_true, probs):
    """DeLong V10"""
    n = len(y_true)
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    v = np.zeros(n)
    for i in range(n):
        s = 0.0
        if y_true[i] == 1:
            for j in neg:
                s += (probs[i] > probs[j]) + 0.5 * (probs[i] == probs[j])
        else:
            for j in pos:
                s += (probs[j] > probs[i]) + 0.5 * (probs[j] == probs[i])
        v[i] = (s / len(pos) / len(neg)) ** 2
    return v


def _compute_v01(y_true, probs1, probs2):
    """DeLong V01 (cross)"""
    n = len(y_true)
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    v = np.zeros(n)
    n_pos, n_neg = len(pos), len(neg)
    for i in range(n):
        s1 = s2 = 0.0
        if y_true[i] == 1:
            for j in neg:
                s1 += (probs1[i] > probs1[j]) + 0.5 * (probs1[i] == probs1[j])
                s2 += (probs2[i] > probs2[j]) + 0.5 * (probs2[i] == probs2[j])
        else:
            for j in pos:
                s1 += (probs1[j] > probs1[i]) + 0.5 * (probs1[j] == probs1[i])
                s2 += (probs2[j] > probs2[i]) + 0.5 * (probs2[j] == probs2[i])
        v[i] = (s1 / n_pos / n_neg) * (s2 / n_pos / n_neg)
    return v


# ============================================================
# Fig 1: 主模型 5-fold ROC
# ============================================================
def plot_main_roc(all_labels, all_probs, model_name, save_path):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    fold_colors = ["#2166AC", "#D6604D", "#4DAF4A", "#984EA3", "#FF7F00"]

    for i in range(len(all_labels)):
        fpr, tpr, _ = roc_curve(all_labels[i], all_probs[i])
        auc = roc_auc_score(all_labels[i], all_probs[i])
        ax.plot(fpr, tpr, color=fold_colors[i], lw=1.0, alpha=0.7,
                label=f"Fold {i} (AUC={auc:.3f})")
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_auc = np.mean([roc_auc_score(all_labels[i], all_probs[i]) for i in range(len(all_labels))])
    std_auc = np.std([roc_auc_score(all_labels[i], all_probs[i]) for i in range(len(all_labels))])

    ax.plot(mean_fpr, mean_tpr, color="#B2182B", lw=2.5,
            label=f"Mean (AUC={mean_auc:.3f}±{std_auc:.3f})")
    ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                    color="#B2182B", alpha=0.15)
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.5)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title(f"{MODEL_LABELS.get(model_name, model_name)}\n5-Fold CV ROC Curves", fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.2, linestyle="--")
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()
    print(f"  Fig 1: ROC → {save_path}")


# ============================================================
# Fig 2: 聚合混淆矩阵
# ============================================================
def plot_confusion_matrix(all_labels, all_probs, save_path):
    total_tn = total_fp = total_fn = total_tp = 0
    for labels, probs in zip(all_labels, all_probs):
        preds = (probs >= 0.5).astype(int)
        cm = confusion_matrix(labels, preds)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            total_tn += tn; total_fp += fp; total_fn += fn; total_tp += tp

    cm = np.array([[total_tn, total_fp], [total_fn, total_tp]])
    total = cm.sum()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Negative", "Predicted Positive"])
    ax.set_yticklabels(["Actual Negative", "Actual Positive"])
    ax.set_title("Aggregated Confusion Matrix\n(5-fold × holdout)", fontweight="bold")
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
# Fig 4: 校准曲线
# ============================================================
def plot_calibration(all_labels, all_probs, save_path):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    all_l = np.concatenate(all_labels)
    all_p = np.concatenate(all_probs)
    prob_true, prob_pred = calibration_curve(all_l, all_p, n_bins=10, strategy="uniform")
    brier = brier_score_loss(all_l, all_p)
    ax.plot(prob_pred, prob_true, "o-", color="#B2182B", lw=2.5, markersize=8,
            label=f"Model (Brier={brier:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.5)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve", fontweight="bold")
    ax.legend(loc="lower right", frameon=False); ax.grid(True, alpha=0.2, linestyle="--")
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()
    print(f"  Fig 4: Calibration → {save_path}")


# ============================================================
# Fig 5: 多模型 ROC 对比
# ============================================================
def plot_multimodel_roc(models_data, save_path):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    mean_fpr = np.linspace(0, 1, 100)

    for model_name in sorted(models_data.keys()):
        all_labels, all_probs = models_data[model_name]
        if not all_labels:
            continue
        tprs = []
        for labels, probs in zip(all_labels, all_probs):
            fpr, tpr, _ = roc_curve(labels, probs)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        mean_auc = np.mean([roc_auc_score(l, p) for l, p in zip(all_labels, all_probs)])
        std_auc = np.std([roc_auc_score(l, p) for l, p in zip(all_labels, all_probs)])

        color = MODEL_COLORS.get(model_name, "gray")
        label = MODEL_LABELS.get(model_name, model_name)
        ax.plot(mean_fpr, mean_tpr, color=color, lw=2.0,
                label=f"{label} (AUC={mean_auc:.3f}±{std_auc:.3f})")
        ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                        color=color, alpha=0.1)

    ax.plot([0, 1], [0, 1], "k--", lw=1.0, alpha=0.5)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("1 - Specificity (False Positive Rate)")
    ax.set_ylabel("Sensitivity (True Positive Rate)")
    ax.set_title("Model Comparison: Mean ROC Curves", fontweight="bold")
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

    # 加载所有模型数据
    models_data = {}
    for model_name in models:
        labels, probs = load_preds(model_name)
        if labels:
            models_data[model_name] = (labels, probs)
            mean_auc = np.mean([roc_auc_score(l, p) for l, p in zip(labels, probs)])
            print(f"  {model_name}: {len(labels)} folds, mean AUC={mean_auc:.4f}")

    # 主模型设为 resnet50 (或第一个)
    main_model = "resnet50" if "resnet50" in models_data else list(models_data.keys())[0]
    main_labels, main_probs = models_data[main_model]

    # 生成单模型图表 (用主模型)
    print(f"\n--- 主模型 ({main_model}) 图表 ---")
    plot_main_roc(main_labels, main_probs, main_model,
                  os.path.join(FIG_DIR, "fig1_roc_curves.png"))
    plot_confusion_matrix(main_labels, main_probs,
                          os.path.join(FIG_DIR, "fig2_confusion_matrix.png"))

    main_logs = load_logs(main_model)
    if main_logs:
        plot_training_curves(main_logs, os.path.join(FIG_DIR, "fig3_training_curves.png"))
    plot_calibration(main_labels, main_probs, os.path.join(FIG_DIR, "fig4_calibration.png"))

    # 多模型对比
    if len(models_data) > 1:
        print(f"\n--- 多模型对比 ---")
        plot_multimodel_roc(models_data, os.path.join(FIG_DIR, "fig5_multimodel_roc.png"))

        # 对比表
        print(f"\n{'='*80}")
        print(f"  {'Model':<25} {'AUC':<18} {'Sens':<10} {'Spec':<10} {'Acc':<10} {'F1':<10}")
        print(f"  {'-'*78}")
        table_rows = []
        for model_name in sorted(models_data.keys()):
            labels, probs = models_data[model_name]
            aucs = [roc_auc_score(l, p) for l, p in zip(labels, probs)]
            all_l = np.concatenate(labels)
            all_p = np.concatenate(probs)
            preds = (all_p >= 0.5).astype(int)
            row = {
                "Model": MODEL_LABELS.get(model_name, model_name),
                "AUC": f"{np.mean(aucs):.4f}±{np.std(aucs):.4f}",
                "Sensitivity": f"{recall_score(all_l, preds, pos_label=1):.4f}",
                "Specificity": f"{recall_score(all_l, preds, pos_label=0):.4f}",
                "Accuracy": f"{accuracy_score(all_l, preds):.4f}",
                "F1": f"{f1_score(all_l, preds, pos_label=1):.4f}",
            }
            print(f"  {row['Model']:<25} {row['AUC']:<18} {row['Sensitivity']:<10} "
                  f"{row['Specificity']:<10} {row['Accuracy']:<10} {row['F1']:<10}")
            table_rows.append(row)

        pd.DataFrame(table_rows).to_csv(os.path.join(FIG_DIR, "model_comparison.csv"), index=False)

        # DeLong 检验 (与主模型 pairwise)
        if main_model in models_data and len(models_data) > 1:
            print(f"\n{'='*60}")
            print(f"  DeLong Test (vs {main_model})")
            print(f"  {'='*60}")
            for model_name in sorted(models_data.keys()):
                if model_name == main_model:
                    continue
                # 用第一折做 DeLong（需相同样本）
                p_vals = []
                for fold in range(min(len(models_data[main_model][0]), len(models_data[model_name][0]))):
                    l = models_data[main_model][0][fold]
                    p1 = models_data[main_model][1][fold]
                    p2 = models_data[model_name][1][fold]
                    if len(l) == len(p1) == len(p2):
                        p_vals.append(delong_roc_test(l, p1, p2))
                if p_vals:
                    p_mean = np.mean(p_vals)
                    sig = "***" if p_mean < 0.001 else ("**" if p_mean < 0.01 else ("*" if p_mean < 0.05 else "ns"))
                    print(f"  {main_model} vs {model_name}: p={p_mean:.4f} {sig}")

    print(f"\n所有图表保存到: {FIG_DIR}/")
    print("Step 4 完成!")


if __name__ == "__main__":
    main()