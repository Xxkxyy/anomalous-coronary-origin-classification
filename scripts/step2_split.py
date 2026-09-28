"""Step 2: 5-fold 交叉验证划分

策略：
1. 先按 8:2 分层抽样，留出 20% 作为固定 holdout 测试集
2. 剩余 80% 做 5-fold 分层 CV：
   - 每折：4/5 训练 + 1/5 验证（早停用）
3. 输出两个 CSV：
   - data_split.csv：CV 划分（train/val/fold_id）
   - data_split_test.csv：holdout 测试集
"""
from __future__ import annotations

import csv
import os
import random
from collections import Counter


def collect_patients(nifti_norm_dir: str) -> list[tuple[str, int]]:
    patients = []
    for label_str in ['0', '1']:
        folder = os.path.join(nifti_norm_dir, label_str)
        if not os.path.isdir(folder):
            continue
        # 排序保证收集顺序确定性（os.listdir 顺序不保证）
        for fn in sorted(os.listdir(folder)):
            if fn.endswith('.nii.gz'):
                patients.append((fn.replace('.nii.gz', ''), int(label_str)))
    return patients


def assert_no_leak(splits: dict[str, set[str]]) -> None:
    """断言 train/val/test 三份病人 ID 集合两两无交集（数据泄漏检查）。

    splits: 形如 {'train': set, 'val': set, 'test': set}，值为患者 ID 集合。
    任何两两交集非空即抛出 AssertionError，并打印交集详情。
    """
    names = ['train', 'val', 'test']
    for name in names:
        if name not in splits:
            raise ValueError(f'assert_no_leak 需要提供 {name} 集合')
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            overlap = splits[n1] & splits[n2]
            if overlap:
                shown = ', '.join(sorted(overlap)[:20])
                if len(overlap) > 20:
                    shown += ', ...'
                raise AssertionError(
                    f'数据泄漏: {n1} 与 {n2} 存在 {len(overlap)} 例重叠患者: {shown}'
                )


def kfold_split(
    patients: list[tuple[str, int]],
    n_folds: int = 5,
    test_ratio: float = 0.2,
    seed: int = 42,
):
    """分层抽样：先 holdout 20% test，剩余做 5-fold CV"""
    random.seed(seed)

    neg = [p for p in patients if p[1] == 0]
    pos = [p for p in patients if p[1] == 1]
    random.shuffle(neg)
    random.shuffle(pos)

    # Holdout test set
    def split_holdout(group):
        n_test = max(1, round(len(group) * test_ratio))
        return group[n_test:], group[:n_test]

    neg_train, neg_test = split_holdout(neg)
    pos_train, pos_test = split_holdout(pos)

    # 5-fold CV
    random.shuffle(neg_train)
    random.shuffle(pos_train)

    def assign_folds(group, n_folds):
        n = len(group)
        fold_size = n // n_folds
        remainder = n % n_folds
        folds = []
        start = 0
        for i in range(n_folds):
            extra = 1 if i < remainder else 0
            end = start + fold_size + extra
            folds.append(group[start:end])
            start = end
        return folds

    neg_folds = assign_folds(neg_train, n_folds)
    pos_folds = assign_folds(pos_train, n_folds)

    # 构建 CV 划分
    cv_rows = []
    for fold_id in range(n_folds):
        # 验证集 = fold_id
        val_set = neg_folds[fold_id] + pos_folds[fold_id]
        # 训练集 = 其余 folds
        train_set = []
        for j in range(n_folds):
            if j != fold_id:
                train_set.extend(neg_folds[j] + pos_folds[j])

        for patient_id, label in train_set:
            cv_rows.append((patient_id, label, f'fold_{fold_id}', 'train'))
        for patient_id, label in val_set:
            cv_rows.append((patient_id, label, f'fold_{fold_id}', 'val'))

    # Holdout test
    test_rows = [(patient_id, label, 'holdout', 'test')
                 for patient_id, label in neg_test + pos_test]

    return cv_rows, test_rows


def print_stats(cv_rows, test_rows, n_folds=5):
    print(f"\n{'Set':<16} {'Total':>6} {'Neg':>6} {'Pos':>6} {'Pos%':>8}")
    print("-" * 50)

    for fold_id in range(n_folds):
        fld = f'fold_{fold_id}'
        train = [r for r in cv_rows if r[2] == fld and r[3] == 'train']
        val = [r for r in cv_rows if r[2] == fld and r[3] == 'val']
        for name, subset in [('  fold_{}_train'.format(fold_id), train),
                              ('  fold_{}_val'.format(fold_id), val)]:
            n = len(subset)
            neg = sum(1 for s in subset if s[1] == 0)
            pos = sum(1 for s in subset if s[1] == 1)
            print(f"{name:<16} {n:>6} {neg:>6} {pos:>6} {pos/n*100:>7.1f}%")

    # Test
    n = len(test_rows)
    neg = sum(1 for s in test_rows if s[1] == 0)
    pos = sum(1 for s in test_rows if s[1] == 1)
    print(f"{'holdout_test':<16} {n:>6} {neg:>6} {pos:>6} {pos/n*100:>7.1f}%")

    # Total
    all_rows = cv_rows + test_rows
    unique = set(r[0] for r in all_rows)
    print("-" * 50)
    print(f"{'Total (unique)':<16} {len(unique):>6}")


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nifti_norm_dir = os.path.join(project_root, 'nifti_norm')
    output_cv = os.path.join(project_root, 'data_split.csv')
    output_test = os.path.join(project_root, 'data_split_test.csv')

    patients = collect_patients(nifti_norm_dir)
    n_neg = sum(1 for _, l in patients if l == 0)
    n_pos = sum(1 for _, l in patients if l == 1)
    print(f'收集到 {len(patients)} 例: 阴性 {n_neg}, 阳性 {n_pos}')

    # 标签冲突检查：同一患者出现在不同标签目录时应提示（确定性收集下结果可复现）
    label_of = {}
    for pid, label in patients:
        if pid in label_of and label_of[pid] != label:
            print(f'WARNING: 患者 {pid} 出现在多个标签目录（{label_of[pid]} 与 {label}），存在标签冲突！')
        label_of[pid] = label

    cv_rows, test_rows = kfold_split(patients, n_folds=5, test_ratio=0.2, seed=42)

    # 写 CV 划分
    with open(output_cv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['patient_id', 'label', 'fold', 'split'])
        w.writerows(cv_rows)
    print(f'\nCV 划分 → {output_cv}')

    # 写 holdout 测试集
    with open(output_test, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['patient_id', 'label', 'fold', 'split'])
        w.writerows(test_rows)
    print(f'Holdout 测试集 → {output_test}')

    print_stats(cv_rows, test_rows)

    # 验证无泄漏
    all_ids = [r[0] for r in cv_rows] + [r[0] for r in test_rows]
    assert len(set(all_ids)) == len(patients), '存在重复/遗漏患者！'

    # 两两交集泄漏检查：每个 fold 内 train/val 互斥，且与该折无关的 holdout test 互斥
    test_ids = {r[0] for r in test_rows}
    for fold_id in range(5):
        fold_name = f'fold_{fold_id}'
        train_ids = {r[0] for r in cv_rows if r[2] == fold_name and r[3] == 'train'}
        val_ids = {r[0] for r in cv_rows if r[2] == fold_name and r[3] == 'val'}
        assert_no_leak({'train': train_ids, 'val': val_ids, 'test': test_ids})
    # 汇总检查：全体 CV 患者（各折 train+val）与 holdout test 互斥
    # 注：跨折的 train/val 重叠是 k-fold CV 的正常特性，不做交集检查
    assert_no_leak({'train': {r[0] for r in cv_rows}, 'val': set(), 'test': test_ids})
    print('\n验证通过: 无泄漏，无遗漏')


if __name__ == '__main__':
    main()