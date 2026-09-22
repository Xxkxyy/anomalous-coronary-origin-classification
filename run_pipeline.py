"""自动化实验流水线 (5-fold CV + 多模型对比)

用法:
  python run_pipeline.py                        # 仅运行主模型 (resnet50)
  python run_pipeline.py --baselines            # 运行所有基线模型
  python run_pipeline.py --model densenet121    # 运行指定模型
  python run_pipeline.py --skip-preprocess      # 跳过预处理

步骤:
  Step 1: DICOM → NIfTI → 裁剪 → 归一化
  Step 2: 8:2 holdout + 5-fold CV 划分
  Step 3: 5-fold CV 训练 (主模型)
  Step 3b: 5-fold CV 训练 (基线模型，可选)
  Step 4: SCI 期刊级评估与多模型对比
"""
import sys, os, logging, time, argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

BASELINE_MODELS = ["densenet121", "single_slice", "resnet18_3d"]


def run_step(name, func):
    t0 = time.time()
    log.info(f"{'='*55}")
    log.info(f"START: {name}")
    log.info(f"{'='*55}")
    func()
    log.info(f"DONE: {name} ({time.time() - t0:.1f}s)\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baselines", action="store_true", help="Run all baseline models")
    parser.add_argument("--model", type=str, default="resnet50", help="Single model to train")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing")
    parser.add_argument("--skip-split", action="store_true", help="Skip split")
    args = parser.parse_args()

    log.info("Pipeline: 冠脉起源异常分类")
    if args.baselines:
        log.info("Mode: 全基线对比")

    # Step 1: 预处理
    if not args.skip_preprocess:
        from step1_preprocess import main as step1
        run_step("Step 1: DICOM → NIfTI → Crop → Normalize", step1)
    else:
        log.info("Step 1: SKIPPED")

    # Step 2: 划分
    if not args.skip_split:
        from step2_split import main as step2
        run_step("Step 2: 5-fold CV Split", step2)
    else:
        log.info("Step 2: SKIPPED")

    # Step 3: 主模型
    if args.baselines:
        models = [args.model] + BASELINE_MODELS
    else:
        models = [args.model]

    for model_name in models:
        log.info(f"Training model: {model_name}")
        import subprocess
        cmd = f'"{sys.executable}" scripts/step3_train.py --model {model_name}'
        t0 = time.time()
        log.info(f"START: Step 3 - {model_name}")
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        log.info(f"DONE: Step 3 - {model_name} ({time.time() - t0:.1f}s)")

    # Step 3b: Radiomics baselines
    if args.baselines:
        import subprocess
        for cls_name in ["svm", "rf"]:
            log.info(f"Training: radiomics + {cls_name}")
            cmd = f'"{sys.executable}" scripts/baseline_radiomics.py --classifier {cls_name}'
            t0 = time.time()
            log.info(f"START: Step 3b - radiomics_{cls_name}")
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
            log.info(f"DONE: Step 3b - radiomics_{cls_name} ({time.time() - t0:.1f}s)")

    # Step 4: 评估
    from step4_evaluate import main as step4
    run_step("Step 4: SCI-grade Evaluation", step4)

    log.info("Pipeline completed!")


if __name__ == '__main__':
    main()