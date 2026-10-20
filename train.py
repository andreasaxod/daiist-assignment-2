"""
Assignment 2: training pipeline.

You're using a script for this stage. If you'd rather use a notebook,
write train.ipynb instead and delete this file. main.py refuses to run if
it finds both train.py and train.ipynb, so exactly one of them must exist.

Replace this docstring and everything below it with your own code. When run
via uv run python main.py train, this file must, start to finish, with no
manual steps in between:

- Get your Roboflow API key via from api_key import load_api_key, so that it
  works whether the key is in your .env file or in the environment, and
  download your dataset version from Roboflow Universe.
- Finetune yolo11n/yolo11s (or the -cls variant) on that dataset, from the
  pretrained checkpoint.
- Evaluate on the test split against the two baselines named in the brief: the
  pretrained COCO model zero-shot, and a trivial predict-nothing / majority
  class baseline.
- Sweep the confidence threshold, compute the expected cost at each point from
  the cost matrix you defined in REPORT.md, and save the sweep.
- Save everything app.py needs to build its dashboard without retraining:
  the metrics, the per-class numbers, the cost sweep, and the chosen operating
  point.

This is your real training pipeline. It takes as long as it takes, and it runs
wherever training is practical for you: your own machine, a GPU you have access
to, or a hosted notebook. Nothing re-runs it to check your submission.

The weights it produces are what you commit to weights/ and what your app
serves. Say in REPORT.md where you trained and what settings you used.

Save the trained model into weights/. That is the file you commit and the file
app.py loads.

How you structure the code beyond that is your call to make and be able to
defend: file layout, function boundaries, naming.
"""
import json
import os
import shutil
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

from dataset import (
    PROJECT,
    RESULTS_DIR,
    VERSION,
    WORKSPACE,
    class_names,
    describe_dataset,
    download_dataset,
    write_local_yaml,
)
from evaluate import evaluate_predictions, load_ground_truth, predict

ROOT = Path(__file__).resolve().parent
PRETRAINED_DIR = ROOT / "pretrained"
RUNS_DIR = ROOT / "runs"
WEIGHTS_DIR = ROOT / "weights"

BASE_MODEL = "yolo11n"
EPOCHS = int(os.environ.get("EPOCHS", "50"))
IMGSZ = 512
BATCH = 16
PATIENCE = 10
SEED = 0
REPORT_CONF = 0.25
COCO_IMGSZ = 640
COCO_PERSON = 0
ZERO_SHOT_TARGET = "swimmer"


def pretrained_path() -> str:
    PRETRAINED_DIR.mkdir(exist_ok=True)
    return str(PRETRAINED_DIR / f"{BASE_MODEL}.pt")


def finetune(data_yaml: Path) -> Path:
    model = YOLO(pretrained_path())
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        seed=SEED,
        project=str(RUNS_DIR),
        name="finetune",
        exist_ok=True,
    )
    WEIGHTS_DIR.mkdir(exist_ok=True)
    target = WEIGHTS_DIR / "best.pt"
    shutil.copy(Path(model.trainer.best), target)
    return target


def evaluate_all(dataset_dir: Path, names: list[str], weights: Path) -> dict:
    truth = load_ground_truth(dataset_dir, "test")
    images = list(truth)

    finetuned = YOLO(str(weights))
    finetuned_predictions = predict(finetuned, images, IMGSZ)

    coco = YOLO(pretrained_path())
    zero_shot_predictions = predict(
        coco,
        images,
        COCO_IMGSZ,
        classes=[COCO_PERSON],
        class_map={COCO_PERSON: names.index(ZERO_SHOT_TARGET)},
    )

    return {
        "finetuned": evaluate_predictions(finetuned_predictions, truth, names, REPORT_CONF),
        "coco_zero_shot": evaluate_predictions(zero_shot_predictions, truth, names, REPORT_CONF),
        "predict_nothing": evaluate_predictions({}, truth, names, REPORT_CONF),
    }


def show(results: dict) -> None:
    for model_name, result in results.items():
        print(f"\n{model_name}: mAP50 {result['map50']}  mAP50-95 {result['map50_95']}  (conf {result['conf_threshold']})")
        for class_name, row in result["per_class"].items():
            print(
                f"  {class_name:10s} n={row['instances']:4d}  AP50 {row['ap50']}  "
                f"P {row['precision']}  R {row['recall']}  TP {row['tp']}  FP {row['fp']}  FN {row['fn']}"
            )
    print("\nfinetuned confusion matrix (rows true, columns predicted)")
    labels = results["finetuned"]["confusion_labels"]
    print("            " + "  ".join(f"{label[:9]:>9s}" for label in labels))
    for label, row in zip(labels, results["finetuned"]["confusion_matrix"]):
        print(f"{label[:10]:>10s}  " + "  ".join(f"{value:9d}" for value in row))


def main() -> None:
    dataset_dir = download_dataset()
    describe_dataset(dataset_dir)
    names = class_names(dataset_dir)
    data_yaml = write_local_yaml(dataset_dir)

    weights = finetune(data_yaml)
    results = evaluate_all(dataset_dir, names, weights)

    payload = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION},
        "training": {
            "base_model": BASE_MODEL,
            "epochs_requested": EPOCHS,
            "imgsz": IMGSZ,
            "batch": BATCH,
            "patience": PATIENCE,
            "seed": SEED,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
        },
        "zero_shot_mapping": {"coco_class": "person", "dataset_class": ZERO_SHOT_TARGET},
        "split": "test",
        "models": results,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(payload, indent=2))
    show(results)


if __name__ == "__main__":
    main()
