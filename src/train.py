"""Train semantic segmentation model."""

import argparse
import glob
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, Optional

# Add parent directory to path for imports
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)


import traceback

import lightning.pytorch as pl
import oyaml as yaml
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger

from callbacks import (
    ConfigCallback,
    PostprocessorrCallback,
    VisualizerCallback,
    get_postprocessors,
    get_visualizers,
)
from datasets import get_data_module
from modules import get_backbone, get_criterion, module


def get_git_commit_hash() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def parse_args() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export_dir",
        required=True,
        help=("Path to export dir which saves logs, metrics, etc."),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to configuration file (*.yaml)",
    )
    parser.add_argument(
        "--ckpt_path",
        required=False,
        default=None,
        help="Provide *.ckpt file to continue training.",
    )
    parser.add_argument(
        "--resume",
        required=False,
        action="store_true",
    )  # implies default = False

    args = vars(parser.parse_args())

    return args


def load_config(path_to_config_file: str) -> Dict:
    assert os.path.exists(path_to_config_file)

    with open(path_to_config_file) as istream:
        config = yaml.safe_load(istream)

    return config


PRETRAINED_URLS = {
    "semantic-seg-erfnet.ckpt": (
        "https://www.ipb.uni-bonn.de/html/projects/phenobench/"
        "semantic_segmentation/semantic-seg-erfnet.ckpt"
    ),
    "semantic-seg-deeplab.ckpt": (
        "https://www.ipb.uni-bonn.de/html/projects/phenobench/"
        "semantic_segmentation/semantic-seg-deeplab.ckpt"
    ),
}


def ensure_checkpoint(ckpt_path: str) -> None:
    """Download pretrained checkpoint if it doesn't exist."""
    if ckpt_path is None or os.path.exists(ckpt_path):
        return

    filename = Path(ckpt_path).name
    url = PRETRAINED_URLS.get(filename)
    if url is None:
        return  # not a known pretrained file, let it fail later

    print(f"Downloading pretrained weights: {filename}")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    urllib.request.urlretrieve(url, ckpt_path)
    print(f"Saved to {ckpt_path}")


def find_last_checkpoint(export_dir: str) -> Optional[str]:
    """Find the most recent last.ckpt in export directory."""
    pattern = os.path.join(export_dir, "**", "last.ckpt")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        return None
    latest = max(matches, key=os.path.getmtime)
    return str(latest)


def find_wandb_run_id(export_dir: str) -> Optional[str]:
    """Find the most recent WandB run ID from export dir."""
    wandb_dir = os.path.join(export_dir, "wandb")
    if not os.path.isdir(wandb_dir):
        return None
    run_dirs = glob.glob(os.path.join(wandb_dir, "run-*"))
    if not run_dirs:
        return None
    latest = max(run_dirs, key=os.path.getmtime)
    # run dir format: run-YYYYMMDD_HHMMSS-<run_id>
    run_id = os.path.basename(str(latest)).split("-")[-1]
    return run_id


def main():
    args = parse_args()

    cfg = load_config(args["config"])
    cfg["git-commit"] = get_git_commit_hash()

    if cfg.get("seed") is None:
        seed_val = int(time.time())
        cfg["seed"] = seed_val
    else:
        seed_val = cfg["seed"]
    pl.seed_everything(seed_val)

    # Auto-detect last checkpoint when --resume without --ckpt_path
    if args["resume"] and args["ckpt_path"] is None:
        last_ckpt = find_last_checkpoint(args["export_dir"])
        if last_ckpt is not None:
            args["ckpt_path"] = last_ckpt
            print(f"Auto-detected checkpoint: {last_ckpt}")
        else:
            print(
                "Warning: --resume specified but no "
                "last.ckpt found. Training from scratch."
            )
            args["resume"] = False

    ensure_checkpoint(args["ckpt_path"])

    datasetmodule = get_data_module(cfg)
    criterion = get_criterion(cfg)

    # define backbone
    network = get_backbone(cfg)

    # Get optimizer kwargs from config (optional)
    optimizer_kwargs = cfg["train"].get("optimizer_kwargs", None)

    if args["ckpt_path"] is not None and not args["resume"]:
        seg_module = module.SegmentationNetwork(
            network,
            criterion,
            cfg["train"]["learning_rate"],
            cfg["train"]["weight_decay"],
            train_step_settings=(cfg["train"]["step_settings"]),
            val_step_settings=(cfg["val"]["step_settings"]),
            ckpt_path=args["ckpt_path"],
            optimizer_kwargs=optimizer_kwargs,
        )
    else:
        seg_module = module.SegmentationNetwork(
            network,
            criterion,
            cfg["train"]["learning_rate"],
            cfg["train"]["weight_decay"],
            train_step_settings=(cfg["train"]["step_settings"]),
            val_step_settings=(cfg["val"]["step_settings"]),
            optimizer_kwargs=optimizer_kwargs,
        )

    # Setup run directory: runs/<experiment_id>/
    run_dir = os.path.join(args["export_dir"], cfg["experiment"]["id"])
    ckpt_dir = os.path.join(run_dir, "checkpoints")

    # Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_best_val_mIoU = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val_mIoU",
        filename="epoch={epoch:02d}_val_mIoU={val_mIoU:.4f}",
        auto_insert_metric_name=False,
        mode="max",
        save_top_k=1,
        save_last=False,
    )

    visualizer_callback = VisualizerCallback(
        get_visualizers(cfg),
        cfg["train"]["vis_train_every_x_epochs"],
        cfg["val"]["vis_val_every_x_epochs"],
    )
    postprocessor_callback = PostprocessorrCallback(
        get_postprocessors(cfg),
        cfg["train"]["postprocess_train_every_x_epochs"],
        cfg["val"]["postprocess_val_every_x_epochs"],
    )
    config_callback = ConfigCallback(cfg)

    # Early stopping configuration (with defaults)
    early_stopping_cfg = cfg["train"].get("early_stopping", {})
    early_stopping = EarlyStopping(
        monitor=early_stopping_cfg.get("monitor", "val_mIoU"),
        mode=early_stopping_cfg.get("mode", "max"),
        patience=early_stopping_cfg.get("patience", 10),
        min_delta=early_stopping_cfg.get("min_delta", 0.0),
    )

    # Setup logger
    wandb_kwargs = {}
    if args["resume"]:
        run_id = find_wandb_run_id(run_dir)
        if run_id is not None:
            wandb_kwargs["id"] = run_id
            wandb_kwargs["resume"] = "must"
            print(f"Resuming WandB run: {run_id}")

    wandb_logger = WandbLogger(
        project="sugarbeet-weed-segmentation",
        name=cfg["experiment"]["id"],
        version="",
        config=cfg,
        save_dir=run_dir,
        **wandb_kwargs,
    )

    # Setup trainer
    trainer = Trainer(
        benchmark=cfg["train"]["benchmark"],
        accelerator="gpu",
        devices=cfg["train"]["n_gpus"],
        default_root_dir=run_dir,
        logger=wandb_logger,
        max_epochs=cfg["train"]["max_epoch"],
        check_val_every_n_epoch=(cfg["val"]["check_val_every_n_epoch"]),
        callbacks=[
            checkpoint_best_val_mIoU,
            lr_monitor,
            early_stopping,
            visualizer_callback,
            postprocessor_callback,
            config_callback,
        ],
    )

    try:
        if args["ckpt_path"] is None:
            print("Train from scratch.")
            trainer.fit(seg_module, datasetmodule)
        elif args["ckpt_path"] is not None and not args["resume"]:
            print(
                "Load pretrained model weights but "
                "other params (e.g. learning rate) "
                "start from scratch."
            )
            trainer.fit(seg_module, datasetmodule)
        elif args["ckpt_path"] is not None and args["resume"]:
            print("Load pretrained model weights and resume training.")
            trainer.fit(
                seg_module,
                datasetmodule,
                ckpt_path=args["ckpt_path"],
            )
        else:
            raise RuntimeError(
                "Can't train any model since the settings are invalid."
            )
    except Exception:
        wandb_logger.experiment.alert(
            title="Training crashed",
            text=(
                f"Experiment: {cfg['experiment']['id']}\n"
                f"Epoch: {trainer.current_epoch}\n"
                f"{traceback.format_exc()}"
            ),
            level="ERROR",
        )
        raise

    # Log training completion status
    best_score = checkpoint_best_val_mIoU.best_model_score
    best_path = checkpoint_best_val_mIoU.best_model_path
    best_epoch = (
        int(os.path.basename(best_path).split("epoch=")[1].split("_")[0])
        if best_path
        else -1
    )
    if early_stopping.stopped_epoch > 0:
        print(
            f"\nTraining stopped early by EarlyStopping "
            f"callback at epoch "
            f"{early_stopping.stopped_epoch}"
        )
    else:
        print(
            f"\nTraining completed successfully for "
            f"{trainer.current_epoch + 1} epochs"
        )
    print(f"Best val_mIoU: {best_score:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
