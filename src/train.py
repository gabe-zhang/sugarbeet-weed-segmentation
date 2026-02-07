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


def parse_args() -> Dict[str, str]:
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
    return latest


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

    if args["ckpt_path"] is not None and not args["resume"]:
        seg_module = module.SegmentationNetwork(
            network,
            criterion,
            cfg["train"]["learning_rate"],
            cfg["train"]["weight_decay"],
            train_step_settings=(cfg["train"]["step_settings"]),
            val_step_settings=(cfg["val"]["step_settings"]),
            ckpt_path=args["ckpt_path"],
        )
    else:
        seg_module = module.SegmentationNetwork(
            network,
            criterion,
            cfg["train"]["learning_rate"],
            cfg["train"]["weight_decay"],
            train_step_settings=(cfg["train"]["step_settings"]),
            val_step_settings=(cfg["val"]["step_settings"]),
        )

    # Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    checkpoint_saver_val_loss = ModelCheckpoint(
        monitor="val_loss",
        filename=(cfg["experiment"]["id"] + "_{epoch:02d}_{val_loss:.4f}"),
        mode="min",
        save_last=True,
    )
    checkpoint_saver_val_mIoU = ModelCheckpoint(
        monitor="val_mIoU",
        filename=(cfg["experiment"]["id"] + "_{epoch:02d}_{val_mIoU:.4f}"),
        mode="max",
        save_last=False,
    )
    checkpoint_saver_train_loss = ModelCheckpoint(
        monitor="train_loss",
        filename=(cfg["experiment"]["id"] + "_{epoch:02d}_{train_loss:.4f}"),
        mode="min",
        save_last=False,
    )
    checkpoint_saver_train_mIoU = ModelCheckpoint(
        monitor="train_mIoU",
        filename=(cfg["experiment"]["id"] + "_{epoch:02d}_{train_mIoU:.4f}"),
        mode="max",
        save_last=False,
    )

    my_checkpoint_savers = [
        var_value
        for var_name, var_value in locals().items()
        if var_name.startswith("checkpoint_saver")
    ]

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
    early_stopping = EarlyStopping(
        monitor="val_mIoU",
        mode="max",
        patience=10,
    )

    # Setup logger
    wandb_logger = WandbLogger(
        project="sugarbeet-weed-segmentation",
        name=cfg["experiment"]["id"],
        config=cfg,
        save_dir=args["export_dir"],
    )

    # Setup trainer
    trainer = Trainer(
        benchmark=cfg["train"]["benchmark"],
        accelerator="gpu",
        devices=cfg["train"]["n_gpus"],
        default_root_dir=args["export_dir"],
        logger=wandb_logger,
        max_epochs=cfg["train"]["max_epoch"],
        check_val_every_n_epoch=(cfg["val"]["check_val_every_n_epoch"]),
        callbacks=[
            *my_checkpoint_savers,
            lr_monitor,
            early_stopping,
            visualizer_callback,
            postprocessor_callback,
            config_callback,
        ],
    )

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


if __name__ == "__main__":
    main()
