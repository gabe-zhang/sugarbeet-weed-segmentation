"""Train a YOLO instance segmentation model on PhenoBench.

Usage:
    uv run src/train_yolo.py --config config/yolo26x-baseline.yaml
"""

import argparse
import tempfile
from pathlib import Path

import wandb
import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO seg on PhenoBench"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="runs",
        help="Base directory for training outputs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    """Load training configuration from a YAML file."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def write_data_yaml(data_cfg: dict) -> str:
    """Write data section to a temp YAML for Ultralytics."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(data_cfg, tmp)
    tmp.close()
    return tmp.name


def log_gradient_stats(trainer) -> None:
    """Log mean per-parameter gradient norm to W&B."""
    if wandb.run is None:
        return
    total_sq = 0.0
    n_params = 0
    for param in trainer.model.parameters():
        if param.grad is not None:
            total_sq += param.grad.data.norm(2).item() ** 2
            n_params += param.numel()
    if n_params > 0:
        wandb.log(
            {"gradients/mean_norm": (total_sq / n_params) ** 0.5},
            commit=False,
        )


def log_per_class_metrics(trainer) -> None:
    """Log per-class seg metrics to W&B after validation."""
    if wandb.run is None:
        return
    validator = trainer.validator
    if validator is None:
        return
    names = validator.metrics.names
    # Box metrics
    if hasattr(validator.metrics, "box"):
        box = validator.metrics.box
        for i, name in names.items():
            wandb.log(
                {
                    f"val/box_AP50_{name}": float(box.ap50[i]),
                },
                commit=False,
            )
    # Mask metrics
    if hasattr(validator.metrics, "seg"):
        seg = validator.metrics.seg
        for i, name in names.items():
            wandb.log(
                {
                    f"val/mask_AP50_{name}": float(seg.ap50[i]),
                },
                commit=False,
            )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    wandb_cfg = cfg.get("wandb", {})
    wandb_project = wandb_cfg.get("project", "sugarbeet-weed-segmentation")
    train_cfg = cfg.get("train", {})
    run_name = cfg["experiment"].get("id")

    project = str(Path(args.export_dir).resolve())
    run_dir = Path(project) / run_name

    if args.resume:
        last_ckpt = run_dir / "weights" / "last.pt"
        if not last_ckpt.exists():
            raise FileNotFoundError(f"No checkpoint at {last_ckpt}")
        model = YOLO(str(last_ckpt))
        model.add_callback("on_fit_epoch_end", log_per_class_metrics)
        model.add_callback("on_train_batch_end", log_gradient_stats)
        model.train(resume=True)
        return

    wandb.init(project=wandb_project, name=run_name, config=cfg)

    data_yaml = write_data_yaml(cfg["data"])
    aug_cfg = cfg.get("augmentation", {})
    model = YOLO(cfg["model"])

    pretrained_weights = cfg.get("pretrained_weights")
    if pretrained_weights:
        model.load(pretrained_weights)

    model.add_callback("on_fit_epoch_end", log_per_class_metrics)
    model.add_callback("on_train_batch_end", log_gradient_stats)

    model.train(
        data=data_yaml,
        task="segment",
        epochs=train_cfg.get("epochs", 100),
        batch=train_cfg.get("batch", 4),
        imgsz=train_cfg.get("imgsz", 1024),
        device=train_cfg.get("device", "0"),
        workers=train_cfg.get("workers", 4),
        project=project,
        name=run_name,
        patience=train_cfg.get("patience", 20),
        optimizer=train_cfg.get("optimizer", "auto"),
        box=train_cfg.get("box", 7.5),
        cls=train_cfg.get("cls", 0.5),
        dfl=train_cfg.get("dfl", 1.5),
        save=True,
        exist_ok=True,
        pretrained=False,
        verbose=True,
        **aug_cfg,
    )


if __name__ == "__main__":
    main()
