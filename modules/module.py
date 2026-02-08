import os
from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
import oyaml as yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics


class SegmentationNetwork(pl.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        criterion: nn.Module,
        learning_rate: float,
        weight_decay: float,
        train_step_settings: Optional[List[str]] = None,
        val_step_settings: Optional[List[str]] = None,
        test_step_settings: Optional[List[str]] = None,
        predict_step_settings: Optional[List[str]] = None,
        ckpt_path=None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.network = network
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_kwargs = optimizer_kwargs or {}

        self.save_hyperparameters(
            "learning_rate", "weight_decay", "optimizer_kwargs"
        )

        # evaluation metrics for all classes
        self.metric_train_iou = torchmetrics.JaccardIndex(
            task="multiclass",
            num_classes=self.network.num_classes,
            average="none",
        )
        self.metric_val_iou = torchmetrics.JaccardIndex(
            task="multiclass",
            num_classes=self.network.num_classes,
            average="none",
        )
        self.metric_test_iou = torchmetrics.JaccardIndex(
            task="multiclass",
            num_classes=self.network.num_classes,
            average="none",
        )

        if train_step_settings is not None:
            self.train_step_settings = train_step_settings
        else:
            self.train_step_settings = []

        if val_step_settings is not None:
            self.val_step_settings = val_step_settings
        else:
            self.val_step_settings = []

        if test_step_settings is not None:
            self.test_step_settings = test_step_settings
        else:
            self.test_step_settings = []

        if predict_step_settings is not None:
            self.predict_step_settings = predict_step_settings
        else:
            self.predict_step_settings = []

        if ckpt_path is not None:
            print("Load pretrained weights of backbone.")
            ckpt_dict = torch.load(ckpt_path, weights_only=False)
            self.load_state_dict(ckpt_dict["state_dict"], strict=False)

    def compute_xentropy_loss(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        mode: str,
        mask_keep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute cross entropy loss.

        Args:
            logits: logits [B x num_classes x H x W]
            y: ground-truth [B x H x W]
            mode: train, val, or test
            mask_keep: 1 := consider, 0 := ignore
                [B x H x W]. Defaults to None.

        Returns:
            torch.Tensor: loss
        """
        return self.criterion(logits, y, mode=mode, mask_keep=mask_keep)

    def forward(self, img_batch: torch.Tensor) -> torch.Tensor:
        """Forward pass of backbone network.

        Args:
            img_batch: input image(s)
                [B x C x H x W]

        Returns:
            torch.Tensor: predictions
                [B x num_classes x H x W]
        """
        output = self.network.forward(img_batch)

        return output

    def training_step(self, batch: dict, batch_idx: int) -> Dict[str, Any]:
        if not self.train_step_settings:
            raise ValueError(
                "You need to specify the settings for the training step."
            )

        processed_steps = []
        if "regular" in self.train_step_settings:
            logits_regular = self.forward(batch["input_image"])

            mask_keep_regular = batch["anno"] != 255
            loss_regular = self.compute_xentropy_loss(
                logits_regular,
                batch["anno"],
                mode="train",
                mask_keep=mask_keep_regular,
            )

            processed_steps.append("regular")
            # update training metrics
            pred = torch.argmax(logits_regular.detach(), dim=1)
            self.metric_train_iou(pred, batch["anno"])

        if self.trainer.current_epoch == 0 and batch_idx == 0:
            print(
                "Processed steps during training: ",
                processed_steps,
            )

        # accumulate all losses
        my_local_loss_values = {
            var_name: var_value
            for var_name, var_value in locals().items()
            if var_name.startswith("loss")
        }
        loss = torch.sum(torch.stack(list(my_local_loss_values.values())))

        # log losses for epoch-level aggregation
        bs = batch["input_image"].shape[0]
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
            batch_size=bs,
        )
        for k, v in my_local_loss_values.items():
            self.log(
                f"train_{k}",
                v,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
                batch_size=bs,
            )

        out_dict = {
            "loss": loss,
            "logits": logits_regular,
            "anno": batch["anno"],
        }
        out_dict.update(my_local_loss_values)
        del my_local_loss_values

        return out_dict

    def on_train_epoch_end(self) -> None:
        # compute final metrics over all batches
        iou_per_class = self.metric_train_iou.compute().detach()
        mIoU = iou_per_class.mean()
        self.metric_train_iou.reset()

        for class_index, iou_class in enumerate(iou_per_class):
            self.log(
                f"train_iou_class_{class_index}",
                iou_class,
                on_epoch=True,
                sync_dist=False,
            )
        self.log(
            "train_mIoU",
            mIoU,
            on_epoch=True,
            sync_dist=False,
        )

    def validation_step(self, batch: dict, batch_idx: int) -> Dict[str, Any]:
        if not self.val_step_settings:
            raise ValueError(
                "You need to specify the settings for the validation step."
            )
        assert len(self.val_step_settings) == 1

        # predictions
        if "regular" in self.val_step_settings:
            logits = self.forward(batch["input_image"])

        # objective
        loss = self.compute_xentropy_loss(logits, batch["anno"], mode="val")

        # update validation metrics
        self.metric_val_iou(
            torch.argmax(logits, dim=1),
            batch["anno"],
        )

        # log loss for epoch-level aggregation
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
            batch_size=batch["input_image"].shape[0],
        )

        return {
            "loss": loss,
            "logits": logits,
            "anno": batch["anno"],
        }

    def on_validation_epoch_end(self) -> None:
        epoch = self.trainer.current_epoch

        # compute final metrics over all batches
        iou_per_class = self.metric_val_iou.compute()
        mIoU = iou_per_class.mean()
        self.metric_val_iou.reset()

        for class_index, iou_class in enumerate(iou_per_class):
            self.log(
                f"val_iou_class_{class_index}",
                iou_class,
                on_epoch=True,
                sync_dist=False,
            )
        self.log(
            "val_mIoU",
            mIoU,
            on_epoch=True,
            sync_dist=False,
        )

        path_to_classwise_iou_dir = os.path.join(
            self.trainer.log_dir,
            "val",
            f"epoch-{epoch:06d}",
        )
        save_iou_metric(iou_per_class, path_to_classwise_iou_dir)

    def test_step(self, batch: dict, batch_idx: int) -> Dict[str, Any]:
        if not self.test_step_settings:
            raise ValueError(
                "You need to specify the settings for the test step."
            )
        assert len(self.test_step_settings) == 1

        # predictions
        if "regular" in self.test_step_settings:
            logits = self.forward(batch["input_image"])

        pred_cls = torch.argmax(logits, dim=1)

        # regular evaluation for all classes
        self.metric_test_iou(pred_cls, batch["anno"])

        return {
            "logits": logits,
            "anno": batch["anno"],
        }

    def on_test_epoch_end(self) -> None:
        # compute final metrics over all batches
        iou_per_class = self.metric_test_iou.compute()
        mIoU = round(float(iou_per_class.mean().cpu()), 3)
        print(f"Test mIoU: {mIoU}")
        self.metric_test_iou.reset()

        epoch = self.trainer.current_epoch

        path_to_classwise_iou_dir = os.path.join(
            self.trainer.log_dir,
            "test",
            f"epoch-{epoch:06d}",
        )
        save_iou_metric(iou_per_class, path_to_classwise_iou_dir)

    def predict_step(self, batch: dict, batch_idx: int) -> Dict[str, Any]:
        if not self.predict_step_settings:
            raise ValueError(
                "You need to specify the settings for the predict step."
            )
        assert len(self.predict_step_settings) == 1

        # predictions
        if "regular" in self.predict_step_settings:
            logits = self.forward(batch["input_image"])

        return {"logits": logits}

    def configure_optimizers(self):
        # Merge default Adam kwargs with user-provided ones
        adam_defaults = {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
            "eps": 1e-8,  # PyTorch default
            "betas": (0.9, 0.999),  # PyTorch default
        }
        adam_kwargs = {**adam_defaults, **self.optimizer_kwargs}
        # Ensure lr and weight_decay from config take precedence
        adam_kwargs["lr"] = self.learning_rate
        adam_kwargs["weight_decay"] = self.weight_decay

        optimizer = optim.Adam(self.parameters(), **adam_kwargs)

        max_epochs = self.trainer.max_epochs

        def lr_scaling(epoch):
            return pow((1 - (epoch / max_epochs)), 3.0)

        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_scaling
        )

        return [optimizer], [scheduler]


def save_iou_metric(metrics: torch.Tensor, path_to_dir: str) -> None:
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)

    iou_info = {}
    for cls_index, iou_metric in enumerate(metrics):
        iou_info[f"class_{cls_index}"] = round(float(iou_metric), 5)

    iou_info["mIoU"] = round(float(metrics.mean()), 5)

    fpath = os.path.join(path_to_dir, "iou.yaml")
    with open(fpath, "w") as ostream:
        yaml.dump(iou_info, ostream)
