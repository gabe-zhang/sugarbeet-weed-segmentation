"""PhenoBench dataset for sugar beet vs weed segmentation.

Provides a PyTorch Dataset and Lightning DataModule for loading
PhenoBench images and semantic annotations (soil / crop / weed).
"""

import os
import random
from typing import Callable, Dict, List, Optional, Tuple

import lightning.pytorch as pl
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import datasets.common as common
from datasets.augmentations_color import get_color_augmentations
from datasets.augmentations_geometry import (
    CopyPasteWeed,
    GeometricDataAugmentation,
    get_copy_paste_augmentation,
    get_geometric_augmentations,
)
from datasets.image_normalizer import ImageNormalizer, get_image_normalizer


class PhenoBench(Dataset):
    """PhenoBench semantic segmentation dataset.

    The directory structure is as following:
    ├── test
    │   ├── images
    │   ├── leaf_instances
    │   ├── leaf_visibility
    │   ├── plant_instances
    │   ├── plant_visibility
    │   └── semantics
    ├── train
    │   ├── images
    │   ├── leaf_instances
    │   ├── leaf_visibility
    │   ├── plant_instances
    │   ├── plant_visibility
    │   └── semantics
    └── val
        ├── images
        ├── leaf_instances
        ├── leaf_visibility
        ├── plant_instances
        ├── plant_visibility
        ├── semantics
        └── visualize
    """

    def __init__(
        self,
        path_to_dataset: str,
        mode: str,
        img_normalizer: ImageNormalizer,
        augmentations_geometric: List[GeometricDataAugmentation],
        augmentations_color: List[Callable],
        copy_paste: Optional[CopyPasteWeed] = None,
    ):
        """Get the path to all images and its corresponding annotations.

        Args:
            path_to_dataset: Path to dataset directory.
            mode: train, val, or predict.
            img_normalizer: Image normalizer.
            augmentations_geometric: Geometric augmentations
                applied to image and annotation.
            augmentations_color: Color augmentations
                applied to the image.
        """

        assert os.path.exists(path_to_dataset), (
            f"The path to the dataset does not exist: {path_to_dataset}."
        )

        super().__init__()

        assert mode in ["train", "val", "predict"]
        self.mode = mode

        self.img_normalizer = img_normalizer
        self.augmentations_geometric = augmentations_geometric
        self.augmentations_color = augmentations_color
        self.copy_paste = copy_paste

        # ------------- Prepare Training -------------
        self.path_to_train_images = os.path.join(
            path_to_dataset, "train", "images"
        )
        self.path_to_train_annos = os.path.join(
            path_to_dataset, "train", "semantics"
        )
        self.filenames_train = common.get_img_fnames_in_dir(
            self.path_to_train_images
        )

        # ------------- Prepare Training -------------
        self.path_to_val_images = os.path.join(
            path_to_dataset, "val", "images"
        )
        self.path_to_val_annos = os.path.join(
            path_to_dataset, "val", "semantics"
        )
        self.filenames_val = common.get_img_fnames_in_dir(
            self.path_to_val_images
        )

        # ------------- Prepare Prediction -------------
        self.path_to_predict_images = os.path.join(
            path_to_dataset, "test", "images"
        )
        self.filenames_predict = common.get_img_fnames_in_dir(
            self.path_to_predict_images
        )

        # specify image transformations
        self.img_to_tensor = transforms.ToTensor()

    def get_train_item(self, idx: int) -> Dict:
        path_to_current_img = os.path.join(
            self.path_to_train_images, self.filenames_train[idx]
        )
        img_pil = Image.open(path_to_current_img)
        img = self.img_to_tensor(img_pil)  # [C x H x W] with values in [0, 1]

        if random.random() > 0.25:
            for augmentor_color_fn in self.augmentations_color:
                img = augmentor_color_fn(img)

        path_to_current_anno = os.path.join(
            self.path_to_train_annos, self.filenames_train[idx]
        )
        anno = np.array(Image.open(path_to_current_anno))  # dtype: int32
        if len(anno.shape) > 2:
            anno = anno[:, :, 0]
        anno = anno.astype(np.int64)
        anno = torch.Tensor(anno).type(torch.int64)  # [H x W]
        anno = anno.unsqueeze(0)  # [1 x H x W]

        for augmentor_geometric in self.augmentations_geometric:
            img, anno = augmentor_geometric(img, anno)
        anno = anno.squeeze(0)  # [H x W]

        mask_3 = anno == 3
        anno[mask_3] = 1

        mask_4 = anno == 4
        anno[mask_4] = 2

        if self.copy_paste is not None:
            donor_idx = random.randint(0, len(self.filenames_train) - 1)
            d_img, d_anno = self._load_augmented(donor_idx)
            img, anno = self.copy_paste(img, anno, d_img, d_anno)

        img_before_norm = img.clone()
        img = self.img_normalizer.normalize(img)

        return {
            "input_image_before_norm": img_before_norm,
            "input_image": img,
            "anno": anno,
            "fname": self.filenames_train[idx],
        }

    def _load_augmented(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a training sample with augmentations applied.

        Used to obtain a donor image for copy-paste.

        Args:
            idx: index into filenames_train

        Returns:
            Tuple of (image [C x H x W], anno [H x W])
        """
        path_img = os.path.join(
            self.path_to_train_images,
            self.filenames_train[idx],
        )
        img = self.img_to_tensor(Image.open(path_img))

        if random.random() > 0.25:
            for aug in self.augmentations_color:
                img = aug(img)

        path_anno = os.path.join(
            self.path_to_train_annos,
            self.filenames_train[idx],
        )
        anno = np.array(Image.open(path_anno))
        if len(anno.shape) > 2:
            anno = anno[:, :, 0]
        anno = torch.tensor(
            anno.astype(np.int64), dtype=torch.int64
        ).unsqueeze(0)

        for aug in self.augmentations_geometric:
            img, anno = aug(img, anno)
        anno = anno.squeeze(0)

        anno[anno == 3] = 1
        anno[anno == 4] = 2

        return img, anno

    def _get_eval_item(
        self,
        idx: int,
        path_to_images: str,
        path_to_annos: str,
        filenames: List[str],
    ) -> Dict:
        """Load a val sample with preprocessing."""
        path_img = os.path.join(path_to_images, filenames[idx])
        img = self.img_to_tensor(Image.open(path_img))

        path_anno = os.path.join(path_to_annos, filenames[idx])
        anno = np.array(Image.open(path_anno))
        if len(anno.shape) > 2:
            anno = anno[:, :, 0]
        anno = anno.astype(np.int64)
        anno = torch.Tensor(anno).type(torch.int64)
        anno = anno.unsqueeze(0)  # [1 x H x W]

        for aug in self.augmentations_geometric:
            img, anno = aug(img, anno)
        anno = anno.squeeze(0)  # [H x W]

        mask_3 = anno == 3
        anno[mask_3] = 1

        mask_4 = anno == 4
        anno[mask_4] = 2

        img_before_norm = img.clone()
        img = self.img_normalizer.normalize(img)

        return {
            "input_image_before_norm": img_before_norm,
            "input_image": img,
            "anno": anno,
            "fname": filenames[idx],
        }

    def _get_predict_item(self, idx: int) -> Dict:
        """Load a predict sample (no annotation)."""
        path_img = os.path.join(
            self.path_to_predict_images,
            self.filenames_predict[idx],
        )
        img = self.img_to_tensor(Image.open(path_img))

        dummy_anno = torch.zeros(
            1,
            img.shape[1],
            img.shape[2],
            dtype=torch.int64,
        )
        for aug in self.augmentations_geometric:
            img, dummy_anno = aug(img, dummy_anno)

        img_before_norm = img.clone()
        img = self.img_normalizer.normalize(img)

        return {
            "input_image_before_norm": img_before_norm,
            "input_image": img,
            "fname": self.filenames_predict[idx],
        }

    def __getitem__(self, idx: int):
        if self.mode == "train":
            return self.get_train_item(idx)

        if self.mode == "val":
            return self._get_eval_item(
                idx,
                self.path_to_val_images,
                self.path_to_val_annos,
                self.filenames_val,
            )

        if self.mode == "predict":
            return self._get_predict_item(idx)

    def __len__(self):
        if self.mode == "train":
            return len(self.filenames_train)

        if self.mode == "val":
            return len(self.filenames_val)

        if self.mode == "predict":
            return len(self.filenames_predict)


class PhenoBenchModule(pl.LightningDataModule):
    """Lightning DataModule for the PhenoBench dataset."""

    def __init__(self, cfg: Dict):
        super().__init__()

        self.cfg = cfg

    def setup(self, stage: Optional[str] = None):
        """Data operations we perform on every GPU.

        Here we define the how to split the dataset.

        Args:
            stage (Optional[str], optional): _description_. Defaults to None.
        """
        path_to_dataset = self.cfg["data"]["path_to_dataset"]
        image_normalizer = get_image_normalizer(self.cfg)

        if (stage == "fit") or (stage == "validate") or (stage is None):
            # ----------- TRAIN -----------
            train_augmentations_geometric = get_geometric_augmentations(
                self.cfg, "train"
            )
            train_augmentations_color = get_color_augmentations(
                self.cfg, "train"
            )
            train_copy_paste = get_copy_paste_augmentation(self.cfg, "train")
            self.train_ds = PhenoBench(
                path_to_dataset,
                mode="train",
                img_normalizer=image_normalizer,
                augmentations_geometric=train_augmentations_geometric,
                augmentations_color=train_augmentations_color,
                copy_paste=train_copy_paste,
            )

            # ----------- VAL -----------
            val_augmentations_geometric = get_geometric_augmentations(
                self.cfg, "val"
            )
            self.val_ds = PhenoBench(
                path_to_dataset,
                mode="val",
                img_normalizer=image_normalizer,
                augmentations_geometric=val_augmentations_geometric,
                augmentations_color=[],
            )

        if stage == "predict" or stage is None:
            # ----------- PREDICT -----------
            predict_augs_geometric = get_geometric_augmentations(
                self.cfg, "predict"
            )
            self.predict_ds = PhenoBench(
                path_to_dataset,
                mode="predict",
                img_normalizer=image_normalizer,
                augmentations_geometric=predict_augs_geometric,
                augmentations_color=[],
            )

    def train_dataloader(self) -> DataLoader:
        # Return DataLoader for Training Data here
        shuffle: bool = self.cfg["train"]["shuffle"]
        batch_size: int = self.cfg["train"]["batch_size"]
        n_workers: int = self.cfg["data"]["num_workers"]

        loader = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=n_workers,
            drop_last=True,
            pin_memory=True,
        )

        return loader

    def val_dataloader(self) -> DataLoader:
        batch_size: int = self.cfg["val"]["batch_size"]
        n_workers: int = self.cfg["data"]["num_workers"]

        loader = DataLoader(
            self.val_ds,
            batch_size=batch_size,
            num_workers=n_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
        )

        return loader

    def predict_dataloader(self) -> DataLoader:
        batch_size: int = self.cfg["predict"]["batch_size"]
        n_workers: int = self.cfg["data"]["num_workers"]

        loader = DataLoader(
            self.predict_ds,
            batch_size=batch_size,
            num_workers=n_workers,
            shuffle=False,
            pin_memory=True,
        )

        return loader
