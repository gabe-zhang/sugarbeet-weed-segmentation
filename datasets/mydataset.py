import os
from typing import Callable, Dict, List, Optional

import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import datasets.common as common
from datasets.augmentations_geometry import (
    GeometricDataAugmentation,
    get_geometric_augmentations,
)
from datasets.image_normalizer import ImageNormalizer, get_image_normalizer


class MyDataset(Dataset):
    """Represents the PDC dataset.

    The directory structure is as following:
    └── predict
        └── images
    """

    def __init__(
        self,
        path_to_dataset: str,
        mode: str,
        img_normalizer: ImageNormalizer,
        augmentations_geometric: List[GeometricDataAugmentation],
        augmentations_color: List[Callable],
    ):
        """Get the path to all images and its corresponding
        annotations.

        Args:
            path_to_dataset (str):
                Path to dir that contains the images and annotations
            mode(str): train, val, or test
            img_normalizer (ImageNormalizer):
                Specifies how to normalize the input images
            augmentations_geometric (List[GeometricDataAugmentation]):
                Geometric data augmentations applied to
                the image and its annotations
            augmentations_color (List[Callable]):
                Color data augmentations applied to the image
        """

        assert os.path.exists(path_to_dataset), (
            f"The path to the dataset does not exist: {path_to_dataset}."
        )

        super().__init__()

        assert mode in ["predict"]
        self.mode = mode

        self.img_normalizer = img_normalizer
        self.augmentations_geometric = augmentations_geometric
        self.augmentations_color = augmentations_color

        # ------------- Prepare Predicting -------------
        self.path_to_predict_images = os.path.join(
            path_to_dataset, "predict", "images"
        )
        self.filenames_predict = common.get_img_fnames_in_dir(
            self.path_to_predict_images
        )

        # specify image transformations
        self.img_to_tensor = transforms.ToTensor()

    def get_predict_item(self, idx: int) -> Dict:
        path_to_current_img = os.path.join(
            self.path_to_predict_images, self.filenames_predict[idx]
        )
        img_pil = Image.open(path_to_current_img)
        img = self.img_to_tensor(img_pil)  # [C x H x W] with values in [0, 1]

        for augmentor_geometric in self.augmentations_geometric:
            img = augmentor_geometric(img)

        img_before_norm = img.clone()
        img = self.img_normalizer.normalize(img)

        # return img
        return {
            "input_image_before_norm": img_before_norm,
            "input_image": img,
            "fname": self.filenames_predict[idx],
        }

    def __getitem__(self, idx: int):
        if self.mode == "predict":
            items = self.get_predict_item(idx)
            return items

    def __len__(self):
        if self.mode == "predict":
            return len(self.filenames_predict)


class MyDatasetModule(pl.LightningDataModule):
    """Encapsulates all the steps needed to
    process data for prediction."""

    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg

    def __len__(self):
        return len(self.predict_ds)

    def setup(self, stage: Optional[str] = None):
        """Data operations we perform on every GPU.

        Here we define the how to split the dataset.

        Args:
            stage (Optional[str], optional):
                either 'fit', 'validate', 'test', or 'predict'
        """
        path_to_dataset = self.cfg["data"]["path_to_dataset"]
        image_normalizer = get_image_normalizer(self.cfg)

        if stage == "predict" or stage is None:
            predict_augmentations_geometric = get_geometric_augmentations(
                self.cfg, "predict"
            )
            self.predict_ds = MyDataset(
                path_to_dataset,
                mode="predict",
                img_normalizer=image_normalizer,
                augmentations_geometric=predict_augmentations_geometric,
                augmentations_color=[],
            )

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
