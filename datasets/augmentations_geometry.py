"""Geometric data augmentations for semantic segmentation.

Applies identical transforms to both image and annotation.
"""

import math
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
from scipy.ndimage import label as ndimage_label


class GeometricDataAugmentation(ABC):
    """Transform applied to image and annotation."""

    @abstractmethod
    def __call__(
        self, image: torch.Tensor, anno: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply a geometric transformation.

        Args:
          image: input image to be transformed.
          anno: annotation to be transformed.

        Returns:
          Transformed image and annotation.
        """
        raise NotImplementedError


class ResizeTransform(GeometricDataAugmentation):
    def __init__(self):
        self.h = 536
        self.w = 960

    def __call__(
        self,
        image: torch.Tensor,
        anno: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        resized_img = functional.resize(
            image,
            size=[self.h, self.w],
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        resized_anno = functional.resize(
            anno,
            size=[self.h, self.w],
            interpolation=transforms.InterpolationMode.NEAREST,
        )
        return resized_img, resized_anno


class RandomTranslationTransform(GeometricDataAugmentation):
    """Randomly translate an image and its annotation.

    Translation values in (-1, 1) are treated as fractions of
    the image dimensions.  Values outside that range are treated
    as absolute pixel offsets.
    """

    def __init__(
        self,
        min_translation: float = 0,
        max_translation: float = 0,
    ):
        self.min_translation = min_translation
        self.max_translation = max_translation
        assert max_translation >= min_translation

    def __call__(
        self, image: torch.Tensor, anno: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert image.shape[1] == anno.shape[1], (
            "Dimensions of all input should be identical."
        )
        assert image.shape[2] == anno.shape[2], (
            "Dimensions of all input should be identical."
        )

        h, w = image.shape[1], image.shape[2]
        if -1 < self.min_translation < 1:
            tx = int(
                random.uniform(
                    self.min_translation * w,
                    self.max_translation * w,
                )
            )
            ty = int(
                random.uniform(
                    self.min_translation * h,
                    self.max_translation * h,
                )
            )
        else:
            tx = random.randint(
                int(self.min_translation),
                int(self.max_translation),
            )
            ty = random.randint(
                int(self.min_translation),
                int(self.max_translation),
            )

        image_translated = functional.affine(
            image,
            angle=0,
            translate=[tx, ty],
            scale=1.0,
            shear=[0, 0],
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        anno_translated = functional.affine(
            anno,
            angle=0,
            translate=[tx, ty],
            scale=1.0,
            shear=[0, 0],
            interpolation=transforms.InterpolationMode.NEAREST,
        )

        return image_translated, anno_translated


class RandomRotationTransform(GeometricDataAugmentation):
    """Randomly rotate an image and its annotation by a random angle."""

    def __init__(
        self,
        min_angle_in_deg: float = -180,
        max_angle_in_deg: float = 180,
    ):
        assert min_angle_in_deg < max_angle_in_deg

        self.min_angle_in_deg = min_angle_in_deg
        self.max_angle_in_deg = max_angle_in_deg

    def __call__(
        self, image: torch.Tensor, anno: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # dimension of each input should be identical
        assert image.shape[1] == anno.shape[1], (
            "Dimensions of all input should be identical."
        )
        assert image.shape[2] == anno.shape[2], (
            "Dimensions of all input should be identical."
        )

        angle = random.uniform(self.min_angle_in_deg, self.max_angle_in_deg)

        image_rotated = functional.rotate(
            image,
            angle=angle,
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        anno_rotated = functional.rotate(
            anno,
            angle=angle,
            interpolation=transforms.InterpolationMode.NEAREST,
        )

        return image_rotated, anno_rotated


class RandomGaussianRotationTransform(GeometricDataAugmentation):
    """Rotate by a Gaussian-sampled angle."""

    def __init__(self, mean: float = 0.0, std: float = 0.0):
        """Rotate by a Gaussian-sampled angle.

        Args:
            mean: mean rotation angle in radians.
            std: std of rotation angle in radians.
        """
        self.mean = mean
        self.std = std

    def __call__(
        self, image: torch.Tensor, anno: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # dimension of each input should be identical
        assert image.shape[1] == anno.shape[1], (
            "Dimensions of all input should be identical."
        )
        assert image.shape[2] == anno.shape[2], (
            "Dimensions of all input should be identical."
        )

        angle = random.gauss(self.mean, self.std)  # radians
        angle = angle * (180 / math.pi)

        image_rotated = functional.rotate(
            image,
            angle=angle,
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        anno_rotated = functional.rotate(
            anno,
            angle=angle,
            interpolation=transforms.InterpolationMode.NEAREST,
        )

        return image_rotated, anno_rotated


class RandomHorizontalFlipTransform(GeometricDataAugmentation):
    """Apply random horizontal flipping."""

    def __init__(self, prob: float = 0.5):
        """Apply random horizontal flipping.

        Args:
            prob: probability of flipping.
        """
        assert prob >= 0.0
        assert prob <= 1.0
        self.prob = prob

    def __call__(
        self, image: torch.Tensor, anno: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # dimension of each input should be identical
        assert image.shape[1] == anno.shape[1], (
            "Dimensions of all input should be identical."
        )
        assert image.shape[2] == anno.shape[2], (
            "Dimensions of all input should be identical."
        )

        if random.random() <= self.prob:
            image_hz_flipped = functional.hflip(image)
            anno_hz_flipped = functional.hflip(anno)

            return image_hz_flipped, anno_hz_flipped
        else:
            return image, anno


class RandomVerticalFlipTransform(GeometricDataAugmentation):
    """Apply random vertical flipping."""

    def __init__(self, prob: float = 0.5):
        """Apply random vertical flipping.

        Args:
            prob: probability of flipping.
        """
        assert prob >= 0.0
        assert prob <= 1.0
        self.prob = prob

    def __call__(
        self, image: torch.Tensor, anno: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # dimension of each input should be identical
        assert image.shape[1] == anno.shape[1], (
            "Dimensions of all input should be identical."
        )
        assert image.shape[2] == anno.shape[2], (
            "Dimensions of all input should be identical."
        )

        if random.random() <= self.prob:
            image_v_flipped = functional.vflip(image)
            anno_v_flipped = functional.vflip(anno)

            return image_v_flipped, anno_v_flipped
        else:
            return image, anno


class CenterCropTransform(GeometricDataAugmentation):
    """Extract a patch from the image center."""

    def __init__(
        self,
        crop_height: Optional[int] = None,
        crop_width: Optional[int] = None,
    ):
        """Set height and width of cropping region.

        Args:
            crop_height: Height of cropping region.
            crop_width: Width of cropping region.
        """
        self.crop_height = crop_height
        self.crop_width = crop_width

    def __call__(
        self, image: torch.Tensor, anno: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (self.crop_height is None) or (self.crop_width is None):
            return image, anno

        img_chans, img_height, img_width = image.shape[:3]
        anno_chans = anno.shape[0]

        if self.crop_width > img_width:
            raise ValueError("Crop width must not exceed image width")
        if self.crop_height > img_height:
            raise ValueError("Crop height must not exceed image height")

        image_cropped = functional.center_crop(
            image, [self.crop_height, self.crop_width]
        )
        anno_cropped = functional.center_crop(
            anno, [self.crop_height, self.crop_width]
        )

        assert image_cropped.shape[0] == img_chans, (
            "Cropped image has unexpected channels."
        )
        assert image_cropped.shape[1] == self.crop_height, (
            "Cropped image has wrong height."
        )
        assert image_cropped.shape[2] == self.crop_width, (
            "Cropped image has wrong width."
        )

        assert anno_cropped.shape[0] == anno_chans, (
            "Cropped anno has unexpected channels."
        )
        assert anno_cropped.shape[1] == self.crop_height, (
            "Cropped anno has wrong height."
        )
        assert anno_cropped.shape[2] == self.crop_width, (
            "Cropped anno has wrong width."
        )

        return image_cropped, anno_cropped


class RandomCropTransform(GeometricDataAugmentation):
    """Extract a random patch from image and annotation."""

    def __init__(
        self,
        crop_height: Optional[int] = None,
        crop_width: Optional[int] = None,
    ):
        """Set height and width of cropping region.

        Args:
          crop_height: Height of cropping region.
          crop_width: Width of cropping region.
        """
        self.crop_height = crop_height
        self.crop_width = crop_width

    def __call__(
        self, image: torch.Tensor, anno: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # dimension of each input should be identical
        assert image.shape[1] == anno.shape[1], (
            "Dimensions of all input should be identical."
        )
        assert image.shape[2] == anno.shape[2], (
            "Dimensions of all input should be identical."
        )

        if (self.crop_height is None) or (self.crop_width is None):
            return image, anno

        img_chans, img_height, img_width = image.shape[:3]
        anno_chans = anno.shape[0]

        if self.crop_width > img_width:
            raise ValueError(
                f"Crop width exceeds image width:"
                f" {self.crop_width} vs {img_width}"
            )
        if self.crop_height > img_height:
            raise ValueError(
                f"Crop height exceeds image height:"
                f" {self.crop_height} vs {img_height}"
            )

        max_x = img_width - self.crop_width
        x_start = random.randint(0, max_x)

        max_y = img_height - self.crop_height
        y_start = random.randint(0, max_y)

        assert (x_start + self.crop_width) <= img_width, (
            "Cropping region (width) exceeds image dims."
        )
        assert (y_start + self.crop_height) <= img_height, (
            "Cropping region (height) exceeds image dims."
        )

        image_cropped = functional.crop(
            image, y_start, x_start, self.crop_height, self.crop_width
        )
        anno_cropped = functional.crop(
            anno, y_start, x_start, self.crop_height, self.crop_width
        )

        assert image_cropped.shape[0] == img_chans, (
            "Cropped image has an unexpected number of channels."
        )
        assert image_cropped.shape[1] == self.crop_height, (
            "Cropped image has not the desired size."
        )
        assert image_cropped.shape[2] == self.crop_width, (
            "Cropped image has not the desired width."
        )

        assert anno_cropped.shape[0] == anno_chans, (
            "Cropped anno has an unexpected number of channels."
        )
        assert anno_cropped.shape[1] == self.crop_height, (
            "Cropped anno has not the desired size."
        )
        assert anno_cropped.shape[2] == self.crop_width, (
            "Cropped anno has not the desired width."
        )

        return image_cropped, anno_cropped


class MyRandomScaleTransform(GeometricDataAugmentation):
    """Apply random overall scaling."""

    def __init__(self, min_scale: float, max_scale: float, prob: float = 0.5):
        """Apply random overall scaling.

        Args:
            min_scale (float): minimum scaling factor.
            max_scale (float): maximum scaling factor.
            prob: probability of scaling.
        """
        assert prob >= 0.0
        assert prob <= 1.0
        assert min_scale > 0.0
        assert max_scale >= min_scale

        self.min_scale = min_scale
        self.max_scale = max_scale
        self.prob = prob

    def __call__(
        self, image: torch.Tensor, anno: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # dimension of each input should be identical
        assert image.shape[1] == anno.shape[1], (
            "Dimensions of all input should be identical."
        )
        assert image.shape[2] == anno.shape[2], (
            "Dimensions of all input should be identical."
        )

        img_chans, img_height, img_width = (
            image.shape[0],
            image.shape[1],
            image.shape[2],
        )
        anno_chans = anno.shape[0]

        if random.random() < self.prob:
            random_scale = random.uniform(self.min_scale, self.max_scale)

            image = functional.affine(
                image,
                angle=0,
                translate=[0, 0],
                scale=random_scale,
                shear=[0, 0],
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
            anno = functional.affine(
                anno,
                angle=0,
                translate=[0, 0],
                scale=random_scale,
                shear=[0, 0],
                interpolation=transforms.InterpolationMode.NEAREST,
            )

        assert img_chans == image.shape[0]
        assert img_height == image.shape[1]
        assert img_width == image.shape[2]

        assert anno_chans == anno.shape[0]
        assert img_height == anno.shape[1]
        assert img_width == anno.shape[2]

        return image, anno


class MyRandomAspectRatioTransform(GeometricDataAugmentation):
    """Rescale in one dimension to diversify aspect ratio."""

    def __init__(
        self,
        min_scale: float,
        max_scale: float,
        prob: float = 0.5,
    ):
        """Rescale in one dimension.

        See: http://www.robesafe.es/personal/
        eduardo.romera/pdfs/Romera18iv.pdf

        Args:
            min_scale (float): minimum scaling factor.
            max_scale (float): maximum scaling factor.
            prob: probability of scaling.
        """
        assert prob >= 0.0
        assert prob <= 1.0
        assert min_scale > 0.0
        assert max_scale >= min_scale

        self.min_scale = min_scale
        self.max_scale = max_scale
        self.prob = prob

    def __call__(
        self, image: torch.Tensor, anno: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # dimension of each input should be identical
        assert image.shape[1] == anno.shape[1], (
            "Dimensions of all input should be identical."
        )
        assert image.shape[2] == anno.shape[2], (
            "Dimensions of all input should be identical."
        )

        img_chans, img_height, img_width = (
            image.shape[0],
            image.shape[1],
            image.shape[2],
        )
        anno_chans = anno.shape[0]

        if random.random() < self.prob:
            random_scale = random.uniform(self.min_scale, self.max_scale)
            if random.random() > 0.5:
                # specify new height
                img_height_new = int(round(img_height * random_scale))
                img_width_new = img_width  # unchanged
            else:
                # specify new width
                img_height_new = img_height  # unchanged
                img_width_new = int(round(img_width * random_scale))

            image = functional.resize(
                image,
                size=[img_height_new, img_width_new],
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
            anno = functional.resize(
                anno,
                size=[img_height_new, img_width_new],
                interpolation=transforms.InterpolationMode.NEAREST,
            )

        assert img_chans == image.shape[0]
        assert anno_chans == anno.shape[0]

        return image, anno


class MyRandomShearTransform(GeometricDataAugmentation):
    """Apply random shear along x- and y-axis."""

    def __init__(
        self, max_x_shear: float, max_y_shear: float, prob: float = 0.5
    ):
        """Apply random shear.

        Args:
            x_shear (float): maximum shear along x-axis (in degrees).
            y_shear (float): maximum shear along y-axis (in degrees).
            prob: probability of shearing.
        """
        assert prob >= 0.0
        assert prob <= 1.0

        self.max_x_shear = max_x_shear
        self.max_y_shear = max_y_shear
        self.prob = prob

    def __call__(
        self, image: torch.Tensor, anno: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # dimension of each input should be identical
        assert image.shape[1] == anno.shape[1], (
            "Dimensions of all input should be identical."
        )
        assert image.shape[2] == anno.shape[2], (
            "Dimensions of all input should be identical."
        )

        img_chans, img_height, img_width = (
            image.shape[0],
            image.shape[1],
            image.shape[2],
        )
        anno_chans = anno.shape[0]

        if random.random() < self.prob:
            x_shear = random.uniform(-self.max_x_shear, self.max_x_shear)

            image = functional.affine(
                image,
                angle=0,
                translate=[0, 0],
                scale=1.0,
                shear=[x_shear, 0],
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
            anno = functional.affine(
                anno,
                angle=0,
                translate=[0, 0],
                scale=1.0,
                shear=[x_shear, 0],
                interpolation=transforms.InterpolationMode.NEAREST,
            )

        if random.random() < self.prob:
            y_shear = random.uniform(-self.max_y_shear, self.max_y_shear)

            image = functional.affine(
                image,
                angle=0,
                translate=[0, 0],
                scale=1.0,
                shear=[0, y_shear],
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
            anno = functional.affine(
                anno,
                angle=0,
                translate=[0, 0],
                scale=1.0,
                shear=[0, y_shear],
                interpolation=transforms.InterpolationMode.NEAREST,
            )

        assert img_chans == image.shape[0]
        assert img_height == image.shape[1]
        assert img_width == image.shape[2]

        assert anno_chans == anno.shape[0]
        assert img_height == anno.shape[1]
        assert img_width == anno.shape[2]

        return image, anno


class MyPaddingTransform(GeometricDataAugmentation):
    def __init__(self, divisor: int = 8):
        """Apply padding to right and bottom"""
        self.divisor = divisor

    def __call__(
        self, image: torch.Tensor, anno: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad image/anno height+width to a multiple of divisor."""
        img_height, img_width = image.shape[-2:]
        pad_h = 0
        pad_w = 0

        if img_height % self.divisor != 0:
            pad_h = self.divisor - (img_height % self.divisor)

        if img_width % self.divisor != 0:
            pad_w = self.divisor - (img_width % self.divisor)

        padded_img = functional.pad(
            image,
            [0, 0, pad_w, pad_h],
            fill=0,
            padding_mode="constant",
        )
        padded_anno = functional.pad(
            anno,
            [0, 0, pad_w, pad_h],
            fill=0,
            padding_mode="constant",
        )

        return padded_img, padded_anno


def get_geometric_augmentations(
    cfg, stage: str
) -> List[GeometricDataAugmentation]:
    assert stage in ["train", "val", "test", "predict"]

    geometric_augmentations = []

    augs = cfg[stage].get("geometric_data_augmentations")
    if not augs:
        return geometric_augmentations

    for tf_name in augs.keys():
        if tf_name == "resize":
            augmentor = ResizeTransform()
            geometric_augmentations.append(augmentor)

        if tf_name == "pad":
            padding_divisor = cfg[stage]["geometric_data_augmentations"][
                tf_name
            ]["divisor"]
            augmentor = MyPaddingTransform(padding_divisor)
            geometric_augmentations.append(augmentor)

        if tf_name == "random_hflip":
            augmentor = RandomHorizontalFlipTransform()
            geometric_augmentations.append(augmentor)

        if tf_name == "random_vflip":
            augmentor = RandomVerticalFlipTransform()
            geometric_augmentations.append(augmentor)

        if tf_name == "random_translation":
            min_translation = cfg[stage]["geometric_data_augmentations"][
                tf_name
            ]["min_translation"]
            max_translation = cfg[stage]["geometric_data_augmentations"][
                tf_name
            ]["max_translation"]
            augmentor = RandomTranslationTransform(
                min_translation, max_translation
            )
            geometric_augmentations.append(augmentor)

        if tf_name == "random_rotate_gaussian":
            mean = cfg[stage]["geometric_data_augmentations"][tf_name]["mean"]
            std = cfg[stage]["geometric_data_augmentations"][tf_name]["std"]
            augmentor = RandomGaussianRotationTransform(mean, std)
            geometric_augmentations.append(augmentor)

        if tf_name == "random_rotate":
            min_angle_in_deg = cfg[stage]["geometric_data_augmentations"][
                tf_name
            ]["min_angle_in_deg"]
            max_angle_in_deg = cfg[stage]["geometric_data_augmentations"][
                tf_name
            ]["max_angle_in_deg"]
            augmentor = RandomRotationTransform(
                min_angle_in_deg, max_angle_in_deg
            )
            geometric_augmentations.append(augmentor)

        if tf_name == "random_shear":
            max_x_shear = cfg[stage]["geometric_data_augmentations"][tf_name][
                "max_x_shear"
            ]
            max_y_shear = cfg[stage]["geometric_data_augmentations"][tf_name][
                "max_y_shear"
            ]
            augmentor = MyRandomShearTransform(max_x_shear, max_y_shear)
            geometric_augmentations.append(augmentor)

        if tf_name == "random_scale":
            min_scale = cfg[stage]["geometric_data_augmentations"][tf_name][
                "min_scale"
            ]
            max_scale = cfg[stage]["geometric_data_augmentations"][tf_name][
                "max_scale"
            ]
            augmentor = MyRandomScaleTransform(min_scale, max_scale)
            geometric_augmentations.append(augmentor)

        if tf_name == "random_aspect_ratio":
            min_ap_scale = cfg[stage]["geometric_data_augmentations"][tf_name][
                "min_scale"
            ]
            max_ap_scale = cfg[stage]["geometric_data_augmentations"][tf_name][
                "max_scale"
            ]
            augmentor = MyRandomAspectRatioTransform(
                min_ap_scale, max_ap_scale
            )
            geometric_augmentations.append(augmentor)

        if tf_name == "center_crop":
            crop_height = cfg[stage]["geometric_data_augmentations"][tf_name][
                "height"
            ]
            crop_width = cfg[stage]["geometric_data_augmentations"][tf_name][
                "width"
            ]
            augmentor = CenterCropTransform(crop_height, crop_width)
            geometric_augmentations.append(augmentor)

        if tf_name == "random_crop":
            crop_height = cfg[stage]["geometric_data_augmentations"][tf_name][
                "height"
            ]
            crop_width = cfg[stage]["geometric_data_augmentations"][tf_name][
                "width"
            ]
            augmentor = RandomCropTransform(crop_height, crop_width)
            geometric_augmentations.append(augmentor)

    assert len(geometric_augmentations) == len(
        cfg[stage]["geometric_data_augmentations"].keys()
    )
    return geometric_augmentations


# -------------------- COPY-PASTE AUGMENTATION --------------------


class CopyPasteWeed:
    """Paste weed instances from a donor image onto target
    image to address class imbalance."""

    def __init__(
        self,
        weed_class: int = 2,
        soil_class: int = 0,
        prob: float = 0.5,
        max_instances: int = 3,
    ):
        """Initialize copy-paste augmentation.

        Args:
            weed_class: class index for weeds. Default: 2.
            soil_class: class index for soil. Default: 0.
            prob: probability of applying the augmentation.
            max_instances: max weed instances to paste.
        """
        assert 0.0 <= prob <= 1.0
        assert max_instances >= 1
        self.weed_class = weed_class
        self.soil_class = soil_class
        self.prob = prob
        self.max_instances = max_instances

    def __call__(
        self,
        image: torch.Tensor,
        anno: torch.Tensor,
        donor_image: torch.Tensor,
        donor_anno: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Paste weed instances from donor onto target.

        Args:
            image: target image [C x H x W]
            anno: target annotation [H x W]
            donor_image: donor image [C x H x W]
            donor_anno: donor annotation [H x W]

        Returns:
            Tuple of augmented image and annotation.
        """
        if random.random() > self.prob:
            return image, anno

        donor_anno_np = donor_anno.cpu().numpy()
        weed_np = (donor_anno_np == self.weed_class).astype(np.uint8)
        if not weed_np.any():
            return image, anno

        labeled, n_instances = ndimage_label(weed_np)
        if n_instances == 0:
            return image, anno

        indices = list(range(1, n_instances + 1))
        random.shuffle(indices)
        selected = indices[: self.max_instances]

        # Build combined mask in numpy, skip high-overlap
        anno_np = anno.cpu().numpy()
        combined = np.zeros_like(weed_np, dtype=bool)
        for inst_id in selected:
            inst_mask = labeled == inst_id
            non_soil = anno_np[inst_mask] != self.soil_class
            if non_soil.sum() > 0.5 * inst_mask.sum():
                continue
            combined |= inst_mask

        if not combined.any():
            return image, anno

        # Single tensor conversion and paste
        mask_t = torch.from_numpy(combined).to(image.device)
        image = image.clone()
        anno = anno.clone()
        image[:, mask_t] = donor_image[:, mask_t]
        anno[mask_t] = self.weed_class

        return image, anno


class MosaicAugmentation:
    """Stitch 4 images into a 2x2 grid.

    Each image is resized to fill its quadrant. A random
    center offset adds variety to quadrant proportions.

    Args:
        prob: Probability of applying mosaic.
        output_size: Target output size (H, W).
    """

    def __init__(
        self,
        prob: float = 0.5,
        output_size: Tuple[int, int] = (768, 768),
    ):
        self.prob = prob
        self.output_h, self.output_w = output_size

    def __call__(
        self,
        images: List[torch.Tensor],
        annos: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Combine 4 image-anno pairs into a mosaic.

        Args:
            images: List of 4 tensors [C x H x W].
            annos: List of 4 tensors [1 x H x W].

        Returns:
            Tuple of (image [C x out_h x out_w],
                      anno [1 x out_h x out_w]).
        """
        assert len(images) == 4 and len(annos) == 4

        oh, ow = self.output_h, self.output_w

        # Random center with +/-25% jitter
        cy = oh // 2 + random.randint(-oh // 8, oh // 8)
        cx = ow // 2 + random.randint(-ow // 8, ow // 8)

        c = images[0].shape[0]
        out_img = torch.zeros(c, oh, ow, dtype=images[0].dtype)
        out_anno = torch.zeros(1, oh, ow, dtype=annos[0].dtype)

        # Quadrant sizes based on center point
        sizes = [
            (cy, cx),
            (cy, ow - cx),
            (oh - cy, cx),
            (oh - cy, ow - cx),
        ]
        # Quadrant offsets (top-left corner)
        offsets = [
            (0, 0),
            (0, cx),
            (cy, 0),
            (cy, cx),
        ]

        for i in range(4):
            qh, qw = sizes[i]
            if qh <= 0 or qw <= 0:
                continue
            oy, ox = offsets[i]

            img_r = functional.resize(
                images[i],
                [qh, qw],
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            )
            anno_r = functional.resize(
                annos[i],
                [qh, qw],
                interpolation=transforms.InterpolationMode.NEAREST,
            )

            out_img[:, oy : oy + qh, ox : ox + qw] = img_r
            out_anno[:, oy : oy + qh, ox : ox + qw] = anno_r

        return out_img, out_anno


def get_mosaic_augmentation(cfg, stage: str) -> Optional["MosaicAugmentation"]:
    """Return MosaicAugmentation if configured, else None."""
    if stage != "train":
        return None
    aug_cfg = cfg.get(stage, {}).get("mosaic")
    if aug_cfg is None:
        return None
    # Derive output size from random_crop config
    geo_cfg = cfg.get(stage, {}).get("geometric_data_augmentations", {})
    crop_cfg = geo_cfg.get("random_crop", {})
    h = crop_cfg.get("height", 768)
    w = crop_cfg.get("width", 768)
    return MosaicAugmentation(
        prob=aug_cfg.get("prob", 0.5),
        output_size=(h, w),
    )


def get_copy_paste_augmentation(cfg, stage: str) -> Optional["CopyPasteWeed"]:
    """Return CopyPasteWeed if configured, else None."""
    if stage != "train":
        return None
    aug_cfg = cfg.get(stage, {}).get("copy_paste_weed")
    if aug_cfg is None:
        return None
    return CopyPasteWeed(
        weed_class=aug_cfg.get("weed_class", 2),
        soil_class=aug_cfg.get("soil_class", 0),
        prob=aug_cfg.get("prob", 0.5),
        max_instances=aug_cfg.get("max_instances", 3),
    )
