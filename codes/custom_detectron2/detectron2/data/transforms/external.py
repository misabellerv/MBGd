#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 10:22:52 2021
@author: Sidharth Sharma
"""
import numpy as np
from .augmentation import Augmentation  #find the imageAugmentor equilavent in dectron2
from fvcore.transforms.transform import Transform, NoOpTransform

__all__ = ['Albumentations']


class AlbumentationsTransform(Transform):
    def __init__(self, aug, params):
        self.aug = aug
        self.params = params

    def apply_image(self, img):
        return self.aug.apply(img, **self.params)

    def apply_coords(self, coords):
        return coords

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        try:

            return np.array(self.aug.apply_to_bboxes(box.tolist(), **self.params))
        except AttributeError:
            return box

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        try:
            return self.aug.apply_to_mask(segmentation, **self.params)
        except AttributeError:
            return segmentation


class Albumentations(Augmentation):
    """
    Wrap an augmentor form the albumentations library: https://github.com/albu/albumentations.
    Coordinate augmentation is not supported by the library.
    Example:
    .. code-block:: python
        import detectron2.data.transforms.external as  A
        import albumentations as AB
        ## Resize
        #augs1 = A.Albumentations(AB.SmallestMaxSize(max_size=1024, interpolation=1, always_apply=False, p=1))
        #augs1 = A.Albumentations(AB.RandomScale(scale_limit=0.8, interpolation=1, always_apply=False, p=0.5))
        ## Rotate
        augs1 = A.Albumentations(AB.RandomRotate90(p=1))
        transform_1 = augs1(input)
        image_transformed_1 = input.image
        cv2_imshow(image_transformed_1)
    """
    def __init__(self, augmentor):
        """
        Args:
            augmentor (albumentations.BasicTransform):
        """
        #super(Albumentations, self).__init__() - using python > 3.7 no need to call rng
        self._aug = augmentor

    # def get_transform(self, img):
    #     return AlbumentationsTransform(self._aug, self._aug.get_params())

    def get_transform(self, image):
        do = self._rand_range() < self._aug.p
        if do:
            params = self.prepare_param(image)
            return AlbumentationsTransform(self._aug, params)
        else:
            return NoOpTransform()

    def prepare_param(self, image):
        params = self._aug.get_params()
        if self._aug.targets_as_params:
            targets_as_params = {"image": image}
            params_dependent_on_targets = self._aug.get_params_dependent_on_targets(
                targets_as_params)
            params.update(params_dependent_on_targets)
        params = self._aug.update_params(params, **{"image": image})
        return params
