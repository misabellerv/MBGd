import custom_detectron2.detectron2.data.transforms as T
import albumentations as AB
import numpy as np
import random
import math
import cv2

## random perspective ###


class RandomPerspective(T.Augmentation):
    def __init__(self,
                 degrees=10,
                 translate=.1,
                 scale=.1,
                 shear=10,
                 perspective=0.0,
                 border=(0, 0)):
        super().__init__()
        self.degress = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border
        # self._init(locals())

    def get_transform(self, image):
        return RandomPerspectiveTransform(self.degress, self.translate, self.scale, self.shear,
                                          self.perspective, self.border)


class RandomPerspectiveTransform(T.Transform):
    def __init__(self,
                 degrees=10,
                 translate=.1,
                 scale=.1,
                 shear=10,
                 perspective=0.0,
                 border=(0, 0)):
        super().__init__()
        self.degress = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border

    def apply_image(self, img):
        height = img.shape[0] + self.border[0] * 2  # shape(h,w,c)
        width = img.shape[1] + self.border[1] * 2

        # Center
        C = np.eye(3)
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-self.degress, self.degress)  # x rotation in degrees
        s = random.uniform(1 - self.scale, 1 + self.scale)  # x scale
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - self.translate,
                                 0.5 + self.translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate,
                                 0.5 + self.translate) * height  # y translation (pixels)

        # Combined rotation matrix
        self.M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        if (self.border[0] != 0) or (self.border[1] != 0) or (self.M !=
                                                              np.eye(3)).any():  # image changed
            if self.perspective:
                augmented_img = cv2.warpPerspective(img,
                                                    self.M,
                                                    dsize=(width, height),
                                                    borderValue=(114, 114, 114))
            else:  # affine
                augmented_img = cv2.warpAffine(img,
                                               self.M[:2],
                                               dsize=(width, height),
                                               borderValue=(114, 114, 114))

            return augmented_img

    def apply_coords(self, coords):
        modified_coords = coords.copy()
        xy = np.ones((4, 3))
        xy[:, :2] = modified_coords
        xy = xy @ self.M.T
        xy = xy[:, :2] / xy[:,
                            2:3] if self.perspective else xy[:, :2]  # perspective rescale or affine

        return xy


## Augment HSV ###


class AugmentHSV(T.Augmentation):
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        # self._init(locals())

    def get_transform(self, image):
        return AugmentHSVTransform(self.hgain, self.sgain, self.vgain)


class AugmentHSVTransform(T.Transform):
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5):
        super().__init__()
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        # self._init(locals())

    def apply_image(self, img):
        if self.hgain or self.sgain or self.vgain:
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain
                                               ] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val,
                                                                                      lut_val)))
            augmented_img = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)

            return augmented_img

    def apply_coords(self, coords):
        # coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        # coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords


# Cutout ###


class CutOut(T.Augmentation):
    def __init__(self, p=0.5):
        self.p = p
        # self._init(locals())

    def get_transform(self, image):
        return CutOutTransform(p=self.p)


class CutOutTransform(T.Transform):
    def __init__(self, p):
        super().__init__()
        self.p = p
        # self._init(locals())

    def apply_image(self, img):
        augmented_image = img.copy()

        if random.random() < self.p:
            h, w = augmented_image.shape[:2]
            scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [
                0.03125
            ] * 16  # image size fraction

            self.list_cutout_boxes = []
            for s in scales:
                mask_h = random.randint(1, int(h * s))  # create random masks
                mask_w = random.randint(1, int(w * s))

                # box
                xmin = max(0, random.randint(0, w) - mask_w // 2)
                ymin = max(0, random.randint(0, h) - mask_h // 2)
                xmax = min(w, xmin + mask_w)
                ymax = min(h, ymin + mask_h)

                # apply random color mask
                augmented_image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

                self.list_cutout_boxes.append({
                    'scale':
                    s,
                    'box':
                    np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                })

                # return unobscured labels
                # if len(labels) and s > 0.03:
                #     box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                #     ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
                #     labels = labels[ioa < 0.60]  # remove >60% obscured labels

        return augmented_image

    def apply_coords(self, coords):
        modified_coords = coords.copy()

        # TODO: implement the following logic
        # for cutout_box in self.list_cutout_boxes:
        # s = cutout_box['scale']
        # box = cutout_box['box']

        # if len(modified_coords) and s > 0.03:
        #     ioa = bbox_ioa(box, modified_coords)  # intersection over area
        #     modified_coords = modified_coords[ioa < 0.60]  # remove >60% obscured labels

        return modified_coords


#####
class Clahe(T.Augmentation):
    def __init__(self, clip_lim, win_size):
        self.clip_lim = clip_lim
        self.win_size = win_size
        self._init(locals())

    def get_transform(self, image):
        return ClaheTransform(self.clip_lim, self.win_size)


class ClaheTransform(T.Transform):
    def __init__(self, clip_lim, win_size):
        super().__init__()
        self.clip_limit = clip_lim
        self.win_size = win_size
        self._set_attributes(locals())

    def apply_image(self, img):
        transform = AB.CLAHE(clip_limit=self.clip_lim,
                             tile_grid_size=(self.win_size, self.win_size))
        augmented_image = transform(image=img)['image']
        return augmented_image

    def apply_coords(self, coords):
        # coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        # coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation)
        return segmentation

    def inverse(self):
        return ClaheTransform(self.clip_lim, self.win_size)


def bbox_ioa(box1, box2, eps=1e-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area