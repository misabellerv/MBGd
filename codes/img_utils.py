import cv2
import numpy as np
import skimage.feature as ft
import torch
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

from detectron2.structures.boxes import Boxes


def add_bb_on_image(image, bounding_box, color=(255, 0, 0), thickness=5, label=None):

    image = np.asarray(image).clip(0, 255).astype(np.uint8)

    # color
    r = int(color[0])
    g = int(color[1])
    b = int(color[2])

    # font params
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # font_scale = 0.5
    # font_thickness = 1

    # bb limits
    x_i = int(bounding_box[0])
    y_i = int(bounding_box[1])
    x_o = int(bounding_box[2])
    y_o = int(bounding_box[3])

    # draw bounding box
    image = cv2.rectangle(image, (x_i, y_i), (x_o, y_o), color=(b, g, r), thickness=thickness)

    # TODO: put text
    if label is not None:

        label = label.split("-")[0]

        # Finds space required by the text so that we can put a background with that amount of width.
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 5)

        # Prints the text.
        image = cv2.rectangle(image, (x_i, y_i - 35), (x_i + w, y_i), (b, g, r), -1)
        image = cv2.putText(image, label, (x_i, y_i - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                            (255, 255, 255), 5)

        # cv2.rectangle(image, (x_i, y_i - 35), (x_o, y_i), (b, g, r), -1)
        # cv2.putText(image,
        #             label.split("-")[0], (x_i, y_i - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (b, g, r),
        #             5)
    return image


def add_bboxes_on_image(image, bboxes, color=(255, 0, 0), thickness=5, label=None):
    if isinstance(bboxes, Boxes):
        bboxes = bboxes.tensor

    if isinstance(bboxes, (np.ndarray, torch.Tensor, list)):
        for idx in range(len(bboxes)):
            image = add_bb_on_image(image, bboxes[idx], color, thickness)
        return image

    for key, bb in bboxes.items():
        image = add_bb_on_image(image, bb, color, thickness, label=key)
    return image


def compute_mse(img1, img2):
    return mean_squared_error(img1, img2)


def compute_psnr(img1, img2):
    return peak_signal_noise_ratio(img1, img2)


def compute_ssim(img1, img2, multichannel=True):
    return structural_similarity(img1, img2, multichannel=multichannel)


def compute_lbp(img_rgb, p=24, r=3, method='uniform'):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    lbp = ft.local_binary_pattern(img_gray, p, r, method)

    return lbp


def hsv_hist(img_rgb, n_bins=64, weights=None):

    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(img_hsv)

    if weights is not None and weights.ndim > 1:
        weights = weights[:, :, 0]

    hist_h, _ = np.histogram(h, bins=n_bins, density=True, weights=weights)
    hist_s, _ = np.histogram(s, bins=n_bins, density=True, weights=weights)
    hist_v, _ = np.histogram(v, bins=n_bins, density=True, weights=weights)

    return hist_h, hist_s, hist_v


def lbp_hist(img_rgb, n_bins=64, weights=None, p=24, r=3, method='uniform'):
    lbp = compute_lbp(img_rgb, p=p, r=r, method=method)

    if weights is not None and weights.ndim > 1:
        weights = weights[:, :, 0]

    hist_lbp, _ = np.histogram(lbp, bins=n_bins, density=True, weights=weights)

    return hist_lbp


def phase_correlation(img, img_offset, scale=None):
    """Image translation registration by cross-correlation.
    It obtains an estimate of the cross-correlation peak by an FFT.

    It is very similar to `skimage.feature.register_translation`.
    However, our function runs faster because we restrict it to our application.


    Args:
        img (array): image.
        img_offset (array): offset image. Must have the same dim as `img`.
        scale (float, optional): If not `None`, rescale input images to run faster.
        Defaults to None.
    Returns:
        array:  shift vector (in pixels) required to register `img_offset` with
        `img`.  Axis ordering is consistent with numpy (e.g. Z, Y, X)
    """

    if img.shape != img_offset.shape:
        raise ValueError("Error: images must be same size")

    # if images in BGR, tranform to grayscal and use fft2
    # which is much faster than using fftn
    # if img.ndim > 2:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # if img_offset.ndim > 2:
    #     img_offset = cv2.cvtColor(img_offset, cv2.COLOR_BGR2GRAY)

    img = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    img_offset = np.float32(cv2.cvtColor(img_offset, cv2.COLOR_BGR2GRAY))

    if scale is not None:
        img = scale_img(img, scale)
        img_offset = scale_img(img_offset, scale)

    (x, y), c = cv2.phaseCorrelate(img, img_offset)

    # img_fft = np.fft.fft2(img)
    # img_offset_fft = np.fft.fft2(img_offset)

    # shape = img_fft.shape
    # midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

    # R = img_fft * img_offset_fft.conj()
    # # R /= np.absolute(R)  # normalize give wrong results for large frames interval (???)
    # # (print('norm'))
    # cross_correlation = np.fft.ifft2(R)

    # shifts = np.unravel_index(np.argmax(np.absolute(cross_correlation)), shape)
    # shifts = np.array(shifts, dtype=np.float64)
    # shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if scale is not None:
        x /= scale
        y /= scale

    return (x, y), c


def scale_img(src, scale):
    # calculate new dimensions
    new_w = int(src.shape[1] * scale)
    new_h = int(src.shape[0] * scale)

    return cv2.resize(src, (new_w, new_h))


def is_on_margin(box, img_size, margin_size):
    """Check if a box is in a margin of size `margin_size`.

    Args:
        box (list): containing box dimensions (x,y,x,y)
        img_size (tuple): image size (h,w)
        margin_size (tuple): margin size (h,w)
    """
    h_margin, w_margin = margin_size
    h_img, w_img = img_size

    x_tl = box[0]
    y_tl = box[1]
    x_br = box[2]
    y_br = box[3]

    # check if is in right margin
    if x_tl > (w_img - w_margin):
        return True

    # check if is in left margin
    if x_br < (w_margin):
        return True

    # check if is in bottom margin
    if y_tl > (h_img - h_margin):
        return True

    # check if is in top margin
    if y_br < (h_margin):
        return True

    return False