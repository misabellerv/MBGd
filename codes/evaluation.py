import contextlib
import io
import logging
import os

import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO

from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation import DatasetEvaluator
from detectron2.structures.boxes import Boxes, BoxMode, pairwise_iou
from detectron2.structures.instances import Instances
from detectron2.utils.logger import create_small_table
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference_single_image
from img_utils import is_on_margin
import numpy as np


class CfnMat(DatasetEvaluator):
    def __init__(self, dataset_name, thr=0.5, output_dir=None):
        self.thr = thr

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(f"json_file was not found in MetaDataCatalog for '{dataset_name}'")

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset

    def reset(self):
        self._predictions = []
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = -1

    def process(self, inputs, outputs):

        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            prediction["instances"] = output["instances"].to(self._cpu_device)
            self._predictions.append(prediction)

    def evaluate(self):

        self._logger.info("Evaluating confusion matrix ...")
        print("Evaluating confusion matrix ...")
        imgs_errors = []

        for pred in self._predictions:

            pred_boxes = pred["instances"].pred_boxes

            ann_ids = self._coco_api.getAnnIds(imgIds=pred["image_id"])
            anno = self._coco_api.loadAnns(ann_ids)

            gt_boxes = [
                BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS) for obj in anno
                if obj["iscrowd"] == 0
            ]

            gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
            gt_boxes = Boxes(gt_boxes)

            tp, fp, fn = self._cnf_mat(pred_boxes, gt_boxes, thr=self.thr)

            assert (tp + fn) == len(gt_boxes)

            if fp or fn:
                imgs_errors.append({
                    "img_id": pred["image_id"],
                    "gt": len(gt_boxes),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                })

            self.tp += tp
            self.fp += fp
            self.fn += fn

        # self.precision = self.tp / (self.tp + self.fp)
        # self.recall = self.tp / (self.tp + self.fn)
        # self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)

        res = {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            # "P": self.precision,
            # "R": self.recall,
            # "F1": self.f1,
        }

        self._logger.info("Confusion matrix metrics: \n" + create_small_table(res))
        print("Confusion matrix metrics: \n" + create_small_table(res))

        self._logger.info("Images with errors: ")
        print("Images with errors: ")

        for img_errors in imgs_errors:
            self._logger.info("{}".format(img_errors))
            print(img_errors)

        # print("True positive:{} ".format(self.tp))
        # print("False positive: {} ".format(self.fp))
        # print("False negative: {} ".format(self.fn))

        return res

    # @staticmethod
    def _cnf_mat(self, pred, gt, thr=0.5):
        gt_overlaps = torch.zeros(len(gt))
        overlaps = pairwise_iou(pred, gt)

        all_boxe_ind = torch.zeros(len(pred))

        for j in range(min(len(pred), len(gt))):

            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0

            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]

            # record the iou coverage of this gt box
            gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert gt_overlaps[j] == gt_ovr

            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

            if gt_ovr >= thr:
                all_boxe_ind[box_ind] = 1

        tp = (gt_overlaps >= thr).int().sum().item()
        assert tp >= 0

        fp = len(pred) - tp
        assert fp >= 0

        fn = len(gt) - tp
        assert fn >= 0

        # return tp, fp, fn, all_boxe_ind
        return tp, fp, fn


def filter_preds_score_image(instances: Instances, score_thresh: float):

    boxes = instances.get('pred_boxes').tensor
    scores = instances.get('scores')
    pred_classes = instances.get('pred_classes')
    image_shape = instances.image_size

    # filtering out low-confidence detections.
    filter_mask = scores >= score_thresh  # R x K
    # filter_inds = filter_mask.nonzero()

    boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    pred_classes = pred_classes[filter_mask]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = pred_classes

    # result, filter_inds = fast_rcnn_inference_single_image(boxes, scores, image_shape, score_thresh,
    #                                                        nms_thresh, topk_per_image)

    return result


def filter_preds_score_video(preds: dict, score_thresh: float):
    preds = {
        frame: {
            'instances': filter_preds_score_image(
                preds[frame]['instances'],
                score_thresh,
            )
        }
        for frame in preds.keys()
    }

    return preds


def filter_preds_margin_image(instances: Instances, margin_size: tuple):
    boxes = instances.get('pred_boxes').tensor
    scores = instances.get('scores')
    pred_classes = instances.get('pred_classes')
    image_shape = instances.image_size

    filter_mask = [is_on_margin(box, image_shape, margin_size) for box in boxes]

    if any(filter_mask):
        print('entrou aqui!')

    # actualy I want to keep boxes that are not in the margin
    filter_mask = [not elem for elem in filter_mask]

    boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    pred_classes = pred_classes[filter_mask]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = pred_classes

    return result


def filter_preds_margin_video(preds: dict, margin_size: tuple):
    preds = {
        frame: {
            'instances': filter_preds_margin_image(
                preds[frame]['instances'],
                margin_size,
            )
        }
        for frame in preds.keys()
    }

    return preds


def filter_gt_margin_image(object: dict, img_size: tuple, margin_size: tuple):

    new_object = {
        obj: box
        for obj, box in object.items() if not (is_on_margin(box, img_size, margin_size))
    }

    return new_object


def filter_annot_margin_image(object: dict, img_size: tuple, margin_size: tuple):
    boxes = object['boxes']
    frames = object['frames']

    filter_mask = [is_on_margin(box, img_size, margin_size) for box in boxes]
    # actualy I want to keep boxes that are not in the margin
    filter_mask = [not elem for elem in filter_mask]
    filter_mask = np.where(filter_mask)[0]

    new_boxes = [boxes[n] for n in filter_mask]
    new_frames = [frames[n] for n in filter_mask]

    result = {'boxes': new_boxes, 'frames': new_frames}

    return result


def filter_annot_margin_video(annot: dict, img_size: tuple, margin_size: tuple):
    annot = {
        obj: filter_annot_margin_image(
            annot[obj],
            img_size,
            margin_size,
        )
        for obj in annot.keys()
    }

    return annot