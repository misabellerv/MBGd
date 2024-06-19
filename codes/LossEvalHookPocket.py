import custom_detectron2.detectron2.data.transforms as T
import os
import torch
import detectron2.utils.comm as comm
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import COCOEvaluator
import nni
import copy
import numpy as np
from detectron2.engine.hooks import HookBase, PeriodicWriter
import albumentations as AB
from detectron2.structures import BoxMode, Boxes
from custom_augmentations import AugmentHSV, RandomPerspective
from custom_detectron2.detectron2.data import DatasetMapper, detection_utils
import yaml

# Load the general YAML configuration
with open('./configs/config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

class ValidationLoss(HookBase):
    def __init__(self, cfg, model, checkpointer, min_delta=0.001, patience=2000) -> None:
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.DATASETS.TRAIN = self.cfg.DATASETS.VAL
        self.cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
        self._loader = iter(build_detection_train_loader(self.cfg))
        self.model = model
        self._checkpointer = checkpointer
        self.best_loss = float('inf')
        self.min_delta = min_delta
        self.patience = patience
        self.counter = 0
        self.stop_training = False
        self.cfg.OUTPUT_DIR = cfg.OUTPUT_DIR

    def after_step(self):
        if self.stop_training:
            return
        
        data = next(self._loader)
        with torch.no_grad():
            loss_dict = self.model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {
                "val_" + k: v.item()
                for k, v in comm.reduce_dict(loss_dict).items()
            }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            
            if comm.is_main_process():
                self.trainer.storage.put_scalars(total_val_loss=losses_reduced, **loss_dict_reduced)

            nni.report_intermediate_result(losses_reduced)

            
            # Save the best model so far
            if losses_reduced < self.best_loss - self.min_delta:
                self.best_loss = losses_reduced
                self._checkpointer.save("model_early_stopping")
                torch.save({'iteration': self.trainer.iter}, f"{self.cfg.OUTPUT_DIR}/iteration_best.pth")
                self.counter = 0  # Reset counter if improved
            else:
                self.counter += 1
                patience = config['TRAINING'].get('PATIENCE')
                print(f'\nPATIENCE COUNTER: {self.counter}/{patience}\n')

                if self.counter >= self.patience:
                    print(f"Stopping at iteration {self.trainer.iter} due to no improvement in validation loss.")
                    self.stop_training = True
                    self.cfg.SOLVER.MAX_ITER = self.trainer.iter
                    self.trainer.storage.put_scalar('stopped_at', self.trainer.iter)
                    raise StopIteration

    def after_train(self):
        nni.report_final_result(self.trainer.storage.latest()["total_val_loss"][0])

def build_train_aug(cfg):
    augs = [
        T.Albumentations(AB.Blur(p=0.01)),
        T.Albumentations(AB.MedianBlur(p=0.01)),
        T.Albumentations(AB.ToGray(p=0.01)),
        T.Albumentations(AB.CLAHE(p=0.01)),
        T.Albumentations(AB.RandomBrightnessContrast(p=0.0)),
        T.Albumentations(AB.RandomGamma(p=0.0)),
        T.Albumentations(AB.ImageCompression(quality_lower=75, p=0.0)),
    ]

    augs.append(AugmentHSV(hgain=0.015, sgain=0.7, vgain=0.4))
    augs.append(RandomPerspective(degrees=0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0))

    augs.append(T.RandomFlip())

    augs.append(
        T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN,
                             cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING))
    return augs

class MyTrainer(DefaultTrainer):
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, ValidationLoss(self.cfg, self.model, self.checkpointer, self.cfg.MIN_DELTA, self.cfg.PATIENCE))
        hooks[-1] = PeriodicWriter(self.build_writers(), period=self.cfg.VAL_PERIOD)
        return hooks

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        coco_evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
        return [coco_evaluator]
    
    @classmethod
    def build_train_loader(cls, cfg):
        
        if config['AUGMENTATION'].get('ENABLE') == True:
            
            print("USING CUSTOM AUGMENTATIONS! FOR MORE INFORMATIONS CHECK LossEvalHook.py")
            mapper = CustomDatasetMapper(cfg, is_train=True, augmentations=build_train_aug(cfg))
            
        elif config['AUGMENTATION'].get('ENABLE') == False:
            print("USING DEFAULT AUGMENTATIONS! FOR MORE INFORMATIONS CHECK LossEvalHook.py")
            mapper = None
            
        return build_detection_train_loader(cfg, mapper=mapper)

class CustomDatasetMapper(DatasetMapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = detection_utils.read_image(dataset_dict["file_name"], format=self.image_format)
        h, w = image[:2]
        detection_utils.check_image_size(dataset_dict, image)

        # annos = dataset_dict["annotations"]
        # instances = detection_utils.annotations_to_instances(annos, image.shape[:2])

        # boxes = instances.get("gt_boxes").tensor

        # # Transform XYXY_ABS -> XYXY_REL
        # boxes = np.array(boxes) / np.array(
        #     [image.shape[1], image.shape[0], image.shape[1], image.shape[0]])

        # augmentation in fact
        aug_input = T.AugInput(image)  #, boxes=boxes)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        # boxes = aug_input.boxes
        h, w = image[:2]

        # Transform XYXY_REL -> XYXY_ABS
        # boxes = np.array(boxes) * np.array(
        #     [image.shape[1], image.shape[0], image.shape[1], image.shape[0]])

        # instances.gt_boxes = Boxes(boxes)

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                detection_utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices)
                for obj in dataset_dict.pop("annotations") if obj.get("iscrowd", 0) == 0
            ]
            instances = detection_utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format)

            dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)

        return dataset_dict