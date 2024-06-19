import os
from detectron2.config import get_cfg
from LossEvalHookPocket import MyTrainer
from detectron2.utils.logger import setup_logger
import argparse
import logging
import warnings
import yaml
from register_dataset import register_mosquitoes


warnings.filterwarnings("ignore")

# Setup logger
logger = logging.getLogger("lm")

# Argument parser
parser = argparse.ArgumentParser(description="MBG Training")
parser.add_argument("--config-file", default="config.yaml", metavar="FILE", help="path to the general config file")
parser.add_argument("--data-train", default=None, metavar="FILE", help="path to data")
parser.add_argument("--data-val", default=None, metavar="FILE", help="path to data")
args = parser.parse_args()

# Extract fold index from data_train argument
fold_index = args.data_train.split('_')[1][-1]
obj_index = args.data_train.split('_')[-1]
fold_index = f"fold{fold_index}_{obj_index}"

# Initialize Detectron2 configuration
cfg = get_cfg()

# Load the general YAML configuration
with open(args.config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

# Access the specific model configuration
model_name = (config['TRAINING']['MODEL_NAME']).upper()
model_config = config['MODELS'][model_name]

# Apply model-specific configurations
cfg.merge_from_file(model_config['_BASE_'])
cfg.DATASETS.TRAIN = (args.data_train,)
cfg.DATASETS.VAL = (args.data_val,)
cfg.MODEL.WEIGHTS = model_config['MODEL']['WEIGHTS']
cfg.MODEL.MASK_ON = model_config['MODEL']['MASK_ON']
cfg.MODEL.RESNETS.DEPTH = model_config['MODEL']['RESNETS']['DEPTH']
cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = model_config['MODEL']['RPN']['PRE_NMS_TOPK_TRAIN']
cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = model_config['MODEL']['RPN']['PRE_NMS_TOPK_TEST']
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = model_config['MODEL']['RPN']['POST_NMS_TOPK_TRAIN']
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = model_config['MODEL']['RPN']['POST_NMS_TOPK_TEST']
cfg.MODEL.RPN.NMS_THRESH = model_config['MODEL']['RPN']['NMS_THRESH']
cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = model_config['MODEL']['ROI_BOX_HEAD']['POOLER_SAMPLING_RATIO']
cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = model_config['MODEL']['ROI_BOX_HEAD']['POOLER_RESOLUTION']
cfg.MODEL.BACKBONE.FREEZE_AT = model_config['MODEL']['BACKBONE']['FREEZE_AT']
cfg.MODEL.ROI_HEADS.NUM_CLASSES = model_config['MODEL']['ROI_HEADS']['NUM_CLASSES']

# Add training parameters
cfg.VAL_PERIOD = config['TRAINING']['VAL_PERIOD']
cfg.DATALOADER.NUM_WORKERS = config['TRAINING']['TRAINING_NUM_WORKERS']
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config['TRAINING']['BATCH_SIZE_PER_IMAGE']
cfg.SOLVER.IMS_PER_BATCH = config['TRAINING']['IMAGES_PER_BATCH']
cfg.SOLVER.BASE_LR = config['TRAINING']['LEARNING_RATE']
cfg.SOLVER.WEIGHT_DECAY = config['TRAINING']['WEIGHT_DECAY']
cfg.SOLVER.STEPS = config['TRAINING']['STEPS']
cfg.SOLVER.MAX_ITER = config['TRAINING']['MAX_ITER']
cfg.MIN_DELTA = config['TRAINING'].get('MIN_DELTA')
cfg.PATIENCE = config['TRAINING'].get('PATIENCE') 
cfg.VAL_PERIOD = config['TRAINING'].get('VAL_PERIOD')  
cfg.AUGMENTATION = config['AUGMENTATION'].get('ENABLE') 

# Define output directory
cfg.OUTPUT_DIR = config['TEST']['OUTPUT_DIR']
cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, fold_index)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
setup_logger(cfg.OUTPUT_DIR)

# Register datasets using the parameters from JSON
register_mosquitoes()

# Save the final configuration to a file
with open(os.path.join(cfg.OUTPUT_DIR, "training_config.yaml"), "w") as f:
    f.write(cfg.dump())

# Train the model
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
