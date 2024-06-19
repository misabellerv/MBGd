import argparse
import os
import pandas as pd
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, inference_on_dataset
from detectron2.utils.logger import setup_logger
from evaluation import CfnMat
from register_dataset import register_mosquitoes
import numpy as np
import yaml

# Argument parser
parser = argparse.ArgumentParser(description="MBG Eval")
parser.add_argument("--config-file", default=None, metavar="FILE", help="model name")
parser.add_argument("--data-train", default=None, metavar="FILE", help="path to training data")
parser.add_argument("--data-val", default=None, metavar="FILE", help="path to test data")
args = parser.parse_args()

# Extract fold index
fold_index = args.data_train.split('_')[1][-1]
obj = args.data_train.split('_')[-1]
fold_index = fold_index.replace('mbg_train', '')
fold_dir = f"fold{fold_index}_{obj}"

# Initialize Yaml general configuration
with open(args.config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
    
# Initialize Detectron2 configuration
cfg = get_cfg()

# Register datasets using the parameters from JSON
register_mosquitoes()

# Access the specific model configuration
model_name = (config['TRAINING']['MODEL_NAME']).upper()
model_config = config['MODELS'][model_name]

# Set model parameters
cfg.merge_from_file(model_config['_BASE_'])
cfg.DATASETS.TRAIN = (args.data_train,)
cfg.DATASETS.VAL = (args.data_val,)
cfg.OUTPUT_DIR = config['TEST']['OUTPUT_DIR']
cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, fold_dir)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
setup_logger(cfg.OUTPUT_DIR)
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
cfg.DATALOADER.NUM_WORKERS = config['TRAINING']['TRAINING_NUM_WORKERS']
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config['TRAINING']['BATCH_SIZE_PER_IMAGE']
cfg.SOLVER.IMS_PER_BATCH = config['TRAINING']['IMAGES_PER_BATCH']
cfg.SOLVER.BASE_LR = config['TRAINING']['LEARNING_RATE']
cfg.SOLVER.WEIGHT_DECAY = config['TRAINING']['WEIGHT_DECAY']
cfg.SOLVER.STEPS = config['TRAINING']['STEPS']
cfg.SOLVER.MAX_ITER = config['TRAINING']['MAX_ITER']
cfg.MIN_DELTA = config['TRAINING'].get('MIN_DELTA')
cfg.PATIENCE = config['TRAINING'].get('PATIENCE') 
cfg.MODE = config['TRAINING'].get('MODE', 'pocket_test')  
cfg.VAL_PERIOD = config['TRAINING'].get('VAL_PERIOD')  
cfg.AUGMENTATION = config['AUGMENTATION'].get('ENABLE')

# Update model weights to the best model found
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, config['TEST']['TEST_WEIGHTS'])

def init_res_dict():
    return {
        'score': [],
        'TP': [],
        'FP': [],
        'FN': [],
        'Pr': [],
        'Rc': [],
        'F1': [],
        'AP50': [],
    }

res = init_res_dict()

scores = np.arange(0.1, 1, 0.02).tolist()

best_score = None
best_f1 = -1

for score in scores:
    print(f'EVALUATION USING SCORE = {score}')

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score

    trainer = DefaultTrainer(cfg)
    #trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    val_loader = build_detection_test_loader(cfg, args.data_val)
    evaluator = COCOEvaluator(args.data_val,
                              cfg,
                              False,
                              output_dir=os.path.join(cfg.OUTPUT_DIR, args.data_val))
    cfn_mat = CfnMat(args.data_val, output_dir=cfg.OUTPUT_DIR)

    results = inference_on_dataset(
        trainer.model,
        val_loader,
        DatasetEvaluators([evaluator, cfn_mat]),
    )

    pr = results['tp'] / (results['tp'] + results['fp'] + 1e-16)
    rc = results['tp'] / (results['tp'] + results['fn'] + 1e-16)
    f1 = (2 * pr * rc) / (pr + rc + 1e-16)

    res['score'].append(score)
    res['TP'].append(results['tp'])
    res['FP'].append(results['fp'])
    res['FN'].append(results['fn'])
    res['AP50'].append(results['bbox']['AP50'])
    res['Pr'].append(pr)
    res['Rc'].append(rc)
    res['F1'].append(f1)

    # Update best score if current score has a higher F1 or if it ties F1 but has a higher score
    if f1 > best_f1 or (f1 == best_f1 and score > best_score):
        best_f1 = f1
        best_score = score

# Create DataFrame from results
df = pd.DataFrame(res)

# Filter results for the best score
best_results = df[df['score'] == best_score]

# Save filtered results to CSV
save_results_dir = os.path.dirname(cfg.MODEL.WEIGHTS)
name_base = f'{obj}_{model_name}_{args.data_val.split("_")[-2]}'
best_results.to_csv(os.path.join(save_results_dir, name_base + '.csv'), index=False)

print(f"Results saved for the best score (F1 = {best_f1}, score = {best_score})")
