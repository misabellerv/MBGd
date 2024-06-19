import os
from itertools import product
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
import numpy as np
import yaml

# Load the general YAML configuration
with open('./configs/config.yaml', 'r') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
    
FOLDS = config['REGISTER_DATASETS']['FOLDS']
JSON_PATH = config['REGISTER_DATASETS']['JSON_PATH']
FRAMES_PATH = config['REGISTER_DATASETS']['FRAMES_PATH']
OBJECTS = config['REGISTER_DATASETS']['OBJECTS']

def register_datasets(dataset_names, json_dir, img_root):

    for d in dataset_names:
        register_coco_instances(f"mbg_{d.lower()}",{},os.path.join(json_dir, f"coco_format_{d}.json"),img_root,)
        cdc_metadata = MetadataCatalog.get(f"mbg_{d.lower()}")
        
def register_mosquitoes():

    sets = [f"train{n}" for n in np.arange(FOLDS)]
    sets += [f"val{n}" for n in np.arange(FOLDS)]

    comb = list(product(sets, OBJECTS))

    sets = ["_".join(c) for c in comb]

    register_datasets(sets, JSON_PATH, FRAMES_PATH)

if __name__ == "__main__":
    register_mosquitoes()