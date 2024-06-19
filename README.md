![MBGd](https://github.com/misabellerv/Mosquitoes/assets/88784791/d784e854-210e-4855-a25c-59c5018b9fa7)



## About the Project
In progress...

## Repo structure
In progress...

## Installation

### Requirements
- CUDA 11.8+
- Python 3.8+
- Conda
- Linux system to run bash script files

Check if you have the necessary requirements and run the following command:
```bash
conda env create -f requirements.yml
```
You can change the environment name in requirements.yml. Activate the environment:
```bash
conda activate <env>
```
Now let's install proper Pytorch versions with GPU:
```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
```
And detectron2:
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
Install some more libraries:
```bash
pip install nni scikit-image
```
Check if your detectron2 version is 0.6+. 

## MBGd dataset
In progress...

## Research Methodology
In progress...

## Training and Validation Split
In progress...

## Results
In progress...

## Configurations File
In `configs/config.yaml` you can find the configurations file. This is a general YAML that aims to set all the workflow parameters in a single file, so you can always easly modify the settings.

In more general words, this file allows you to store multiple model's configurations at the same time and configs from register datasets, training, test, augmentation steps.

Let's discuss what each of these flags mean.

- `MODELS`: stores detectron2 models. You can have multiple models inside `MODELS`. The models' format must match detectron2 default YAML file (such as identation levels and parameters names).
  - `FASTER_RCNN_R_50_FPN_1X`: The main model used to this research.
    - `_BASE_`: Base default YAML path. You can either download it to your local machine or use detectron2 link as in my config file.
    - `MODEL`: Sets architecture parameters for this specific model.
      - `WEIGHTS`: Path to the pre-trained weights. You can either download it to your machine or use detectron2 link as in my config file.
      - `MASK_ON`: Boolean indicating whether to enable mask prediction (True) or not (False).
      - `RPN`: Parameters for Region Proposal Network (RPN).
        - `PRE_NMS_TOPK_TRAIN`: Number of top scoring RPN proposals to keep before NMS during training.
        - `PRE_NMS_TOPK_TEST`: Number of top scoring RPN proposals to keep before NMS during testing.
        - `POST_NMS_TOPK_TRAIN`: Number of top scoring RPN proposals to keep after NMS during training.
        - `POST_NMS_TOPK_TEST`: Number of top scoring RPN proposals to keep after NMS during testing.
        - `NMS_THRESH`: IoU threshold for non-maximum suppression (NMS) of RPN proposals.
      - `RESNETS`: ResNet architecture configuration.
        - `DEPTH`: Depth of the ResNet backbone (e.g., 50 for ResNet-50).
      - `ROI_BOX_HEAD`: Region of Interest (ROI) box head configuration.
        - `POOLER_SAMPLING_RATIO`: Ratio of the input height/width to the output height/width when using ROI align or ROI pool.
        - `POOLER_RESOLUTION`: Output resolution of the ROI align or ROI pooler.
      - `BACKBONE`: Backbone configuration.
        - `FREEZE_AT`: Number of stages of the backbone network to freeze during training.
      - `ROI_HEADS`: ROI heads configuration.
        - `NUM_CLASSES`: Number of classes for object detection (e.g., 1 for single class detection).
  - `REGISTER_DATASETS`: Configuration for registering datasets used in the project.
    - `JSON_PATH`: Path to the directory containing COCO JSON files used for dataset registration.
    - `FRAMES_PATH`: Path to the directory containing frame files used for dataset registration.
    - `FOLDS`: Number of folds for cross-validation or dataset splitting.
    - `OBJECTS`: List of objects/classes to be registered (e.g., ["tire"]).
  - `TRAINING`: Configuration for training the model.
    - `VAL_PERIOD`: Validation period (in epochs) to perform validation during training.
    - `TRAINING_NUM_WORKERS`: Number of worker processes to use for data loading during training.
    - `BATCH_SIZE_PER_IMAGE`: Batch size per image (total batch size will be IMAGES_PER_BATCH * BATCH_SIZE_PER_IMAGE).
    - `IMAGES_PER_BATCH`: Number of images per batch.
    - `PATIENCE`: Patience parameter for early stopping during training.
    - `MAX_ITER`: Maximum number of training iterations.
    - `MIN_DELTA`: Minimum change in the monitored quantity to qualify as an improvement for early stopping.
    - `LEARNING_RATE`: Initial learning rate for training.
    - `WEIGHT_DECAY`: Weight decay (L2 penalty) to apply on model parameters.
    - `STEPS`: Optional list indicating steps for learning rate scheduling (e.g., [1000, 2000]).
    - `CUDA_DEVICE`: CUDA device index to use for training (e.g., 0 for GPU training).
    - `MODEL_NAME`: Name of the model being trained (e.g., 'faster_rcnn_R_50_FPN_1x').
    - `OBJECT`: Object of interest for the training dataset (e.g., 'tire').
  - `TEST`: Configuration for testing the model.
    - `TEST_WEIGHTS`: Path to the saved model weights used for testing.
    - `OUTPUT_DIR`: Directory path for saving test outputs (e.g., predictions, evaluation results). Recommended: `outputs/model_name/experiment_name`. 
    - `FILTER_EMPTY_ANNOTATIONS`: Boolean indicating whether to use filtering of empty annotations. If set `True`, your validation set will not take into account frames/images that have no objects of interest (empty annotations). If set `False`, the validation set uses all frames/images in the workflow.
  - `AUGMENTATION`: Configuration for data augmentation.
    - `ENABLE`: Boolean indicating whether data augmentation is enabled (True) or not (False).

## Running the Project

### Training and Evaluation:
Go to `configs/config.yaml` and change `JSON_PATH` and `FRAMES_PATH` paths to your local paths. 

They must point to where your COCO JSON and data frames are located.
If you want to change parameters and run a more customized model, check **Documentation** (still in progress).

```bash
nohup bash training_mosquitoes.sh > train+test.log 2>&1 &
```
### Loss curves
You can acess model's loss curves by running `notebooks/stop_criterion.ipynb`. Please, don't forge to change system path to your local path:
```python
import sys
sys.path.append(<your_path_here>)
from codes.stop_criterion import plot_loss
...
```
To run a single fold, just run:
```python
import sys
sys.path.append(<your_path_here>)
from codes.stop_criterion import plot_fold

root = <output_folder_path> 
model = <model_name> # string 
obj = <object> # string
fold = <fold_number> # int

plot_fold(root, model, obj, fold)
```
If you want to run 2 or more folds, you can do:
```python
import sys
sys.path.append(<your_path_here>)
from codes.stop_criterion import plot_loss

root = <output_folder_path> 
model = <model_name> # string 
obj = <object> # string
fold = <number_of_folds> # 2 or more

plot_fold(root, model, obj, fold)
```

