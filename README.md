# Mosquitoes Breeding Grounds Detector 🦟🔎

## Installation

### Requirements
- CUDA 11.8 or higher
- Python 3.8+
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

## Running the project

### Training + Validation:
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
sys.path.append(YOUR_PATH_HERE)
from codes.stop_criterion import plot_loss
...
```


