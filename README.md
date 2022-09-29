## Segmentation of Multiple Scelerosis Lesions across Hospitals: Learn Continually or Train from Scratch?

This is the official PyTorch-based repository containing the code and instructions for reproducing the results of the above-mentioned paper. 

### Introduction

This work presents a case of for using continual learning for the segmentation of Multiple Sclerosis (MS) lesions across multi-center data. In particular, the problem is formalized as domain-incremental learning and uses experience replay, a well-known continual learning method, for MS lesion segmentation across eight different hospitals/centers. As shown in the figure, four types of experiments are performed: _single-domain_, _multi-domain_, _sequential fine-tuning_, and _experience replay_. Our results show that replay performs better than fine-tuning as more data arrive and also achieves _positive backward transfer_, both in terms of the Dice score on a held-out test set. More importantly, replay also outperforms multi-domain (IID) training, hence suggesting that lifelong learning is a promising long-term solution for improving automated segmenttation of MS lesions compared to training from scratch. 
 

### Structure of the Repository

1. `main_pl_*.py`: These files contain the main code for the four types of experiments, each having a separate file. 
2. `train.sh`: Contains the bash script for calling one of the `main_pl_*.py` files to train the model across multiple seeds.
3. `utils/`: Contains 3 files

    a. `create_json_data.py`: Creates a `json` file (in the Decathlon format) for each center based on the defined train/test split

    b. `generate_json.sh`: Bash script for generate json files mentioned above.

    c. `metrics.py`: Contains the implementations of some continual learning metrics. 
4. `plots/`: Contains code for creating the plots described in the paper. 


### Getting Started

The code uses the following main packages - `torch`, `pytorch-lightning`, `monai`, `wandb`, and `ivadomed`. It is tested only on a Linux environment with Python 3.8. The first step is to clone the repository:

> git clone https://github.com/naga-karthik/continual-learning-ms

Then, 
```
cd continual-learning-ms/
conda create -n venv_cl_ms python=3.8
conda activate venv_cl_ms
pip install -r requirements.txt
```
