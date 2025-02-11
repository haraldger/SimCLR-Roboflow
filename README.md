## Setup

Start by cloning or forking this repo. It is recommended to create a virtual environment to execute the code in this repo. Do so by running the command `python -m venv .venv` and activate the environment with `source .venv/bin/activate`. Next, install the required dependencies by executing the following commands:

```
pip install --upgrade pip
pip install -r requirements.txt
```

## Project structure

**main.py**: This program will perform the SimCLR pre-training on the Roboflow Firefighter dataset. To run the program, execute the following: `python main.py`. These are the optional arguments, e.g., training hyperparameters:
- `--epochs`
- `--model_size`
- `--batch_size`
- `--optimizer`
- `--learning_rate`
- `--weight_decay`
- `--tau`
- `--output_path`

**hp_sweep.py**: This program provides infrastructure and execution for performing hyperparameter sweeps. The Weights & Biases framework is used to log and visualize each run in the sweep. The user must provide their own api key, and will be prompted to do so in the console when running the script. The following optional command line arguments can be provided to the program:

- `wandb_project`
- `count` - use this to specify the total number of runs in the sweep.

Additional configs are found in the file `configs/sweep_config.py`.

**data/**: This directory contains the data files retrieved from Roboflow. The data is organized into directories for `train/`, `valid/` and `test/`.

**models/**: Directory for model architectures. The file `CLRNet.py` contains the network definition for a ResNet-based backbone with a SimCLR head.

**training/**: Directory containing utils and training loop definitions.

- `training/epochs.py`: defines the training and validation epochs.
- `training/factory.py`: factory functions to instantiate required execution objects.
- `training/utils.py`: other utility functions.

**configs/**: Directory containing config files, provided as Python dictionaries.

## Limitations

SimCLR, and contrastive learning approaches more broadly, benefit greatly from larger batch sizes as outlined in <a href="https://arxiv.org/abs/2002.05709" target="_blank">the original SimCLR paper</a>. Availability of computational resources is as such one of the most pressing limiting factors, where processors with large memory and powerful GPUs are required to efficiently train with high volume. The particular dataset is small, with only approximately 150 data samples compared to the default batch size of 4096 in the original implementation of the authors. This may ultimately hinder the success of the SimCLR pre-training regimen for the Firefighter symbols of this dataset.

## Continuation

With proper time and resources, thorough hyperparameter sweeps would be performed to find the optimal value of training hyperparameters as well as an appropriate model size. Infrastructure to do this has been provided in the script `hp_sweep.py`, and it is recommended that this be run in the cloud using compute clusters with a GPU and processors of adequate memory size. 

Additionally, a proper evaluation of learnt representation would be performed. This is typically done through the method of Linear Evaluation, whereby the base encoder is frozen and the projection head is replaced with a simple linear classification head. The model is then trained on the downstream task (classification or, more appropriately in this repository, object detection), and the test accuracy of the model serves as a proxy evaluation of the SimCLR representations.

Alternatively, for the purpose of this object detection dataset, the pre-trained SimCLR model can be finetuned on the same dataset. Though this deviates from the typical paradigm of pre-training on larger datasets followed by finetuning on smaller, specialized datasets, and therefore does not enjoy the same benefits of large volume pre-training to learn intelligent representations, the more flexible and dynamic nature of finetuning (rather than the frozen linear evaluation) is likely to result in better test performance on the downstream object detection task. 

Finally, a more realistic real-life scenario is one where a new schematic is given, containing symbols that are not shared with the dataset in this repository and therefore would never have been seen during training. A proposed approach would be to use the provided Firefighter dataset only for SimCLR pretraining to efficiently learn representations of schematics with symbols, followed by downstream training with some zero-shot learning of one-shot learning approach. As an initial approach, <a href="https://arxiv.org/abs/1810.09502">MAML++</a> could be experimented with, a meta-learning training method that learns base model parameters (model instantiations) that quickly adapt to new and unseen data given only one or few training samples. 

Note: for those who want to perform classification on the symbols in the dataset rather than object detection (i.e., classification + localization), some data extraction would have to be performed to retrieve the bounding boxes with symbols using the provided labels. Classification is, however, beyond the scope of this project.