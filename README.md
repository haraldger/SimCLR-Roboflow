## Setup

It is recommended to create a virtual environment to execute the code in this repo. Do so by running the command `python -m venv .venv` and activate the environment with `source .venv/bin/activate`. Next, install the required dependencies by executing the following commands:

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

**data/**: This directory contains the data files retrieved from Roboflow.

**models/**: Directory for model architectures. The file `CLRNet.py` contains the network definition for a ResNet-based backbone with a SimCLR head.

**training/**: Directory containing utils and training loop definitions.

## Limitations

SimCLR, and contrastive learning approaches more broadly, benefit greatly from larger batch sizes as outlined in <a href="https://arxiv.org/abs/2002.05709" target="_blank">the original SimCLR paper</a>. Availability of computational resources is as such one of the most pressing limiting factors, where processors with large memory and powerful GPUs are required to efficiently train with high volume. The particular dataset is small, with only approximately 150 data samples compared to the default batch size of 4096 in the original implementation of the authors. This may ultimately hinder the success of the SimCLR pre-training regimen for the Firefighter symbols of this dataset.

## Continuation

With proper time and resources, thorough hyperparameter sweeps would be performed to find the optimal value of training hyperparameters as well as an appropriate model size. Infrastructure to do this has been provided in the script `hp_sweep.py`, and it is recommended 