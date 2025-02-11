import argparse
import wandb

from configs import sweep_config
from training.epochs import training_epoch, validation_epoch
from training.factory import data_factory, training_factory

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        data_objects = data_factory(config=config)
        training_objects = training_factory(config=config, objects=data_objects)

        # Train the model
        for epoch in range(config["epochs"]):
            training_loss = training_epoch(data_objects=data_objects, training_objects=training_objects)
            wandb.log({"training_loss": training_loss, "epoch": epoch})

            validation_loss = validation_epoch(data_objects=data_objects, training_objects=training_objects)
            wandb.log({"validation_loss": validation_loss, "epoch": epoch})

def main(config):
    wandb.login()
    sweep_id = wandb.sweep(config["sweep"], project=config["project"])
    wandb.agent(sweep_id, train, count=config["count"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_project', type=str, default='SimCLR-Roboflow')
    parser.add_argument('--count', type=int, default=5)
    args = parser.parse_args()

    sweep_config = sweep_config.get_sweep_config()
    config = {
        "sweep": sweep_config,
        "project": args.wandb_project,
        "count": args.count
    }

    main(config)