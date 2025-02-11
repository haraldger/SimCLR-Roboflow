import argparse
import torch

from training.epochs import training_epoch, validation_epoch
from training.factory import data_factory, training_factory

def main(config):
    data_objects = data_factory(config=config)
    training_objects = training_factory(config=config, objects=data_objects)

    # Train the model
    best_loss = float('inf')
    for epoch in range(config["epochs"]):
        print(f'Epoch {epoch}/{config["epochs"]}')

        # Train the model
        training_loss = training_epoch(data_objects=data_objects, training_objects=training_objects)
        print(f'Training Loss: {training_loss}')

        # Evaluate the model
        validation_loss = validation_epoch(data_objects=data_objects, training_objects=training_objects)
        print(f'Validation Loss: {validation_loss}')

        # Save the model if validation loss is the best we've seen so far
        if validation_loss < best_loss:
            print('Validation loss decreased. Saving model...')
            best_loss = validation_loss

            model = training_objects["model"]
            torch.save(model.state_dict(), config["output_path"])

    print(f'Training complete. Model saved to {config["output_path"]}')



if __name__ == "__main__":
    config = dict()
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model_size', type=int, default=18, choices=[18, 34, 50, 101, 152])
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--learning_rate', type=float, default=0.01875)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--output_path', type=str, default='model.pt')
    args = parser.parse_args()

    config['epochs'] = args.epochs
    config['model_size'] = args.model_size
    config['batch_size'] = args.batch_size
    config['optimizer'] = args.optimizer
    config['learning_rate'] = args.learning_rate
    config['weight_decay'] = args.weight_decay
    config['tau'] = args.tau
    config['output_path'] = args.output_path
    
    main(config)