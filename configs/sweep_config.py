method = "random"
metric = {
    "name": "validation_loss",
    "goal": "minimize",
}
parameters = {
    "model_size": {
        "values": [18, 34, 50, 101, 152]
    },
    "batch_size": {
        "values": [2, 4, 8, 16, 32, 64]
    },
    "optimizer": {
        "values": ["adam", "sgd"]
    },
    "learning_rate": {
        "distribution": "uniform",
        "max": 0.1,
        "min": 0.0001
    },
    "weight_decay": {
        "distribution": "uniform",
        "max": 0.1,
        "min": 1e-6
    },
    "tau": {
        "values": [0.05, 0.1, 0.2]
    },
    "epochs": {
        "value": 10
    },
}

def get_sweep_config():
    return {
        "method": method,
        "metric": metric,
        "parameters": parameters
    }