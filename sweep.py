import wandb

from train import main


if __name__ == "__main__":
    sweep_configuration = {
        "method": "random",
        "name": "sweep01",
        "description": "Sweep over dropout rates",
        "metric": {"goal": "maximize", "name": "highest_val_acc"},
        "parameters": {
            "optimizer": {"value": "Adam"},
            "lr": {"value": 0.01},
            "epochs": {"value": 400},
            "max_hops": {"value": 3},
            "replicas": {"value": 3},
            "gcn_hidden_dim": {"value": 10},
            "output_layer": {"value": "concat"},
            "input_dropout": {"values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
            "layer_dropout": {"values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
        },
        "early_terminate": {"type": "hyperband", "min_iter": 20, "eta": 2},
    }

    sweep_id = wandb.sweep(sweep_configuration, project="ngcn-pytorch", entity="xyqyear")
    wandb.agent(sweep_id, main, count=200)
