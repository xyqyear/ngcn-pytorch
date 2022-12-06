import torch
import wandb

from torch_geometric.datasets import Planetoid
from model import NGCN

from utils import pyg_data_to_sparse_adj


def train(
    model,
    data,
    device,
    optimizer,
    criterion,
    epochs,
    *,
    regularize_attention_weights=True,
    use_wandb=True,
    scheduler=None,
):
    model = model.to(device)
    adj = pyg_data_to_sparse_adj(data).to(device)
    data = data.to(device)

    wsum_flag = model.output_layer == "wsum"

    model.train()
    highest_val_acc = 0
    for _ in range(epochs):
        optimizer.zero_grad()
        out, cell_out = model(data.x, adj)
        num_cells = cell_out.shape[1]
        num_classes = cell_out.shape[2]
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss += (
            criterion(
                #    [N, replicas * (max_hops + 1), num_classes]
                # => [replicas * (max_hops + 1), N, num_classes]
                # => [replicas * (max_hops + 1) * N, num_classes]
                cell_out[data.train_mask].permute(1, 0, 2).reshape(-1, num_classes),
                #    [N]
                # => [replicas * (max_hops + 1) * N]
                data.y[data.train_mask].repeat(num_cells),
            )
            # we times num_cells because in the above loss, we averaged over all cells and outputs
            # if we want to get the sum of each cell's loss, we need to times num_cells
            * num_cells
        )
        if wsum_flag and regularize_attention_weights:
            loss += torch.mean(model.weights.view(-1) ** 2) * 1e-3
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        val_acc = val(model, data, device)
        if val_acc > highest_val_acc:
            highest_val_acc = val_acc

        if use_wandb:
            wandb.log(
                {
                    "loss": loss.item(),
                    "val_acc": val_acc,
                    "highest_val_acc": highest_val_acc,
                }
            )


def test(model, data, device):
    return test_acc(model, data, data.test_mask, device)


def val(model, data, device):
    return test_acc(model, data, data.val_mask, device)


def test_acc(model, data, y_mask, device):
    model = model.to(device)
    adj = pyg_data_to_sparse_adj(data).to(device)
    data = data.to(device)

    model.eval()
    pred = model(data.x, adj)[0].argmax(dim=1)
    correct = (pred[y_mask] == data.y[y_mask]).sum()
    acc = int(correct) / int(y_mask.sum())
    return acc


def main():
    cora_dataset = Planetoid(
        root=r"D:\sync\machine_learning\datasets\planetoid\Cora",
        name="Cora",
    )

    wandb.init(project="ngcn-pytorch", entity="xyqyear")

    if wandb.config.keys():
        hyperparams = wandb.config.as_dict()
    else:
        hyperparams = {
            "optimizer": "Adam",
            "lr": 0.01,
            "epochs": 300,
            "max_hops": 3,
            "replicas": 3,
            "gcn_hidden_dim": 10,
            "output_layer": "concat",
            "input_dropout": 0.8,
            "layer_dropout": 0.4,
        }
        wandb.config.update(hyperparams)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cora_data = cora_dataset[0]
    model = NGCN(
        input_dim=cora_data.x.shape[1],
        output_dim=cora_dataset.num_classes,
        max_hops=hyperparams["max_hops"],
        replicas=hyperparams["replicas"],
        gcn_hidden_dim=hyperparams["gcn_hidden_dim"],
        output_layer=hyperparams["output_layer"],
        input_dropout=hyperparams["input_dropout"],
        layer_dropout=hyperparams["layer_dropout"],
    )
    scheduler = None
    match hyperparams["optimizer"]:
        case "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])
        case "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=hyperparams["lr"])
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=40, gamma=0.5
            )
        case _:
            raise ValueError(
                f"optimizer {hyperparams['optimizer']} not supported for now"
            )
    criterion = torch.nn.CrossEntropyLoss()

    train(
        model,
        cora_data,
        device,
        optimizer,
        criterion,
        hyperparams["epochs"],
        regularize_attention_weights=True,
        use_wandb=True,
        scheduler=scheduler,
    )


if __name__ == "__main__":
    main()
