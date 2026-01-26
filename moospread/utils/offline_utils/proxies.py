"""
Adapted from: https://github.com/lamda-bbo/offline-moo/blob/main/off_moo_baselines/multiple_models/nets.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

def offdata_get_dataloader(
    X,
    y,
    train_ratio: float = 0.9,
    batch_size: int = 32,
):

    tensor_dataset = TensorDataset(X, y)
    lengths = [
        int(train_ratio * len(tensor_dataset)),
        len(tensor_dataset) - int(train_ratio * len(tensor_dataset)),
    ]
    train_dataset, val_dataset = random_split(tensor_dataset, lengths)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 4, shuffle=False, drop_last=False
    )

    return train_loader, val_loader

def spearman_correlation(x, y):
    n = x.size(0)
    _, rank_x = x.sort(0)
    _, rank_y = y.sort(0)

    d = rank_x - rank_y
    d_squared_sum = (d**2).sum(0).float()

    rho = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
    return rho

class SingleModelBaseTrainer(nn.Module):

    def __init__(self, model, which_obj, args):
        super(SingleModelBaseTrainer, self).__init__()
        self.args = args

        self.forward_lr = args["proxies_lr"]
        self.forward_lr_decay = args["proxies_lr_decay"]
        self.n_epochs = args["proxies_epochs"]
        self.device = args["device"]
        self.verbose = args["verbose"]

        self.model = model

        self.which_obj = which_obj

        self.forward_opt = Adam(model.parameters(), lr=args["proxies_lr"])
        self.train_criterion = lambda yhat, y: (
            torch.sum(torch.mean((yhat - y) ** 2, dim=1))
        )
        self.mse_criterion = nn.MSELoss()

    def _evaluate_performance(self, statistics, epoch, train_loader, val_loader):
        self.model.eval()
        with torch.no_grad():
            y_all = torch.zeros((0, self.n_obj)).to(self.device)
            outputs_all = torch.zeros((0, self.n_obj)).to(self.device)
            for (
                batch_x,
                batch_y,
            ) in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                y_all = torch.cat((y_all, batch_y), dim=0)
                outputs = self.model(batch_x)
                outputs_all = torch.cat((outputs_all, outputs), dim=0)

            train_mse = self.mse_criterion(outputs_all, y_all)
            train_corr = spearman_correlation(outputs_all, y_all)
            train_pcc = self.compute_pcc(outputs_all, y_all)

            statistics[f"model_{self.which_obj}/train/mse"] = train_mse.item()
            for i in range(self.n_obj):
                statistics[f"model_{self.which_obj}/train/rank_corr_{i + 1}"] = (
                    train_corr[i].item()
                )
            # if self.verbose:
            #     print(
            #         "Epoch [{}/{}], MSE: {:}, PCC: {:}".format(
            #             epoch + 1, self.n_epochs, train_mse.item(), train_pcc.item()
            #         )
            #     )

        with torch.no_grad():
            y_all = torch.zeros((0, self.n_obj)).to(self.device)
            outputs_all = torch.zeros((0, self.n_obj)).to(self.device)

            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                y_all = torch.cat((y_all, batch_y), dim=0)
                outputs = self.model(batch_x)
                outputs_all = torch.cat((outputs_all, outputs))

            val_mse = self.mse_criterion(outputs_all, y_all)
            val_corr = spearman_correlation(outputs_all, y_all)
            val_pcc = self.compute_pcc(outputs_all, y_all)

            statistics[f"model_{self.which_obj}/valid/mse"] = val_mse.item()
            for i in range(self.n_obj):
                statistics[f"model_{self.which_obj}/valid/rank_corr_{i + 1}"] = (
                    val_corr[i].item()
                )

            # if self.verbose:
            #     print(
            #         "Valid MSE: {:}, Valid PCC: {:}".format(val_mse.item(), val_pcc.item())
            #     )
    
            if val_pcc.item() > self.min_pcc:
                # if self.verbose:
                #     print("🌸 New best epoch! 🌸")
                self.min_pcc = val_pcc.item()
                self.model.save(val_pcc=self.min_pcc)
        return statistics

    def launch(
        self,
        train_loader=None,
        val_loader=None,
        retrain_model: bool = True,
    ):

        def update_lr(optimizer, lr):
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        if not retrain_model and os.path.exists(self.model.save_path):
            self.model.load()
            return

        assert train_loader is not None
        assert val_loader is not None

        self.n_obj = None
        self.min_pcc = -1.0
        statistics = {}

        with tqdm(
            total=self.n_epochs,
            desc=f"Proxy Training",
            unit="epoch",
        ) as pbar:
            
            for epoch in range(self.n_epochs):
                self.model.train()

                losses = []
                epoch_loss = 0.0
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    if self.n_obj is None:
                        self.n_obj = batch_y.shape[1]

                    self.forward_opt.zero_grad()
                    outputs = self.model(batch_x)
                    loss = self.train_criterion(outputs, batch_y)
                    losses.append(loss.item() / batch_x.size(0))
                    loss.backward()
                    self.forward_opt.step()
                    epoch_loss += loss.item()

                statistics[f"model_{self.which_obj}/train/loss/mean"] = np.array(
                    losses
                ).mean()
                statistics[f"model_{self.which_obj}/train/loss/std"] = np.array(
                    losses
                ).std()
                statistics[f"model_{self.which_obj}/train/loss/max"] = np.array(
                    losses
                ).max()

                self._evaluate_performance(statistics, epoch, train_loader, val_loader)

                statistics[f"model_{self.which_obj}/train/lr"] = self.forward_lr
                self.forward_lr *= self.forward_lr_decay
                update_lr(self.forward_opt, self.forward_lr)

                epoch_loss /= len(train_loader)
                pbar.set_postfix({
                        "Loss": epoch_loss,
                        })
                pbar.update(1)

    def compute_pcc(self, valid_preds, valid_labels):
        vx = valid_preds - torch.mean(valid_preds)
        vy = valid_labels - torch.mean(valid_labels)
        pcc = torch.sum(vx * vy) / (
            torch.sqrt(torch.sum(vx**2) + 1e-12) * torch.sqrt(torch.sum(vy**2) + 1e-12)
        )
        return pcc


class MultipleModels(nn.Module):
    def __init__(
        self, n_dim, n_obj, hidden_size, train_mode, device,
        save_dir=None, save_prefix=None
    ):
        super(MultipleModels, self).__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.device = device

        self.obj2model = {}
        self.hidden_size = hidden_size
        self.train_mode = train_mode

        self.save_dir = save_dir
        self.save_prefix = save_prefix
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        for obj in range(self.n_obj):
            self.create_models(obj)

    def create_models(self, learning_obj):
        model = SingleModel
        new_model = model(
            self.n_dim,
            self.hidden_size,
            which_obj=learning_obj,
            device=self.device,
            save_dir=self.save_dir,
            save_prefix=self.save_prefix,
        )
        self.obj2model[learning_obj] = new_model

    def set_kwargs(self, device=None, dtype=torch.float32):
        for model in self.obj2model.values():
            model.set_kwargs(device=device, dtype=dtype)
            model.to(device=device, dtype=dtype)

    def forward(self, x, forward_objs=None):
        if forward_objs is None:
            forward_objs = list(self.obj2model.keys())
        x = [self.obj2model[obj](x) for obj in forward_objs]
        x = torch.cat(x, dim=1)
        return x


activate_functions = [nn.LeakyReLU(), nn.LeakyReLU()]


class SingleModel(nn.Module):
    def __init__(
        self, input_size, hidden_size, which_obj, device,
        save_dir=None, save_prefix=None
    ):
        super(SingleModel, self).__init__()
        self.n_dim = input_size
        self.n_obj = 1
        self.which_obj = which_obj
        self.activate_functions = activate_functions
        self.device = device

        layers = []
        layers.append(nn.Linear(input_size, hidden_size[0]))
        for i in range(len(hidden_size) - 1):
            layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
        layers.append(nn.Linear(hidden_size[len(hidden_size) - 1], 1))

        self.layers = nn.Sequential(*layers)
        self.hidden_size = hidden_size

        self.save_path = os.path.join(save_dir, f"{save_prefix}-{which_obj}.pt")

    def forward(self, x):
        for i in range(len(self.hidden_size)):
            x = self.layers[i](x)
            x = self.activate_functions[i](x)

        x = self.layers[len(self.hidden_size)](x)
        out = x

        return out

    def set_kwargs(self, device=None, dtype=torch.float32):
        self.to(device=device, dtype=dtype)

    def check_model_path_exist(self, save_path=None):
        assert (
            self.save_path is not None or save_path is not None
        ), "save path should be specified"
        if save_path is None:
            save_path = self.save_path
        return os.path.exists(save_path)

    def save(self, val_pcc=None, save_path=None):
        assert (
            self.save_path is not None or save_path is not None
        ), "save path should be specified"
        if save_path is None:
            save_path = self.save_path

        self = self.to("cpu")
        checkpoint = {
            "model_state_dict": self.state_dict(),
        }
        if val_pcc is not None:
            checkpoint["valid_pcc"] = val_pcc

        torch.save(checkpoint, save_path)
        self = self.to(self.device)

    def load(self, save_path=None):
        assert (
            self.save_path is not None or save_path is not None
        ), "save path should be specified"
        if save_path is None:
            save_path = self.save_path

        checkpoint = torch.load(save_path, weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])
        valid_pcc = checkpoint["valid_pcc"]
        print(
            f"Successfully load trained proxy model from {save_path} "
            f"with valid PCC = {valid_pcc}"
        )