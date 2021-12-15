import torch
from bites.utils.Simple_Network import *
from geomloss import SamplesLoss
from torch import Tensor
from torch import nn


class CFRNet(nn.Module):
    """Network structure similar to the DeepHit paper, but without the residual
    connections (for simplicity).
    """

    def __init__(self, in_features, num_nodes_shared, num_nodes_indiv, out_features,
                 num_treatments=2, batch_norm=True, dropout=None):
        super().__init__()
        self.shared_net = MLPVanilla(
            in_features, num_nodes_shared[:-1], num_nodes_shared[-1],
            batch_norm, dropout,
        )
        self.risk_nets = torch.nn.ModuleList()
        for _ in range(2):
            net = MLPVanilla(
                num_nodes_shared[-1], num_nodes_indiv, out_features,
                batch_norm, dropout,output_bias=False)
            self.risk_nets.append(net)

        self.baseline_hazards_ = []
        self.baseline_cumulative_hazards_ = []

    def forward(self, input, treatment):
        # treatment=input[:,-1]
        N = treatment.shape[0]
        y = torch.zeros_like(treatment)

        out = self.shared_net(input)

        out0 = out[treatment == 0]
        out1 = out[treatment == 1]

        out0 = self.risk_nets[0](out0)
        out1 = self.risk_nets[1](out1)

        k, j = 0, 0
        for i in range(N):
            if treatment[i] == 0:
                y[i] = out0[k]
                k = k + 1
            else:
                y[i] = out1[j]
                j = j + 1

        return y, out

    def predict(self, X, treatment):
        self.eval()
        out = self(X, treatment)
        self.train()
        return out[0], out[1]

    def predict_numpy(self, X, treatment):
        self.eval()
        X = torch.Tensor(X)
        treatment = torch.Tensor(treatment)
        out = self(X, treatment)
        self.train()
        return out[0].detach(), out[1].detach()


class CFRNet_Loss(torch.nn.Module):
    """Loss function for the Tar net structure
    uses the same Cox Loss as PyCox but separates between different treatment classes
    """

    def __init__(self, alpha=0, blur=0.05):
        self.alpha = alpha
        self.blur = blur
        super().__init__()

    def forward(self, log_h: Tensor,out:Tensor, durations: Tensor, treatments: Tensor) -> Tensor:
        mask0 = treatments == 0
        mask1 = treatments == 1

        mse=nn.MSELoss()

        loss_t0 = mse(log_h[mask0], durations[mask0])
        loss_t1 = mse(log_h[mask1], durations[mask1])

        """Imbalance loss"""
        # p=torch.sum(mask1)/(torch.sum(mask0)+torch.sum(mask0))
        # sig = torch.tensor(self.sig)
        p = torch.sum(mask1) / treatments.shape[0]
        # imbalace_loss = mmd2_rbf(out, treatments, p, sig)
        # imbalace_loss = mmd2_lin(out, treatments, p)

        if self.alpha == 0.0:
            imbalance_loss = 0.0
        else:
            samples_loss = SamplesLoss(loss="sinkhorn", p=2, blur=self.blur, backend="tensorized")
            phi0 = out[mask0]
            phi1 = out[mask1]
            imbalance_loss = samples_loss(phi1, phi0)

        return (1.0 - p) * loss_t0 + p * loss_t1 + self.alpha * imbalance_loss