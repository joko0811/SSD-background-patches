import torch
from torch.optim import Optimizer


class MI_FGSM(Optimizer):
    """Momentum Iterative Fast Gradient Sign Method"""

    def __init__(self, params, lr=0.1, momentum=0.1, use_pre_backward=False) -> None:
        """
        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
            lr (float, optional): learning rate (default: 0.1)
            momentum (float, optional): momentum factor (default: 0.1)
        """
        self.use_pre_backward = use_pre_backward

        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < momentum:
            raise ValueError("Invalid momentum factor: {}".format(momentum))

        defaults = {"lr": lr, "momentum": momentum}
        super().__init__(params, defaults)

        self.state = dict()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p] = dict()
                self.state[p]["g"] = torch.zeros_like(p.data)
                self.state[p]["lr"] = group["lr"]
                self.state[p]["momentum"] = group["momentum"]

    def step(self, closure=None) -> None:
        # x_nes = p.data + learning_rate * momentum * self.state[p]["g"]

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                if p not in self.state:
                    self.state[p] = {
                        "g": torch.zeros_like(p),
                        "lr": group["lr"],
                        "momentum": group["momentum"],
                    }

                learning_rate = group["lr"]
                momentum = group["momentum"]

                g = momentum * self.state[p]["g"] + (p.grad / torch.norm(p.grad, p=1))

                p.data = torch.clip(p + learning_rate * torch.sign(g), min=0, max=255)

                self.state[p] = dict()
                self.state[p]["g"] = g
                self.state[p]["lr"] = learning_rate
                self.state[p]["momentum"] = momentum

    def pre_backward(self, params: list) -> None:
        """Nesterov Iterative Fast Gradient Sign Method
        Use with MI-FGSM
        Call before backward
        Args:
            params (iterable): iterable of parameters to optimize or dicts defining //
                Note - The contents and order of the list must be the same as at initialisation.
            state (dict): Optimizer state returned from MI_FGSM.step()
        """
        if not self.use_pre_backward:
            return

        for p in params:
            if p not in self.state:
                continue
            g = self.state[p]["g"]
            learning_rate = self.state[p]["lr"]
            momentum = self.state[p]["momentum"]
            p.data = p + learning_rate * momentum * g
