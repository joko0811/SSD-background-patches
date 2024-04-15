import torch
import numpy as np

import hydra
from omegaconf import DictConfig
from tensorboardX import SummaryWriter

from detection.detection_base import DetectionsBase


class LossCalculatorRecorder:
    def __init__(self, config: DictConfig, tbx_writer: SummaryWriter = None):
        self.loss_names = list(config.keys())
        self.tbx_writer = tbx_writer

        func_dict = dict()
        weight_dict = dict()

        each_loss_list_dict = dict()

        for key, value in config.items():
            weight_dict[key] = value.weight
            func_dict[key] = hydra.utils.instantiate(value.func, _convert_="partial")

            each_loss_list_dict[key] = list()

        self.weight_dict = weight_dict
        self.func_dict = func_dict

        self.loss_recorder_per_epoch = each_loss_list_dict

    def init_per_iter(self):
        loss_recorder_per_iter = dict()
        for key in self.loss_names:
            loss_recorder_per_iter[key] = list()

        self.loss_recorder_per_iter = loss_recorder_per_iter

    def step_per_img(self, predictions, targets):
        """calculate the loss and record the loss to the loss recorder list"""

        loss_dict = self._apply_all_loss_func(predictions, targets)
        self._record_per_img(loss_dict)

        return loss_dict

    def _apply_all_loss_func(self, predictions, targets):
        loss_dict = dict()

        if predictions is None:
            for loss_key in self.loss_names:
                loss_dict[loss_key] = torch.tensor(0.0, device=targets.conf.device)
            return loss_dict

        for loss_key, loss_func in self.func_dict.items():
            loss_dict[loss_key] = loss_func(predictions, targets)

        return loss_dict

    def _record_per_img(self, loss_dict):
        for loss_key, loss_val in loss_dict.items():

            self.loss_recorder_per_iter[loss_key].append(
                self.weight_dict[loss_key] * loss_val
            )

    def step_per_iter(self):
        """before backpropagation, apply the loss and record the loss to tensorboard"""

        loss_dict = self._mean_per_iter()
        self._record_per_iter(loss_dict)

        total_loss = torch.stack(list(loss_dict.values())).sum()
        return total_loss

    def _mean_per_iter(self):
        loss_dict = dict()
        for loss_key, loss_list in self.loss_recorder_per_iter.items():
            loss_dict[loss_key] = torch.mean(torch.stack(loss_list))

        return loss_dict

    def _record_per_iter(self, loss_dict):
        for key in self.loss_names:
            self.loss_recorder_per_epoch[key].append(
                loss_dict[key].detach().cpu().resolve_conj().resolve_neg().numpy()
            )

    def step_per_epoch(self, epoch):
        self._record_per_epoch(epoch)

    def _record_per_epoch(self, epoch):

        if self.tbx_writer is None:
            return

        total_loss = 0
        for loss_key, loss_vales in self.loss_recorder_per_epoch.items():
            loss = np.stack(loss_vales).mean()
            total_loss += loss

            self.tbx_writer.add_scalar(
                f"loss/{loss_key}",
                loss,
                epoch,
            )

        self.tbx_writer.add_scalar("loss/total_loss", total_loss, epoch)
