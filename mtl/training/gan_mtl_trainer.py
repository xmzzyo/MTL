# coding: utf-8

# A modified version of the trainer showcased in GLUE: https://github.com/nyu-mll/GLUE-baselines

import time
from copy import deepcopy

import numpy as np

from typing import List, Optional, Dict, Iterable, Union, Any

import torch
from allennlp.data import DataIterator, Instance
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer
from allennlp.training.tensorboard_writer import TensorboardWriter
from overrides import overrides

import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common.util import peak_memory_mb, gpu_memory_mb
from allennlp.models.model import Model
from allennlp.training import util as training_util, Trainer

# from apex.optimizers import FP16_Optimizer
from mtl.common.logger import logger
from train_stmcls import TASKS_NAME

from mtl.tasks import Task
from mtl.training import MultiTaskTrainer


# from apex import amp

# amp_handle = amp.init()


@Trainer.register("gan_mtl_trainer")
class GanMtlTrainer(Trainer):
    def __init__(self,
                 model: Model,
                 task_list: List[Task],
                 optimizer_params: Params,
                 lr_scheduler_params: Params,
                 patience: Optional[int] = None,
                 num_epochs: int = 20,
                 serialization_dir: Optional[str] = None,
                 cuda_device: int = -1,
                 grad_norm: Optional[float] = None,
                 grad_clipping: Optional[float] = None,
                 min_lr: float = 0.00001,
                 momentum_scheduler: Optional[MomentumScheduler] = None,
                 summary_interval: int = 50,
                 histogram_interval: int = 50,
                 should_log_parameter_statistics: bool = True,
                 should_log_learning_rate: bool = True,
                 moving_average: Optional[MovingAverage] = None) -> None:
        super().__init__(model, task_list, optimizer_params, patience, num_epochs, serialization_dir, cuda_device,
                         grad_norm, grad_clipping, min_lr, lr_scheduler_params, momentum_scheduler, summary_interval,
                         histogram_interval, should_log_parameter_statistics, should_log_learning_rate, moving_average)

        self._model = model
        parameters_to_train = [(n, p) for n, p in self._model.named_parameters() if p.requires_grad]

        self._task_list = task_list
        self._n_tasks = len(self._task_list)

        self._optimizer_params = optimizer_params
        self._optimizers = {}
        self._lr_scheduler_params = lr_scheduler_params
        self._schedulers = {}
        share_encoder_paras = [(n, p) for n, p in self._model.named_parameters() if p.requires_grad and "" in n]

        for task in self._task_list:
            task_name = task._name

            self._optimizers[task_name] = Optimizer.from_params(
                model_parameters=parameters_to_train, params=deepcopy(optimizer_params)
            )
            self._schedulers[task_name] = LearningRateScheduler.from_params(
                optimizer=self._optimizers[task_name], params=deepcopy(lr_scheduler_params)
            )

        self._serialization_dir = serialization_dir

        if self.cuda_device >= 0:
            check_for_gpu(self._cuda_device)
            self.model = self.model.cuda(self._cuda_device)

        self._task_infos = None
        self._metric_infos = None

        self._tr_generators = None

    def train(self) -> Dict[str, Any]:
        return super().train()

    @classmethod
    def from_params(cls, model: Model, serialization_dir: str, iterator: DataIterator, train_data: Iterable[Instance],
                    validation_data: Optional[Iterable[Instance]], params: Params,
                    validation_iterator: DataIterator = None) -> 'Trainer':
        return super().from_params(model, serialization_dir, iterator, train_data, validation_data, params,
                                   validation_iterator)
