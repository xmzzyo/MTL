# coding: utf-8

# A modified version of the trainer showcased in GLUE: https://github.com/nyu-mll/GLUE-baselines
import os
import time
from copy import deepcopy

import numpy as np

from typing import List, Optional, Dict, Iterable, Union, Any, Tuple

import shutil
import torch
from allennlp.data import DataIterator, Instance
from allennlp.nn.util import move_to_device
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
from allennlp.training import util as training_util, Trainer, TrainerBase

from mtl.common.logger import logger

from mtl.tasks import Task


class GanMtlTrainer:
    def __init__(self,
                 model: Model,
                 task_list: List[Task],
                 optimizer_params: Params,
                 lr_scheduler_params: Params,
                 patience: Optional[int] = None,
                 num_epochs: int = 20,
                 serialization_dir: Optional[str] = None,
                 cuda_device: int = -1,
                 gradient_accumulation_steps: int = 1,
                 grad_norm: Optional[float] = None,
                 grad_clipping: Optional[float] = None,
                 min_lr: float = 0.00001,
                 no_tqdm: bool = False,
                 momentum_scheduler: Optional[MomentumScheduler] = None,
                 summary_interval: int = 50,
                 histogram_interval: int = 50,
                 should_log_parameter_statistics: bool = True,
                 should_log_learning_rate: bool = True,
                 sampling_method: str = "proportional",
                 moving_average: Optional[MovingAverage] = None) -> None:

        self._model = model

        self._task_list = task_list
        self._n_tasks = len(self._task_list)

        self._optimizer_params = optimizer_params
        self._optimizers = {}
        self._lr_scheduler_params = lr_scheduler_params
        self._schedulers = {}
        self._all_params = [(n, p) for n, p in self._model.named_parameters() if p.requires_grad]
        self._params_exclude_share_encoder = [(n, p) for n, p in self._model.named_parameters() if
                                              p.requires_grad and "_share_encoder" not in n]
        self._params_exclude_share_discriminator = [(n, p) for n, p in self._model.named_parameters() if
                                                    p.requires_grad and "_s_domain_discriminator" not in n]

        for task in self._task_list:
            task_name = task._name
            self._optimizers[task_name] = {}
            self._optimizers[task_name]["all_params"] = Optimizer.from_params(
                model_parameters=self._all_params, params=deepcopy(optimizer_params)
            )
            self._optimizers[task_name]["exclude_share_encoder"] = Optimizer.from_params(
                model_parameters=self._params_exclude_share_encoder, params=deepcopy(optimizer_params)
            )
            self._optimizers[task_name]["exclude_share_discriminator"] = Optimizer.from_params(
                model_parameters=self._params_exclude_share_discriminator, params=deepcopy(optimizer_params)
            )
            self._schedulers[task_name] = {}
            self._schedulers[task_name] = LearningRateScheduler.from_params(
                optimizer=self._optimizers[task_name]["all_params"], params=deepcopy(lr_scheduler_params)
            )
            # self._schedulers[task_name]["all_params"] = LearningRateScheduler.from_params(
            #     optimizer=self._optimizers[task_name]["all_params"], params=deepcopy(lr_scheduler_params)
            # )
            # self._schedulers[task_name]["exclude_share_encoder"] = LearningRateScheduler.from_params(
            #     optimizer=self._optimizers[task_name]["exclude_share_encoder"], params=deepcopy(lr_scheduler_params)
            # )
            # self._schedulers[task_name]["exclude_share_discriminator"] = LearningRateScheduler.from_params(
            #     optimizer=self._optimizers[task_name]["exclude_share_discriminator"],
            #     params=deepcopy(lr_scheduler_params)
            # )

        self._serialization_dir = serialization_dir
        self._cuda_device = cuda_device
        if self._cuda_device >= 0:
            check_for_gpu(self._cuda_device)
            self._model = self._model.cuda(self._cuda_device)
        self._patience = patience
        self._num_epochs = num_epochs
        self._epoch_trained = 0

        self._gradient_accumulation_steps = gradient_accumulation_steps
        self._grad_norm = grad_norm
        self._grad_clipping = grad_clipping
        self._min_lr = min_lr
        self._no_tqdm = no_tqdm
        self._sampling_method = sampling_method

        self._task_infos = None
        self._metric_infos = None

        self._tr_generators = None
        self._global_step = 0

        self._batch_num_total = 0

        self._tensorboard = TensorboardWriter(
            get_batch_num_total=lambda: self._batch_num_total,
            serialization_dir=serialization_dir,
            summary_interval=summary_interval,
            histogram_interval=histogram_interval,
            should_log_parameter_statistics=should_log_parameter_statistics,
            should_log_learning_rate=should_log_learning_rate)

        self._last_log = 0.0  # time of last logging

        # Enable activation logging.
        if histogram_interval is not None:
            self._tensorboard.enable_activation_logging(self._model)

    def _rescale_gradients(self) -> Optional[float]:
        return training_util.rescale_gradients(self._model, self._grad_norm)

    def _enable_gradient_clipping(self) -> None:
        training_util.enable_gradient_clipping(self._model, self._grad_clipping)

    def _train_epoch(self, total_n_tr_batches: int, sampling_prob: List, reverse=False, train_D=False) -> Dict[
        str, float]:
        self._model.train()  # Set the model to "train" mode.

        if reverse:
            logger.info("Training Generator- Begin")
        elif not train_D:
            logger.info("Training Init Generator- Begin")

        if train_D:
            logger.info("Training Discriminator- Begin")
        logger.info("reverse is {}, train_D is {}", reverse, train_D)

        ### Reset training and trained batches counter before new training epoch ###
        for _, task_info in self._task_infos.items():
            task_info["tr_loss_cum"] = 0.0
            task_info['stm_loss'] = 0.0
            task_info['p_d_loss'] = 0.0
            task_info['s_d_loss'] = 0.0
            task_info['valid_loss'] = 0.0
            task_info["n_batches_trained_this_epoch"] = 0
        all_tr_metrics = {}  # BUG TO COMPLETE COMMENT TO MAKE IT MORE CLEAR

        ### Start training epoch ###
        epoch_tqdm = tqdm.tqdm(range(total_n_tr_batches), total=total_n_tr_batches)
        histogram_parameters = set(self._model.get_parameters_for_histogram_tensorboard_logging())

        for step, _ in enumerate(epoch_tqdm):
            task_idx = np.argmax(np.random.multinomial(1, sampling_prob))
            task = self._task_list[task_idx]
            task_info = self._task_infos[task._name]

            ### One forward + backward pass ###
            # Call next batch to train
            batch = next(self._tr_generators[task._name])
            self._batch_num_total += 1
            task_info["n_batches_trained_this_epoch"] += 1

            # Load optimizer
            if not train_D:
                optimizer = self._optimizers[task._name]["all_params"]
            else:
                optimizer = self._optimizers[task._name]["exclude_share_encoder"]

            # Get the loss for this batch
            output_dict = self._forward(tensor_batch=batch, task=task, for_training=True, reverse=reverse)
            # if reverse or train_D:
            #     output_dict_fake = self._forward(tensor_batch=batch, task=task, for_training=True, reverse=True)
            # loss = output_dict["stm_loss"]
            # if train_D:
            #     loss = (output_dict["stm_loss"] + output_dict["s_d_loss"] + output_dict_fake["stm_loss"] +
            #             output_dict_fake["s_d_loss"]) / 2.0
            # if reverse:
            #     # loss = (output_dict["stm_loss"] + output_dict["p_d_loss"] + 0.005 * output_dict["s_d_loss"] +
            #     #         output_dict_fake["stm_loss"] + output_dict_fake["p_d_loss"] + 0.005 * output_dict_fake[
            #     #             "s_d_loss"]) / 2.0
            #     loss = (output_dict['loss'] + output_dict_fake['loss']) / 2.0
            loss = output_dict['loss']
            if self._gradient_accumulation_steps > 1:
                loss /= self._gradient_accumulation_steps
            loss.backward()
            task_info["tr_loss_cum"] += loss.item()
            task_info['stm_loss'] += output_dict['stm_loss'].item()
            task_info['p_d_loss'] += output_dict['p_d_loss'].item()
            task_info['s_d_loss'] += output_dict['s_d_loss'].item()
            task_info['valid_loss'] += output_dict['valid_loss'].item()
            # if reverse or train_D:
            #     task_info['stm_loss'] += output_dict_fake['stm_loss'].item()
            #     task_info['stm_loss'] /= 2.0
            #     task_info['p_d_loss'] += output_dict_fake['p_d_loss'].item()
            #     task_info['p_d_loss'] /= 2.0
            #     task_info['s_d_loss'] += output_dict_fake['s_d_loss'].item()
            #     task_info['s_d_loss'] /= 2.0
            #     task_info['valid_loss'] += output_dict_fake['valid_loss'].item()
            #     task_info['valid_loss'] /= 2.0
            del loss

            if (step + 1) % self._gradient_accumulation_steps == 0:
                batch_grad_norm = self._rescale_gradients()
                if self._tensorboard.should_log_histograms_this_batch():
                    param_updates = {name: param.detach().cpu().clone()
                                     for name, param in self._model.named_parameters()}
                    optimizer.step()
                    for name, param in self._model.named_parameters():
                        param_updates[name].sub_(param.detach().cpu())
                        update_norm = torch.norm(param_updates[name].view(-1, ))
                        param_norm = torch.norm(param.view(-1, )).cpu()
                        self._tensorboard.add_train_scalar("gradient_update/" + name,
                                                           update_norm / (param_norm + 1e-7))
                else:
                    optimizer.step()
                optimizer.zero_grad()

            ### Get metrics for all progress so far, update tqdm, display description ###
            task_metrics = self._get_metrics(task=task)
            task_metrics["loss"] = float(
                task_info["tr_loss_cum"] / (task_info["n_batches_trained_this_epoch"] + 0.000_001)
            )
            task_metrics["stm_loss"] = float(
                task_info["stm_loss"] / (task_info["n_batches_trained_this_epoch"] + 0.000_001)
            )
            task_metrics["p_d_loss"] = float(
                task_info["p_d_loss"] / (task_info["n_batches_trained_this_epoch"] + 0.000_001)
            )
            task_metrics["s_d_loss"] = float(
                task_info["s_d_loss"] / (task_info["n_batches_trained_this_epoch"] + 0.000_001)
            )
            task_metrics["valid_loss"] = float(
                task_info["valid_loss"] / (task_info["n_batches_trained_this_epoch"] + 0.000_001)
            )
            description = training_util.description_from_metrics(task_metrics)
            epoch_tqdm.set_description(task._name + ", " + description)

            # Log parameter values to Tensorboard
            if self._tensorboard.should_log_this_batch():
                self._tensorboard.log_parameter_and_gradient_statistics(self._model, batch_grad_norm)
                self._tensorboard.log_learning_rates(self._model, optimizer)

                self._tensorboard.log_metrics(
                    {"epoch_metrics/" + task._name + "/" + k: v for k, v in task_metrics.items()})

            if self._tensorboard.should_log_histograms_this_batch():
                self._tensorboard.log_histograms(self._model, histogram_parameters)
            self._global_step += 1

        ### Bookkeeping all the training metrics for all the tasks on the training epoch that just finished ###
        for task in self._task_list:
            task_info = self._task_infos[task._name]

            task_info["total_n_batches_trained"] += task_info["n_batches_trained_this_epoch"]
            task_info["last_log"] = time.time()

            task_metrics = self._get_metrics(task=task, reset=True)
            if task._name not in all_tr_metrics:
                all_tr_metrics[task._name] = {}
            for name, value in task_metrics.items():
                all_tr_metrics[task._name][name] = value
            all_tr_metrics[task._name]["loss"] = float(
                task_info["tr_loss_cum"] / (task_info["n_batches_trained_this_epoch"] + 0.000_000_01)
            )
            all_tr_metrics[task._name]["stm_loss"] = float(
                task_info["stm_loss"] / (task_info["n_batches_trained_this_epoch"] + 0.000_000_01)
            )
            all_tr_metrics[task._name]["p_d_loss"] = float(
                task_info["p_d_loss"] / (task_info["n_batches_trained_this_epoch"] + 0.000_000_01)
            )
            all_tr_metrics[task._name]["s_d_loss"] = float(
                task_info["s_d_loss"] / (task_info["n_batches_trained_this_epoch"] + 0.000_000_01)
            )
            all_tr_metrics[task._name]["valid_loss"] = float(
                task_info["valid_loss"] / (task_info["n_batches_trained_this_epoch"] + 0.000_000_01)
            )

            # Tensorboard - Training metrics for this epoch
            for metric_name, value in all_tr_metrics[task._name].items():
                self._tensorboard.add_train_scalar(
                    name="task_" + task._name + "/" + metric_name, value=value
                )

        logger.info("Train - End")
        return all_tr_metrics

    def _validation(self, n_epoch: int) -> Tuple[float, int]:
        ### Begin validation of the model ###
        logger.info("Validation - Begin")
        all_val_metrics = {}

        self._model.eval()  # Set the model into evaluation mode

        avg_accuracy = 0.0

        for task_idx, task in enumerate(self._task_list):
            logger.info("Validation - Task {}/{}: {}", task_idx + 1, self._n_tasks, task._name)

            val_loss = 0.0
            n_batches_val_this_epoch_this_task = 0
            n_val_batches = self._task_infos[task._name]["n_val_batches"]
            scheduler = self._schedulers[task._name]

            # Create tqdm generator for current tasks's validation
            data_iterator = task._data_iterator
            val_generator = data_iterator(task._validation_data, num_epochs=1, shuffle=False)
            val_generator_tqdm = tqdm.tqdm(val_generator, total=n_val_batches)

            # Iterate over each validation batch for this tasks
            for batch in val_generator_tqdm:
                n_batches_val_this_epoch_this_task += 1

                # Get the loss
                val_output_dict = self._forward(batch, task=task, for_training=False)
                loss = val_output_dict["stm_loss"]
                val_loss += loss.item()
                del loss

                # Get metrics for all progress so far, update tqdm, display description
                task_metrics = self._get_metrics(task=task)
                task_metrics["loss"] = float(val_loss / n_batches_val_this_epoch_this_task)
                description = training_util.description_from_metrics(task_metrics)
                val_generator_tqdm.set_description(description)

            # Get tasks validation metrics and store them in all_val_metrics
            task_metrics = self._get_metrics(task=task, reset=True)
            if task._name not in all_val_metrics:
                all_val_metrics[task._name] = {}
            for name, value in task_metrics.items():
                all_val_metrics[task._name][name] = value
            all_val_metrics[task._name]["loss"] = float(val_loss / n_batches_val_this_epoch_this_task)

            avg_accuracy += task_metrics["sentiment_acc"]

            # Tensorboard - Validation metrics for this epoch
            for metric_name, value in all_val_metrics[task._name].items():
                self._tensorboard.add_validation_scalar(
                    name="task_" + task._name + "/" + metric_name, value=value
                )

            ### Perform a patience check and update the history of validation metric for this tasks ###
            this_epoch_val_metric = all_val_metrics[task._name][task._val_metric]
            metric_history = self._metric_infos[task._name]["hist"]

            metric_history.append(this_epoch_val_metric)
            is_best_so_far, out_of_patience = self._check_history(
                metric_history=metric_history,
                cur_score=this_epoch_val_metric,
                should_decrease=task._val_metric_decreases,
            )

            if is_best_so_far:
                logger.info("Best model found for {}.", task._name)
                self._metric_infos[task._name]["best"] = (n_epoch, all_val_metrics)
            if out_of_patience and not self._metric_infos[task._name]["is_out_of_patience"]:
                self._metric_infos[task._name]["is_out_of_patience"] = True
                logger.info("Task {} is out of patience and vote to stop the training.", task._name)

            # The LRScheduler API is agnostic to whether your schedule requires a validation metric -
            # if it doesn't, the validation metric passed here is ignored.
            scheduler.step(this_epoch_val_metric, n_epoch)

        logger.info("Validation - End")
        return all_val_metrics, avg_accuracy

    def train(self, recover: bool = False) -> Dict[str, Any]:

        # 1 train sentiment classifier & private classifier & domain embeddings => init G 50 epoch
        # 2 fix share encoder(+domain embeddings?), train share classifier(cls&real/fake) & others => train D
        # 3 fix share classifier, train share encoder, reverse share classifier input gradient  min loss => train G
        training_start_time = time.time()

        if recover:
            try:
                n_epoch, should_stop = self._restore_checkpoint()
                logger.info("Loaded model from checkpoint. Starting at epoch {}", n_epoch)
            except RuntimeError:
                raise ConfigurationError(
                    "Could not recover training from the checkpoint.  Did you mean to output to "
                    "a different serialization directory or delete the existing serialization "
                    "directory?"
                )
        else:
            n_epoch, should_stop = 0, False

            ### Store all the necessary informations and attributes about the tasks ###
            task_infos = {task._name: {} for task in self._task_list}
            for task_idx, task in enumerate(self._task_list):
                task_info = task_infos[task._name]

                # Store statistiscs on training and validation batches
                data_iterator = task._data_iterator
                n_tr_batches = data_iterator.get_num_batches(task._train_data)
                n_val_batches = data_iterator.get_num_batches(task._validation_data)
                task_info["n_tr_batches"] = n_tr_batches
                task_info["n_val_batches"] = n_val_batches

                # Create counter for number of batches trained during the whole
                # training for this specific tasks
                task_info["total_n_batches_trained"] = 0

                task_info["last_log"] = time.time()  # Time of last logging
            self._task_infos = task_infos

            ### Bookkeeping the validation metrics ###
            metric_infos = {
                task._name: {
                    "val_metric": task._val_metric,
                    "hist": [],
                    "is_out_of_patience": False,
                    "min_lr_hit": False,
                    "best": (-1, {}),
                }
                for task in self._task_list
            }
            self._metric_infos = metric_infos

        ### Write log ###
        total_n_tr_batches = 0  # The total number of training batches across all the datasets.
        for task_name, info in self._task_infos.items():
            total_n_tr_batches += info["n_tr_batches"]
            logger.info("Task {}:", task_name)
            logger.info("\t{} training batches", info["n_tr_batches"])
            logger.info("\t{} validation batches", info["n_val_batches"])

        ### Create the training generators/iterators tqdm ###
        self._tr_generators = {}
        for task in self._task_list:
            data_iterator = task._data_iterator
            tr_generator = data_iterator(task._train_data, num_epochs=None)
            self._tr_generators[task._name] = tr_generator

        ### Create sampling probability distribution ###
        if self._sampling_method == "uniform":
            sampling_prob = [float(1 / self._n_tasks)] * self._n_tasks
        elif self._sampling_method == "proportional":
            sampling_prob = [float(info["n_tr_batches"] / total_n_tr_batches) for info in self._task_infos.values()]

        ### Enable gradient clipping ###
        # Only if self._grad_clipping is specified
        self._enable_gradient_clipping()

        ### Setup is ready. Training of the model can begin ###
        logger.info("Set up ready. Beginning training/validation.")

        avg_accuracies = []
        best_accuracy = 0.0

        ### Begin Training of the model ###
        while not should_stop:
            ### Log Infos: current epoch count and CPU/GPU usage ###
            logger.info("")
            logger.info("Epoch {}/{} - Begin", n_epoch, self._num_epochs - 1)
            logger.info(f"Peak CPU memory usage MB: {peak_memory_mb()}")
            for gpu, memory in gpu_memory_mb().items():
                logger.info(f"GPU {gpu} memory usage MB: {memory}")

            # if n_epoch <= 10:
            #     # init generator
            #     all_tr_metrics = self._train_epoch(total_n_tr_batches, sampling_prob)
            # # train discriminator 3 epochs
            # # elif 10 < n_epoch < 20 or n_epoch % 2 == 0:
            # #     all_tr_metrics = self._train_epoch(total_n_tr_batches, sampling_prob, train_D=True)
            # else:
            # train adversarial generator every 3 epoch
            all_tr_metrics = self._train_epoch(total_n_tr_batches, sampling_prob, reverse=True)

            all_val_metrics, avg_accuracy = self._validation(n_epoch)
            is_best = False
            if best_accuracy < avg_accuracy:
                best_accuracy = avg_accuracy
                logger.info("Best accuracy found --- {}", best_accuracy / self._n_tasks)
                is_best = True

            ### Print all training and validation metrics for this epoch ###
            logger.info("***** Epoch {}/{} Statistics *****", n_epoch, self._num_epochs - 1)
            for task in self._task_list:
                logger.info("Statistic: {}", task._name)
                logger.info(
                    "\tTraining - {}: {:3d}",
                    "Nb batches trained",
                    self._task_infos[task._name]["n_batches_trained_this_epoch"],
                )
                for metric_name, value in all_tr_metrics[task._name].items():
                    logger.info("\tTraining - {}: {:.3f}", metric_name, value)
                for metric_name, value in all_val_metrics[task._name].items():
                    logger.info("\tValidation - {}: {:.3f}", metric_name, value)
            logger.info("***** Average accuracy is {:.6f} *****", avg_accuracy / self._n_tasks)
            avg_accuracies.append(avg_accuracy / self._n_tasks)
            logger.info("**********")

            ### Check to see if should stop ###
            stop_tr, stop_val = True, True

            for task in self._task_list:
                # task_info = self._task_infos[tasks._name]
                if self._optimizers[task._name]['exclude_share_encoder'].param_groups[0]["lr"] < self._min_lr and \
                        self._optimizers[task._name]['exclude_share_discriminator'].param_groups[0][
                            "lr"] < self._min_lr:
                    logger.info("Minimum lr hit on {}.", task._name)
                    logger.info("Task {} vote to stop training.", task._name)
                    metric_infos[task._name]["min_lr_hit"] = True
                stop_tr = stop_tr and self._metric_infos[task._name]["min_lr_hit"]
                stop_val = stop_val and self._metric_infos[task._name]["is_out_of_patience"]

            if stop_tr:
                should_stop = True
                logger.info("All tasks hit minimum lr. Stopping training.")
            if stop_val:
                should_stop = True
                logger.info("All metrics ran out of patience. Stopping training.")
            if n_epoch >= self._num_epochs - 1:
                should_stop = True
                logger.info("Maximum number of epoch hit. Stopping training.")

            self._save_checkpoint(n_epoch, should_stop, is_best)

            ### Update n_epoch ###
            # One epoch = doing N (forward + backward) pass where N is the total number of training batches.
            n_epoch += 1
            self._epoch_trained = n_epoch

        logger.info("Max accuracy is {:.6f}", max(avg_accuracies))

        ### Summarize training at the end ###
        logger.info("***** Training is finished *****")
        logger.info("Stopped training after {} epochs", n_epoch)
        return_metrics = {}
        for task_name, task_info in self._task_infos.items():
            nb_epoch_trained = int(task_info["total_n_batches_trained"] / task_info["n_tr_batches"])
            logger.info(
                "Trained {} for {} batches ~= {} epochs",
                task_name,
                task_info["total_n_batches_trained"],
                nb_epoch_trained,
            )
            return_metrics[task_name] = {
                "best_epoch": self._metric_infos[task_name]["best"][0],
                "nb_epoch_trained": nb_epoch_trained,
                "best_epoch_val_metrics": self._metric_infos[task_name]["best"][1],
            }

        training_elapsed_time = time.time() - training_start_time
        return_metrics["training_duration"] = time.strftime("%d:%H:%M:%S", time.gmtime(training_elapsed_time))
        return_metrics["nb_epoch_trained"] = n_epoch

        return return_metrics

    def _check_history(self, metric_history: List[float], cur_score: float, should_decrease: bool = False):
        patience = self._patience + 1
        best_fn = min if should_decrease else max
        best_score = best_fn(metric_history)
        if best_score == cur_score:
            best_so_far = metric_history.index(best_score) == len(metric_history) - 1
        else:
            best_so_far = False

        out_of_patience = False
        if len(metric_history) > patience:
            if should_decrease:
                out_of_patience = max(metric_history[-patience:]) <= cur_score
            else:
                out_of_patience = min(metric_history[-patience:]) >= cur_score

        if best_so_far and out_of_patience:  # then something is up
            print("Something is up")

        return best_so_far, out_of_patience

    def _forward(self, tensor_batch: torch.Tensor, task: Task = None, for_training: bool = False, reverse=False):
        if task is not None:
            # tensor_batch = move_to_device(tensor_batch, self._cuda_device)
            output_dict = self._model.forward(
                task_name=task._name, tensor_batch=tensor_batch, reverse=reverse, for_training=for_training,
                epoch_trained=self._epoch_trained
            )
            if for_training:
                try:
                    # loss = output_dict["stm_loss"]
                    output_dict["p_d_loss"] += self._model.get_regularization_penalty()
                except KeyError:
                    raise RuntimeError(
                        "The model you are trying to optimize does not contain a"
                        " `loss` key in the output of model.forward(inputs)."
                    )
            return output_dict
        else:
            raise ConfigurationError("Cannot call forward through tasks `None`")

    def _get_metrics(self, task: Task, reset: bool = False):
        task_tagger = getattr(self._model, "_tagger_" + task._name)
        return task_tagger.get_metrics(reset)

    def _save_checkpoint(self, epoch: int, should_stop: bool, is_best: bool = False) -> None:
        ### Saving training state ###
        training_state = {
            "epoch": epoch,
            "should_stop": should_stop,
            "metric_infos": self._metric_infos,
            "task_infos": self._task_infos,
            "schedulers": {},
            "optimizers": {},
        }

        if self._optimizers is not None:
            for task_name, optimizers in self._optimizers.items():
                training_state["optimizers"][task_name] = {}
                for params_name, optimizer in optimizers.items():
                    training_state["optimizers"][task_name][params_name] = optimizer.state_dict()
        if self._schedulers is not None:
            for task_name, scheduler in self._schedulers.items():
                training_state["schedulers"][task_name] = scheduler.lr_scheduler.state_dict()

        training_path = os.path.join(self._serialization_dir, "training_state.th")
        torch.save(training_state, training_path)
        logger.info("Checkpoint - Saved training state to {}", training_path)

        ### Saving model state ###
        model_path = os.path.join(self._serialization_dir, "model_state.th")
        model_state = self._model.state_dict()
        torch.save(model_state, model_path)
        logger.info("Checkpoint - Saved model state to {}", model_path)

        if is_best:
            logger.info("Checkpoint - Best validation performance so far for all tasks")
            logger.info("Checkpoint - Copying weights to '{}/best_all.th'.", self._serialization_dir)
            shutil.copyfile(model_path, os.path.join(self._serialization_dir, "best_all.th"))

        ### Saving best models for each tasks ###
        for task_name, infos in self._metric_infos.items():
            best_epoch, _ = infos["best"]
            if best_epoch == epoch:
                logger.info("Checkpoint - Best validation performance so far for {} tasks", task_name)
                logger.info("Checkpoint - Copying weights to '{}/best_{}.th'.", self._serialization_dir, task_name)
                shutil.copyfile(model_path, os.path.join(self._serialization_dir, "best_{}.th".format(task_name)))

    @classmethod
    def from_params(cls, model: Model, task_list: List[Task], serialization_dir: str,
                    params: Params) -> "GanMtlTrainer":
        optimizer_params = params.pop("optimizer")
        lr_scheduler_params = params.pop("scheduler")
        patience = params.pop_int("patience", 2)
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = params.pop_int("cuda_device", -1)
        gradient_accumulation_steps = params.pop_int("gradient_accumulation_steps", 1)
        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)
        min_lr = params.pop_float("min_lr", 0.00001)
        no_tqdm = params.pop_bool("no_tqdm", False)
        summary_interval = params.pop("summary_interval", 30)
        histogram_interval = params.pop("histogram_interval", 30)
        sampling_method = params.pop("sampling_method", "proportional")

        params.assert_empty(cls.__name__)
        return GanMtlTrainer(
            model=model,
            task_list=task_list,
            optimizer_params=optimizer_params,
            lr_scheduler_params=lr_scheduler_params,
            patience=patience,
            num_epochs=num_epochs,
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
            gradient_accumulation_steps=gradient_accumulation_steps,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            min_lr=min_lr,
            no_tqdm=no_tqdm,
            summary_interval=summary_interval,
            histogram_interval=histogram_interval,
            sampling_method=sampling_method
        )
