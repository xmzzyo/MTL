# coding: utf-8

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

import tqdm

from allennlp.common import Params, util
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.common.util import peak_memory_mb, gpu_memory_mb
from allennlp.models.model import Model
from allennlp.training import util as training_util, Trainer, TrainerBase

from mtl.common.logger import logger

from mtl.tasks import Task
from train_transfer import TASKS_NAME


class TransferMtlTrainer:
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

        self._optimizers = {}
        self._all_params = [(n, p) for n, p in self._model.named_parameters() if p.requires_grad]
        self._params_share_encoder = [(n, p) for n, p in self._model.named_parameters() if
                                      p.requires_grad and "_shared_encoder" in n]
        self._params_share_encoder_de = [(n, p) for n, p in self._model.named_parameters() if
                                         p.requires_grad and ("_shared_encoder" in n
                                                              or "_seq_vec" in n
                                                              or "_domain_embeddings" in n
                                                              or "_de_attention" in n
                                                              or "_de_feedforward" in n)]
        self._params_share_discriminator = [(n, p) for n, p in self._model.named_parameters() if
                                            p.requires_grad and "_s_domain_discriminator" in n]
        self._params_valid_discriminator = [(n, p) for n, p in self._model.named_parameters() if
                                            p.requires_grad and "_valid_discriminator" in n]

        self._optimizers = dict()
        self._optimizers["all_params"] = Optimizer.from_params(
            model_parameters=self._all_params, params=deepcopy(optimizer_params)
        )
        self._optimizers["share_encoder"] = Optimizer.from_params(
            model_parameters=self._params_share_encoder, params=deepcopy(optimizer_params)
        )
        self._optimizers["share_encoder_de"] = Optimizer.from_params(
            model_parameters=self._params_share_encoder_de, params=deepcopy(optimizer_params)
        )
        self._optimizers["share_discriminator"] = Optimizer.from_params(
            model_parameters=self._params_share_discriminator, params=deepcopy(optimizer_params)
        )
        self._optimizers["valid_discriminator"] = Optimizer.from_params(
            model_parameters=self._params_valid_discriminator, params=deepcopy(optimizer_params)
        )
        self._schedulers = dict()
        self._schedulers["all_params"] = LearningRateScheduler.from_params(
            optimizer=self._optimizers["all_params"], params=deepcopy(lr_scheduler_params)
        )
        self._schedulers["share_encoder"] = LearningRateScheduler.from_params(
            optimizer=self._optimizers["share_encoder"], params=deepcopy(lr_scheduler_params)
        )
        self._schedulers["share_encoder_de"] = LearningRateScheduler.from_params(
            optimizer=self._optimizers["share_encoder_de"], params=deepcopy(lr_scheduler_params)
        )
        self._schedulers["share_discriminator"] = LearningRateScheduler.from_params(
            optimizer=self._optimizers["share_discriminator"], params=deepcopy(lr_scheduler_params)
        )
        self._schedulers["valid_discriminator"] = LearningRateScheduler.from_params(
            optimizer=self._optimizers["valid_discriminator"], params=deepcopy(lr_scheduler_params)
        )
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

    def _log_params_update(self, optimizer):
        if self._tensorboard.should_log_histograms_this_batch():
            # get the magnitude of parameter updates for logging
            # We need a copy of current parameters to compute magnitude of updates,
            # and copy them to CPU so large models won't go OOM on the GPU.
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

    def _log_params_values(self, optimizer, task_name, task_metrics, histogram_parameters):
        batch_grad_norm = self._rescale_gradients()
        # Log parameter values to Tensorboard
        if self._tensorboard.should_log_this_batch():
            self._tensorboard.log_parameter_and_gradient_statistics(self._model, batch_grad_norm)
            self._tensorboard.log_learning_rates(self._model, optimizer)

            self._tensorboard.log_metrics(
                {"epoch_metrics/" + task_name + "/" + k: v for k, v in task_metrics.items()})

        if self._tensorboard.should_log_histograms_this_batch():
            self._tensorboard.log_histograms(self._model, histogram_parameters)

    def _train_epoch(self, total_n_tr_batches: int, sampling_prob: List) -> Dict[
        str, float]:
        self._model.train()  # Set the model to "train" mode.

        ### Reset training and trained batches counter before new training epoch ###
        for _, task_info in self._task_infos.items():
            task_info["tr_loss_cum"] = 0.0
            task_info['stm_loss'] = 0.0
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

            # -------------------------------------
            # train sentiment classify
            # -------------------------------------

            # Load optimizer
            optimizer = self._optimizers["all_params"]

            # Get the loss for this batch
            output_dict = self._forward(tensor_batch=batch, task=task, for_training=True, train_stage="stm")

            loss = output_dict['loss']
            task_info['stm_loss'] += loss.item()
            loss.backward()
            del loss

            self._log_params_update(optimizer)

            # -------------------------------------
            # train domain classify
            # -------------------------------------

            # optimizer = self._optimizers["share_discriminator"]
            #
            # # Get the loss for this batch
            # output_dict = self._forward(tensor_batch=batch, task=task, for_training=True, train_stage="sd")
            #
            # loss = output_dict['loss']
            # task_info['s_d_loss'] += loss.item()
            # loss.backward()
            # del loss
            #
            # self._log_params_update(optimizer)

            # -------------------------------------
            # train adversarial domain classify
            # -------------------------------------
            optimizer = self._optimizers["all_params"]

            # Get the loss for this batch
            output_dict = self._forward(tensor_batch=batch, task=task, for_training=True, reverse=True,
                                        train_stage="sd")

            loss = output_dict['loss']
            loss = 0.05 * loss
            task_info['s_d_loss'] += loss.item()
            loss.backward()
            del loss

            self._log_params_update(optimizer)

            # -------------------------------------
            # train valid classify
            # -------------------------------------
            # optimizer = self._optimizers["valid_discriminator"]
            #
            # # Get the loss for this batch
            # output_dict = self._forward(tensor_batch=batch, task=task, for_training=True, train_stage="valid")
            #
            # loss = output_dict['loss']
            # task_info['valid_loss'] += loss.item()
            # loss.backward()
            # del loss
            #
            # self._log_params_update(optimizer)
            # -------------------------------------
            # train adversarial valid classify
            # -------------------------------------
            optimizer = self._optimizers["all_params"]

            # Get the loss for this batch
            output_dict = self._forward(tensor_batch=batch, task=task, for_training=True, reverse=True,
                                        train_stage="valid")

            loss = output_dict['loss']
            loss = 0.05 * loss
            task_info['valid_loss'] += loss.item()
            loss.backward()
            del loss

            self._log_params_update(optimizer)

            ### Get metrics for all progress so far, update tqdm, display description ###
            task_metrics = self._model.get_metrics(task_name=task._name)
            task_metrics["stm_loss"] = float(
                task_info["stm_loss"] / (task_info["n_batches_trained_this_epoch"] + 0.000_001)
            )
            task_metrics["s_d_loss"] = float(
                task_info["s_d_loss"] / (task_info["n_batches_trained_this_epoch"] + 0.000_001)
            )
            task_metrics["valid_loss"] = float(
                task_info["valid_loss"] / (task_info["n_batches_trained_this_epoch"] + 0.000_001)
            )
            description = training_util.description_from_metrics(task_metrics)
            epoch_tqdm.set_description(description)

            self._log_params_values(optimizer, task._name, task_metrics, histogram_parameters)
            self._global_step += 1

        ### Bookkeeping all the training metrics for all the tasks on the training epoch that just finished ###
        for task in self._task_list:
            task_info = self._task_infos[task._name]

            task_info["total_n_batches_trained"] += task_info["n_batches_trained_this_epoch"]
            task_info["last_log"] = time.time()

            task_metrics = self._model.get_metrics(task_name=task._name, reset=True)
            if task._name not in all_tr_metrics:
                all_tr_metrics[task._name] = {}
            for name, value in task_metrics.items():
                all_tr_metrics[task._name][name] = value
            all_tr_metrics[task._name]["stm_loss"] = float(
                task_info["stm_loss"] / (task_info["n_batches_trained_this_epoch"] + 0.000_000_01)
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

            # Create tqdm generator for current tasks's validation
            data_iterator = task._data_iterator
            val_generator = data_iterator(task._validation_data, num_epochs=1, shuffle=False)
            val_generator_tqdm = tqdm.tqdm(val_generator, total=n_val_batches)

            # Iterate over each validation batch for this tasks
            for batch in val_generator_tqdm:
                n_batches_val_this_epoch_this_task += 1

                # Get the loss
                val_output_dict = self._forward(batch, task=task, for_training=False)
                loss = val_output_dict["loss"]
                val_loss += loss.item()
                del loss

                # Get metrics for all progress so far, update tqdm, display description
                task_metrics = self._model.get_metrics(task_name=task._name)
                task_metrics["loss"] = float(val_loss / n_batches_val_this_epoch_this_task)
                description = training_util.description_from_metrics(task_metrics)
                val_generator_tqdm.set_description(description)

            # Get tasks validation metrics and store them in all_val_metrics
            task_metrics = self._model.get_metrics(task_name=task._name, reset=True)
            if task._name not in all_val_metrics:
                all_val_metrics[task._name] = {}
            for name, value in task_metrics.items():
                all_val_metrics[task._name][name] = value
            all_val_metrics[task._name]["loss"] = float(val_loss / n_batches_val_this_epoch_this_task)

            avg_accuracy += task_metrics["{}_stm_acc".format(task._name)]

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

        for scheduler in self._schedulers.values():
            scheduler.step(epoch=n_epoch)

        logger.info("Validation - End")
        return all_val_metrics, avg_accuracy

    def train(self, recover: bool = False) -> Dict[str, Any]:

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

                # Store statistics on training and validation batches
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
            # num_epochs=None -> generate forever
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
            all_tr_metrics = self._train_epoch(total_n_tr_batches, sampling_prob)

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
                if self._optimizers['all_params'].param_groups[0]["lr"] < self._min_lr:
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

    def _forward(self, tensor_batch: torch.Tensor, task: Task = None, for_training: bool = False, reverse=False,
                 train_stage="stm"):
        train_stages = ["stm", "sd", "valid"]
        if task is not None:
            # tensor_batch = move_to_device(tensor_batch, self._cuda_device)
            task_index = TASKS_NAME.index(task._name)
            tensor_batch['task_index'] = torch.tensor(task_index)
            tensor_batch["reverse"] = torch.tensor(reverse)
            tensor_batch['for_training'] = torch.tensor(for_training)
            train_stage = train_stages.index(train_stage)
            tensor_batch['train_stage'] = torch.tensor(train_stage)
            tensor_batch = move_to_device(tensor_batch, self._cuda_device)
            output_dict = self._model.forward(**tensor_batch)
            if for_training:
                try:
                    # loss = output_dict["stm_loss"]
                    output_dict["loss"] += self._model.get_regularization_penalty()
                except KeyError:
                    raise RuntimeError(
                        "The model you are trying to optimize does not contain a"
                        " `loss` key in the output of model.forward(inputs)."
                    )
            return output_dict
        else:
            raise ConfigurationError("Cannot call forward through tasks `None`")

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
            for task_name, optimizer in self._optimizers.items():
                training_state["optimizers"][task_name] = optimizer.state_dict()
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
                    params: Params) -> "TransferMtlTrainer":
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
        return TransferMtlTrainer(
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
