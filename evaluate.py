# coding: utf-8

"""
The ``evaluate.py`` file can be used to
evaluate a trained model against a dataset
and report any metrics calculated by the model.
It requires a configuration file and a directory in
which to write the results.

.. code-block:: bash

   $ python evaluate.py --help
    usage: evaluate.py [-h] -s SERIALIZATION_DIR [-g]

    optional arguments:
    -h, --help            show this help message and exit
    -s SERIALIZATION_DIR, --serialization_dir SERIALIZATION_DIR
                            Directory in which to save the model and its logs.
    -g, --gold_mentions   Whether or not evaluate using gold mentions in
                            coreference
"""

import argparse
import os
import json
import re
from copy import deepcopy
import tqdm
from typing import Dict, Any, Iterable
import torch

from allennlp.models.model import Model
from allennlp.data import Instance
from allennlp.data.iterators import DataIterator
from allennlp.common.checks import check_for_gpu
from allennlp.common.params import Params
from allennlp.nn import util
from allennlp.data import Vocabulary
from allennlp.nn.util import move_to_device
from allennlp.training import util as training_util

from mtl.common.logger import logger
from mtl.tasks import Task
from mtl.common.util import create_and_set_iterators
from train_transfer import TASKS_NAME


def evaluate(
        model: Model, instances: Iterable[Instance], task_name: str, data_iterator: DataIterator, cuda_device: int
) -> Dict[str, Any]:
    """
    Evaluate a model for a particular tasks (usually after training).
    
    Parameters
    ----------
    model : ``allennlp.models.model.Model``, required
        The model to evaluate
    instances : ``Iterable[Instance]``, required
        The (usually test) dataset on which to evalute the model.
    task_name : ``str``, required
        The name of the tasks on which evaluate the model.
    data_iterator : ``DataIterator``
        Iterator that go through the dataset.
    cuda_device : ``int``
        Cuda device to use.
        
    Returns
    -------
    metrics :  ``Dict[str, Any]``
        A dictionary containing the metrics on the evaluated dataset.
    """
    check_for_gpu(cuda_device)
    with torch.no_grad():
        model.eval()

        iterator = data_iterator(instances, num_epochs=1, shuffle=False)
        logger.info("Iterating over dataset")
        generator_tqdm = tqdm.tqdm(iterator, total=data_iterator.get_num_batches(instances))

        eval_loss = 0
        nb_batches = 0
        for tensor_batch in generator_tqdm:
            nb_batches += 1

            train_stages = ["stm", "sd", "valid"]
            task_index = TASKS_NAME.index(task_name)
            tensor_batch['task_index'] = torch.tensor(task_index)
            tensor_batch["reverse"] = torch.tensor(False)
            tensor_batch['for_training'] = torch.tensor(False)
            train_stage = train_stages.index("stm")
            tensor_batch['train_stage'] = torch.tensor(train_stage)
            tensor_batch = move_to_device(tensor_batch, 0)

            eval_output_dict = model.forward(**tensor_batch)
            loss = eval_output_dict["loss"]
            eval_loss += loss.item()
            metrics = model.get_metrics(task_name=task_name)
            metrics["stm_loss"] = float(eval_loss / nb_batches)

            description = training_util.description_from_metrics(metrics)
            generator_tqdm.set_description(description, refresh=False)

        metrics = model.get_metrics(task_name=task_name, reset=True)
        metrics["stm_loss"] = float(eval_loss / nb_batches)
        return metrics


if __name__ == "__main__":
    ### Evaluate from args ###

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--serialization_dir", required=True, help="Directory in which to save the model and its logs.", type=str
    )
    parser.add_argument(
        "-g",
        "--gold_mentions",
        action="store_true",
        required=False,
        default=False,
        help="Whether or not evaluate using gold mentions in coreference",
    )
    args = parser.parse_args()

    params = Params.from_file(params_file=os.path.join(args.serialization_dir, "config.json"))

    ### Instantiate tasks ###
    task_list = []
    task_keys = [key for key in params.keys() if re.search("^task_", key)]

    for key in task_keys:
        logger.info("Creating {}", key)
        task_params = params.pop(key)
        task_description = task_params.pop("task_description")
        task_data_params = task_params.pop("data_params")

        task = Task.from_params(params=task_description)
        task_list.append(task)

        _, _ = task.load_data_from_params(params=task_data_params)

    ### Load Vocabulary from files ###
    vocab = Vocabulary.from_files(os.path.join(args.serialization_dir, "vocabulary"))
    logger.info("Vocabulary loaded")

    ### Load the data iterators ###
    task_list = create_and_set_iterators(params=params, task_list=task_list, vocab=vocab)

    ### Regularization	###
    regularizer = None

    ### Create model ###
    model_params = params.pop("model")
    model = Model.from_params(vocab=vocab, params=model_params, regularizer=regularizer)

    ### Real evaluation ###
    cuda_device = params.pop("multi_task_trainer").pop_int("cuda_device", -1)

    metrics = {task._name: {} for task in task_list}
    for task in task_list:
        if not task._evaluate_on_test:
            continue

        logger.info("Task {} will be evaluated using the best epoch weights.", task._name)
        assert (
                task._test_data is not None
        ), "Task {} wants to be evaluated on test dataset but no there is no test data loaded.".format(task._name)

        logger.info("Loading the best epoch weights for tasks {}", task._name)
        best_model_state_path = os.path.join(args.serialization_dir, "best_{}.th".format(task._name))
        best_model_state = torch.load(best_model_state_path)
        best_model = model
        best_model.load_state_dict(state_dict=best_model_state)

        test_metric_dict = {}

        avg_accuracy = 0.0

        for pair_task in task_list:
            if not pair_task._evaluate_on_test:
                continue

            logger.info("Pair tasks {} is evaluated with the best model for {}", pair_task._name, task._name)
            test_metric_dict[pair_task._name] = {}
            test_metrics = evaluate(
                model=best_model,
                task_name=pair_task._name,
                instances=pair_task._test_data,
                data_iterator=pair_task._data_iterator,
                cuda_device=cuda_device,
            )

            for metric_name, value in test_metrics.items():
                test_metric_dict[pair_task._name][metric_name] = value

            avg_accuracy += test_metrics["accuracy"]

        logger.info("***** Average accuracy is %f *****", avg_accuracy / 3)

        metrics[task._name]["test"] = deepcopy(test_metric_dict)
        logger.info("Finished evaluation of tasks {}.", task._name)

    metrics_json = json.dumps(metrics, indent=2)
    with open(os.path.join(args.serialization_dir, "evaluate_metrics.json"), "w") as metrics_file:
        metrics_file.write(metrics_json)

    logger.info("Metrics: {}", metrics_json)
