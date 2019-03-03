# coding: utf-8

"""
Various utilities that don't fit anwhere else.
"""

from typing import List, Dict

import torch
from allennlp.common.params import Params
from allennlp.data import Vocabulary
from allennlp.data.iterators import DataIterator
from torch import Tensor

from mtl.tasks import Task


def create_and_set_iterators(params: Params, task_list: List[Task], vocab: Vocabulary) -> List[Task]:
    """
    Each tasks/dataset can have its own specific data iterator. If not precised,
    we use a shared/common data iterator.
    
    Parameters
    ----------
    params: ``Params``
        A parameter object specifing an experiment.
    task_list: ``List[Task]``
        A list containing the tasks of the model to train.
        
    Returns
    -------
    task_list: ``List[Task]``
        The list containing the tasks of the model to train, where each tasks has a new attribute: the data iterator.
    """
    ### Charge default iterator ###
    iterators_params = params.pop("iterators")

    default_iterator_params = iterators_params.pop("iterator")
    default_iterator = DataIterator.from_params(default_iterator_params)
    default_iterator.index_with(vocab)

    ### Charge dataset specific iterators ###
    for task in task_list:
        specific_iterator_params = iterators_params.pop("iterator_" + task._name, None)
        if specific_iterator_params is not None:
            specific_iterator = DataIterator.from_params(specific_iterator_params)
            specific_iterator.index_with(vocab)
            task.set_data_iterator(specific_iterator)
        else:
            task.set_data_iterator(default_iterator)

    return task_list


def tensor2HalfTensor(tensor):
    if isinstance(tensor, Tensor) and tensor.type == torch.float:
        return tensor.half()
    if isinstance(tensor, Dict):
        for key, value in tensor.items():
            tensor[key] = tensor2HalfTensor(value)
        return tensor
    if isinstance(tensor, List):
        return [tensor2HalfTensor(t) for t in tensor]
    else:
        return tensor


def BN_convert_float(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module
