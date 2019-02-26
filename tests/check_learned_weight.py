# -*- coding: utf-8 -*-
import torch
from allennlp.common import Params
from allennlp.models import Model
from allennlp.nn import RegularizerApplicator
from allennlp.nn.util import device_mapping

from MTL.train import tasks_and_vocab_from_params

serialization_dir = "../data/out/"

params = Params.from_file(params_file="../mtl/configs/dsp_srl.json")
regularizer = RegularizerApplicator.from_params(params.pop("regularizer", []))
tasks, vocab = tasks_and_vocab_from_params(params=params, serialization_dir=serialization_dir)
### Create model ###
model_params = params.pop("model")
model = Model.from_params(vocab=vocab, params=model_params, regularizer=regularizer)

model_path = "../data/out/model_state.th"
model_state = torch.load(model_path, map_location=device_mapping(-1))

model.load_state_dict(model_state)

for name, parameter in model.named_parameters():
    print(name)
