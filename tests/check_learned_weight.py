# -*- coding: utf-8 -*-
import os

import torch


model_name = "dsp_model"

serialization_dir = "../data/single_task_weights/"+model_name


model_path = os.path.join(serialization_dir, "weights.th")

# model_state = torch.load(model_path, map_location=device_mapping(-1))
# model.load_state_dict(model_state)


state_dict = torch.load(model_path)

for key, value in state_dict.items():
    print(key, value.shape)
