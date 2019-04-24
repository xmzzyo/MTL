# -*- coding: utf-8 -*-


# from apex.optimizers import FusedAdam
from allennlp.training.optimizers import Optimizer

# We just use the Pytorch optimizers, so here we force them into
# Registry._registry so we can build them from params.
# Registrable._registry[Optimizer] = {  # pylint: disable=protected-access
#     "fused_adam": FusedAdam
# }

# Optimizer.register("fused_adam")(FusedAdam)

