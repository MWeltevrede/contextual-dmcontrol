import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import cdmc.utils as utils
import cdmc.algorithms.modules as m
from cdmc.algorithms.sac import SAC


class RAD(SAC):
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
