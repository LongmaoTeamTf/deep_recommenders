#!/usr/bin/python3
# -*- coding: utf-8 -*-

from deep_recommenders.models.multi_task_learning.shared_bottom import tasks_tower
from deep_recommenders.models.multi_task_learning.shared_bottom import shared_bottom
from deep_recommenders.models.multi_task_learning.shared_bottom import shared_bottom_estimator
from deep_recommenders.models.multi_task_learning.mixture_of_experts import _synthetic_data as synthetic_data
from deep_recommenders.models.multi_task_learning.mixture_of_experts import synthetic_data_input_fn
from deep_recommenders.models.multi_task_learning.mixture_of_experts import one_gate as OMoE
from deep_recommenders.models.multi_task_learning.mixture_of_experts import multi_gate as MMoE
