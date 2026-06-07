"""Distillation: gating network, projections, composite loss, trainer."""
from .gating import GatingNetwork, TeacherProjections
from .losses import (composite_loss, LossComponents,
                     task_loss, gated_kd_loss, gated_geometry_loss)
from .trainer import OFATrainer, TrainState, TrainHistory
