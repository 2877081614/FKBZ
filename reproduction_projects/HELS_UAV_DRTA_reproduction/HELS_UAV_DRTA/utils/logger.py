"""
训练日志工具 (TensorBoard + JSON)
"""
import os
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir, scenario_name, run_id=0):
        self.log_dir = log_dir
        self.scenario_name = scenario_name
        self.run_id = run_id
        self.writer = SummaryWriter(os.path.join(log_dir, f'{scenario_name}_{run_id}'))
        self.metrics_history = []

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_episode(self, episode, metrics):
        """Log all metrics for one episode"""
        for k, v in metrics.items():
            if np.isscalar(v):
                self.writer.add_scalar(f'Episode/{k}', v, episode)
        self.metrics_history.append({'episode': episode, **metrics})

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

    def close(self):
        self.writer.close()
