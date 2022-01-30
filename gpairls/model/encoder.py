"""
Encoder from state to a feature vector.
"""

import torch
import torch.nn as nn

import config


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


class MLPEncoder(nn.Module):
    """
    MLP encoder for continuous states
    """

    def __init__(self, obs_shape, feature_dim, hidden_dim=4):
        super(MLPEncoder, self).__init__()

        self.obs_shape = obs_shape
        self.feature_dim = feature_dim

        self.fc_layers = nn.ModuleList([
            nn.Linear(obs_shape[0], hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, feature_dim),
        ])

        self.outputs = {}

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, obs, detach=False):
        x = obs
        for i, fc_layer in enumerate(self.fc_layers[:-1]):
            x = fc_layer(x)
            x = torch.relu(x)
            self.outputs[f"fc_{i+1}"] = x
        x = self.fc_layers[-1](x)
        self.outputs[f"fc_{len(self.fc_layers)}"] = x
        if detach:
            x = x.detach()
        return x

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram("train_encoder/%s_hist" % k, v, step)
            if len(v.shape) > 2:
                L.log_image("train_encoder/%s_img" % k, v[0], step)

        for i, layer in enumerate(self.fc_layers):
            L.log_param(f"train_encoder/fc_layer_{i+1}", layer, step)

    def copy_weights_from(self, source):
        """Tie layer weights"""

        for i in range(len(self.fc_layers)):
            tie_weights(source.fc_layers[i], self.fc_layers[i])