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

        self.fc_layers = nn.ModuleList(
            [
                nn.Linear(obs_shape[0], hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, feature_dim),
            ]
        )

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


class CNNEncoder(nn.Module):
    """
    CNN encoder for image states
    """

    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        """
        Args:
            obs_shape (tuple): Shape of the observation [C, H, W]
        """
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList([nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)])
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = {2: 61}[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.0
        self.outputs["obs"] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs["conv1"] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs["conv%s" % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs["fc"] = h_fc

        out = self.ln(h_fc)
        self.outputs["ln"] = out

        return out

    def copy_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram("train_encoder/%s_hist" % k, v, step)
            if len(v.shape) > 2:
                L.log_image("train_encoder/%s_img" % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param("train_encoder/conv%s" % (i + 1), self.convs[i], step)
        L.log_param("train_encoder/fc", self.fc, step)
        L.log_param("train_encoder/ln", self.ln, step)
