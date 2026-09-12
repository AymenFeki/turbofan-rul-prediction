"""Sequence networks. Input (batch, seq_len, n_features), output (batch,)."""

import math

import torch
import torch.nn as nn


class LSTMNet(nn.Module):
    def __init__(self, n_features, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(self.dropout(h[-1])).squeeze(-1)


class CNNNet(nn.Module):
    def __init__(self, n_features, channels=64, layers=2, kernel=5, dropout=0.2):
        super().__init__()
        blocks, in_ch = [], n_features
        for _ in range(layers):
            blocks += [
                nn.Conv1d(in_ch, channels, kernel, padding=kernel // 2),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
            ]
            in_ch = channels
        self.conv = nn.Sequential(*blocks, nn.AdaptiveAvgPool1d(1))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(channels, 1)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).squeeze(-1)
        return self.head(self.dropout(x)).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerNet(nn.Module):
    def __init__(self, n_features, d_model=64, heads=4, layers=2, ff_mult=2, dropout=0.1):
        super().__init__()
        if d_model % heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by heads ({heads})")
        self.proj = nn.Linear(n_features, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=ff_mult * d_model,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, return_attention=False):
        x = self.pos(self.proj(x))
        out = self.head(self.dropout(self.encoder(x).mean(dim=1))).squeeze(-1)
        if not return_attention:
            return out
        first = self.encoder.layers[0].self_attn
        _, attn = first(x, x, x, need_weights=True, average_attn_weights=True)
        return out, attn


NETWORKS = {"lstm": LSTMNet, "cnn": CNNNet, "transformer": TransformerNet}


def build_network(name, n_features, **params):
    if name not in NETWORKS:
        raise ValueError(f"unknown network '{name}', choose from {sorted(NETWORKS)}")
    return NETWORKS[name](n_features, **params)


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
