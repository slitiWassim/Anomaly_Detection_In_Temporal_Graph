
from collections import deque

import torch
from torch import nn
from torch_geometric.nn import Linear    

class History(nn.Module):
    def __init__(
        self,
        num_nodes,
        num_timeslots,
        dimension,
        device='cpu',
        history_retrieve="last",
        recurrent="gru",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.history_dimension = dimension
        self.num_timeslots = num_timeslots

        if recurrent == "rnn":
            self.recurrent_network = nn.RNNCell(
                input_size=dimension, hidden_size=dimension
            )
        elif recurrent == "gru":
            self.recurrent_network = nn.GRUCell(
                input_size=dimension, hidden_size=dimension 
            )
        elif recurrent == "lstm":
            self.recurrent_network = nn.LSTMCell(
                input_size=dimension, hidden_size=dimension
            )
        else:
            assert recurrent

        self.history_retrieve = history_retrieve

        self.history = deque(maxlen=num_timeslots)
        for _ in range(num_timeslots):
            self.history.append(
                torch.zeros(num_nodes, dimension,
                            requires_grad=False).to(device)
            )
        self.lin = nn.Sequential(
            Linear(-1, dimension),
            nn.ELU(),
            Linear(dimension, dimension),
        )
        self.recurrent = recurrent

        self.device = device
        
    def get_history(self, node_idxs):
        if self.history_retrieve == "last":
            return self.history[-1][node_idxs]
        elif self.history_retrieve == "mean":
            return torch.stack([m[node_idxs] for m in self.history], dim=0).mean(dim=0)
    
    def set_history(self, node_idxs, values):
        for i in range(len(self.history) - 1):
            self.history[i][node_idxs] = self.history[i + 1][node_idxs]

        self.history[-1][node_idxs] = values.data

    def forward(self, x, idx, update=True):
        x = self.lin(x)
        mem = self.get_history(idx)
        if self.recurrent == "lstm":
            out = self.recurrent_network(x, (mem, torch.zeros_like(mem)))[0]
        else:
            out = self.recurrent_network(x, mem)
        if update:
            self.set_history(idx, out)
        return out, mem
