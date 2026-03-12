from typing import Optional

import torch
import torch.nn.functional as F
import torch_scatter as scatter
from einops import einsum
from torch import Tensor, nn
from torch_geometric.nn import Linear

from .ssm import InitStrategy



class SSMTemporalGate(nn.Module):
    """An alternative approach to time encoding"""

    def __init__(self,
                 d_input,
                 d_state,
                 init_strategy:
                 Optional[InitStrategy] = None,
                 time_unit: int = 86400
                 ):
        super().__init__()
        self.time_unit = time_unit
        self.init_strategy = init_strategy or InitStrategy()
        # mimo impl
        # log(-A) in the diagonal form
        self.log_nA = nn.Parameter(torch.empty(d_state)) 
        self.B = nn.Parameter(torch.empty(d_input, d_state))
        self.C = nn.Parameter(torch.empty(d_state, d_state))
        self.reset_parameters()
        self.state_map = Linear(d_input, d_state)
        self.output_map = Linear(d_input, d_state)

    def reset_parameters(self):
        self.init_strategy.init(self.log_nA, self.B, self.C)

    def forward(self, delta_t, x):
        """Compute a temporal gate applied to the input

        Args:
            delta_t: time difference of shape [E]
            x: gathered tensor of shape [E, d]

        Returns:
            x * gate(x, delta)
        """
        state = self.state_map(x)
        delta_t = delta_t / self.time_unit
        A = - torch.exp(self.log_nA) 
        A_zoh = torch.exp(einsum(delta_t, A, 'e, d_s -> e d_s'))
        B_x = einsum(delta_t, self.B, x, 'e, d_in d_s, e d_in -> e d_s')
        gate = F.selu((A_zoh * state + B_x) @ self.C)
        return gate * self.output_map(x)


class GatedTemporalGraphAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: Optional[int] = None,
        heads: int = 1,
        dropout: float = 0.,
        residual: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim = edge_dim or 0
        self.heads = heads
        self.residual = residual
        if residual and in_channels != out_channels:
            self.lin_res = Linear(in_channels, out_channels, bias=False)
        else:
            self.lin_res = None

        self.d_k = out_channels // self.heads
        self.scale = self.d_k**(-0.5)

        # No time encoder needed, explicitly use temporal gating
        self.query_dim = in_channels
        self.key_dim = in_channels + edge_dim

        self.q_linears = nn.Sequential(Linear(self.query_dim, out_channels),
                                       nn.ReLU())
        self.k = SSMTemporalGate(self.key_dim, out_channels)
        self.v = SSMTemporalGate(self.key_dim, out_channels)

        self.merger = MLP(
            [in_channels + out_channels, in_channels, out_channels], norm=None,
            dropout=dropout)

    def reset_parameters(self):
        for lin in self.q_linears:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()

        self.merger.reset_parameters()
        if self.lin_res is not None:
            self.lin_res.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_attr: Optional[Tensor],
    ) -> Tensor:
        """"""
        node_i = edge_index[0]
        node_j = edge_index[1]
        x_i = x[node_i]
        x_j = x[node_j]

        source_node_vec = x_i
        target_node_vec = x_j
        if edge_attr is not None: 
            target_node_vec = torch.cat([target_node_vec, edge_attr], dim=1)

        q_mat = torch.reshape(self.q_linears(source_node_vec),
                              [-1, self.heads, self.d_k])  # [T, N , D]
        k_mat = torch.reshape(F.relu(self.k(edge_time, target_node_vec)), 
                              [-1, self.heads, self.d_k])
        v_mat = torch.reshape(self.v(edge_time, target_node_vec), 
                              [-1, self.heads, self.d_k])

        res_att = torch.sum(torch.multiply(q_mat, k_mat),
                            dim=-1) * self.scale  # [T, N]

        scores = self.scatter_softmax(res_att, node_i)

        v = torch.multiply(torch.unsqueeze(scores, dim=2), v_mat)
        v = torch.reshape(v, [-1, self.out_channels])

        out = scatter.scatter_add(v, node_i, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        out = self.merger(out)
        if self.residual:
            if self.lin_res is not None:
                residual = self.lin_res(x)
            else:
                residual = x
            out = out + residual
        return out

    def scatter_softmax(self, res_att: Tensor, node_i: Tensor) -> Tensor:
        n_head = self.heads
        scores = torch.zeros_like(res_att)
        for i in range(n_head):
            scores[:, i] = scatter.composite.scatter_softmax(
                res_att[:, i], node_i)

        return scores

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


from torch_geometric.nn.conv.gcn_conv import gcn_norm
class GatedTemporalLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: Optional[int] = None,
        heads: int = 1,
        dropout: float = 0.,
        residual: bool = False,
        aggr='gcn',
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim = edge_dim or 0
        self.heads = heads
        self.residual = residual
        if residual and in_channels != out_channels:
            self.lin_res = Linear(in_channels, out_channels, bias=False)
        else:
            self.lin_res = None

        self.d_k = out_channels // self.heads
        self.scale = self.d_k**(-0.5)

        self.query_dim = in_channels
        self.key_dim = in_channels + edge_dim
        
        self.q_linears = nn.Sequential(Linear(self.query_dim, out_channels),
                                       nn.ReLU())
        self.k = SSMTemporalGate(self.key_dim, out_channels)
        self.v = SSMTemporalGate(self.key_dim, out_channels)
        self.merger = MLP(
            [in_channels + out_channels, in_channels, out_channels], norm=None,
            dropout=dropout)

        self.aggr = aggr # gcn, mean, max

    def reset_parameters(self):
        for lin in self.q_linears:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()

        self.merger.reset_parameters()
        if self.lin_res is not None:
            self.lin_res.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_attr: Optional[Tensor],
    ) -> Tensor:
        """"""
        node_i = edge_index[0]
        node_j = edge_index[1]
        x_i = x[node_i]
        x_j = x[node_j]

        source_node_vec = x_i
        target_node_vec = x_j
        if edge_attr is not None: 
            target_node_vec = torch.cat([target_node_vec, edge_attr], dim=1)

        v_mat = self.v(edge_time, target_node_vec)
        if self.aggr=='gcn':
            edge_index, scores = gcn_norm(edge_index, add_self_loops=False, flow='target_to_source')
            v = torch.multiply(torch.unsqueeze(scores, dim=1), v_mat)
            out = scatter.scatter_add(v, node_i, dim=0, dim_size=x.size(0))
        elif self.aggr=='mean':
            out = scatter.scatter_mean(v_mat, node_i, dim=0, dim_size=x.size(0))
        elif self.aggr=='max':
            out = scatter.scatter_max(v_mat, node_i, dim=0, dim_size=x.size(0))[0]
        out = torch.cat([x, out], dim=1)
        out = self.merger(out)
        if self.residual:
            if self.lin_res is not None:
                residual = self.lin_res(x)
            else:
                residual = x
            out = out + residual
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
    



from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import Identity
import torch_scatter as scatter
from torch_geometric.nn import Linear
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
class TGATConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_encoder: nn.Module,
        time_dim: int,
        edge_dim: Optional[int] = None,
        heads: int = 1,
        dropout: float = 0.,
        residual: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim = edge_dim or 0
        self.time_dim = time_dim
        self.time_encoder = time_encoder
        self.heads = heads
        self.residual = residual
        if residual and in_channels != out_channels:
            self.lin_res = Linear(in_channels, out_channels, bias=False)
        else:
            self.lin_res = None

        self.d_k = out_channels // self.heads
        self.scale = self.d_k**(-0.5)
        self.query_dim = in_channels + time_dim 
        self.key_dim = in_channels + time_dim + edge_dim

        self.q_linears = nn.Sequential(Linear(self.query_dim, out_channels),
                                       nn.ReLU())
        self.k_linears = nn.Sequential(Linear(self.key_dim, out_channels),
                                       nn.ReLU())
        self.v_linears = Linear(self.key_dim, out_channels)

        self.merger = MLP(
            [in_channels + out_channels, in_channels, out_channels], norm=None,
            dropout=dropout)

    def reset_parameters(self):
        for lin in self.q_linears:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()
        for lin in self.k_linears:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()
        self.v_linears.reset_parameters()
        self.merger.reset_parameters()
        if self.lin_res is not None:
            self.lin_res.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_attr: Optional[Tensor],
    ) -> Tensor:
        """"""
        node_i = edge_index[0]
        node_j = edge_index[1]
        x_i = x[node_i]
        x_j = x[node_j]

        src_time = self.time_encoder(x_i.new_zeros(x_i.size(0), 1))
        edge_time = self.time_encoder(edge_time.unsqueeze(1))
        source_node_vec = torch.cat([x_i, src_time], dim=1)
        target_node_vec = torch.cat([x_j, edge_time], dim=1) 
        if edge_attr is not None:
            target_node_vec = torch.cat([target_node_vec, edge_attr], dim=1)

        q_mat = torch.reshape(self.q_linears(source_node_vec),
                              [-1, self.heads, self.d_k])  # [T, N , D]
        k_mat = torch.reshape(self.k_linears(target_node_vec),
                              [-1, self.heads, self.d_k])  # [T, N , D]
        v_mat = torch.reshape(self.v_linears(target_node_vec),
                              [-1, self.heads, self.d_k])  # [T, N , D]

        res_att = torch.sum(torch.multiply(q_mat, k_mat),
                            dim=-1) * self.scale  # [T, N]

        scores = self.scatter_softmax(res_att, node_i)

        v = torch.multiply(torch.unsqueeze(scores, dim=2), v_mat)
        v = torch.reshape(v, [-1, self.out_channels])

        out = scatter.scatter_add(v, node_i, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        out = self.merger(out)
        if self.residual:
            if self.lin_res is not None:
                residual = self.lin_res(x)
            else:
                residual = x
            out = out + residual
        return out

    def scatter_softmax(self, res_att: Tensor, node_i: Tensor) -> Tensor:
        n_head = self.heads
        scores = torch.zeros_like(res_att)
        for i in range(n_head):
            scores[:, i] = scatter.composite.scatter_softmax(
                res_att[:, i], node_i)

        return scores

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class TimeEncoder(nn.Module):
    """Time Encoding proposed by TGAT."""
    def __init__(self, dimension: int, time_unit: int = 86400, trainable=True):
        super().__init__()
        self.dimension = dimension
        self.time_unit = time_unit
        self.w = Linear(1, dimension)
        self.dimension = dimension
        self.trainable = trainable

        self.reset_parameters()

    def reset_parameters(self):
        w = 1 / 10**np.linspace(0, 1.5, self.dimension)
        w = torch.from_numpy(w).view(self.dimension, -1)
        self.w.weight = nn.Parameter(w.to(torch.float))
        self.w.bias = nn.init.zeros_(self.w.bias)
        
        if not self.trainable:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False            

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        out = torch.cos(self.w(t))
        return out



class MLP(torch.nn.Module):
    def __init__(
        self,
        channel_list: Optional[Union[List[int], int]] = None,
        *,
        in_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: Optional[int] = None,
        dropout: float = 0.,
        act: Union[str, Callable, None] = "relu",
        act_first: bool = False,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Union[str, Callable, None] = None,
        norm_kwargs: Optional[Dict[str, Any]] = None,
        bias: Union[bool, List[bool]] = True,
        **kwargs,
    ):
        super().__init__()

        act_first = act_first or kwargs.get("relu_first", False)

        if isinstance(channel_list, int):
            in_channels = channel_list

        if in_channels is not None:
            if num_layers is None:
                raise ValueError("Argument `num_layers` must be given")
            if num_layers > 1 and hidden_channels is None:
                raise ValueError(f"Argument `hidden_channels` must be given "
                                 f"for `num_layers={num_layers}`")
            if out_channels is None:
                raise ValueError("Argument `out_channels` must be given")

            channel_list = [hidden_channels] * (num_layers - 1)
            channel_list = [in_channels] + channel_list + [out_channels]

        assert isinstance(channel_list, (tuple, list))
        assert len(channel_list) >= 2
        self.channel_list = channel_list

        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.act_first = act_first

        if isinstance(bias, bool):
            bias = [bias] * (len(channel_list) - 1)
        if len(bias) != len(channel_list) - 1:
            raise ValueError(
                f"Number of bias values provided ({len(bias)}) does not match "
                f"the number of layers specified ({len(channel_list)-1})")

        self.lins = torch.nn.ModuleList()
        iterator = zip(channel_list[:-1], channel_list[1:], bias)
        for in_channels, out_channels, _bias in iterator:
            self.lins.append(Linear(in_channels, out_channels, bias=_bias))

        self.norms = torch.nn.ModuleList()
        iterator = channel_list[1:-1]  # Do not add norm layer at the end
        for hidden_channels in iterator:
            if norm is not None:
                norm_layer = normalization_resolver(
                    norm,
                    hidden_channels,
                    **(norm_kwargs or {}),
                )
            else:
                norm_layer = Identity()
            self.norms.append(norm_layer)

        self.dropout = torch.nn.Dropout(dropout)
        self.reset_parameters()

    @property
    def in_channels(self) -> int:
        r"""Size of each input sample."""
        return self.channel_list[0]

    @property
    def out_channels(self) -> int:
        r"""Size of each output sample."""
        return self.channel_list[-1]

    @property
    def num_layers(self) -> int:
        r"""The number of layers."""
        return len(self.channel_list) - 1

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        for lin, norm in zip(self.lins[:-1], self.norms):
            x = lin(x)
            if self.act is not None and self.act_first:
                x = self.act(x)
            x = norm(x)
            if self.act is not None and not self.act_first:
                x = self.act(x)
            x = self.dropout(x)
        x = self.lins[-1](x)
        return x
