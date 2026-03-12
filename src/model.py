from typing import Optional

from torch import Tensor, nn
from torch_geometric.nn import Linear
from torch_geometric.utils import scatter

from .layer import MLP, GatedTemporalGraphAttention,GatedTemporalLayer
from .layer import TGATConv,TimeEncoder
    
class GatedTGAT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        hidden_channels: int = 256,
        edge_dim: Optional[int] = None,
        heads: int = 1,
        num_layers: int = 2,
        dropout: float = 0.,
        residual: bool = True,
    ):
        super().__init__()
        self.dropout = dropout

        self.node_encoder = Linear(in_channels, hidden_channels)
        if edge_dim is not None:
            self.edge_encoder = Linear(edge_dim, hidden_channels)
        else:
            self.edge_encoder = None

        self.convs = nn.ModuleList([
            GatedTemporalGraphAttention(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                edge_dim=hidden_channels,
                heads=heads,
                dropout=dropout,
                residual=residual,
            ) for _ in range(num_layers)
        ])
        self.decoder = MLP([hidden_channels, 512, 256, 128, out_channels],
                           norm=None, dropout=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.decoder.reset_parameters()

    def encode(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:

        x = self.node_encoder(x)
        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)

        edge_time = time_difference(
            edge_time, edge_index, num_nodes=x.size(0)).squeeze() 

        for conv in self.convs:
            x = conv(x, edge_index, edge_time, edge_attr)
        return x

    def decode(self, src_emb, dst_emb: Optional[Tensor] = None) -> Tensor:
        if dst_emb is not None:
            return self.decoder(src_emb * dst_emb)
        return self.decoder(src_emb)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """"""
        x = self.encode(x, edge_index, edge_time, edge_attr)
        x = self.decode(x)
        return x


def time_difference(edge_time: Tensor, edge_index: Tensor,
                    num_nodes: Optional[int] = None) -> Tensor:
    """Get the time difference data on the edge.
    First obtain the minimum timestamp sent by each starting
      point, and then subtract
    the minimum timestamp of the starting point from the
      timestamp of each edge.
    """
    dst_node = edge_index[0]
    min_t = scatter(edge_time.view(-1), dst_node, reduce='min', dim=0,
                    dim_size=num_nodes)
    min_t = min_t[dst_node].view(-1, 1)
    delta_t = edge_time.view_as(min_t) - min_t
    return delta_t


class GatedTGNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        hidden_channels: int = 256,
        edge_dim: Optional[int] = None,
        heads: int = 1,
        num_layers: int = 2,
        dropout: float = 0.,
        residual: bool = True,
        aggr='gcn',
    ):
        super().__init__()
        self.dropout = dropout

        self.node_encoder = Linear(in_channels, hidden_channels)
        if edge_dim is not None:
            self.edge_encoder = Linear(edge_dim, hidden_channels)
        else:
            self.edge_encoder = None

        self.convs = nn.ModuleList([
            GatedTemporalLayer(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                edge_dim=hidden_channels,
                heads=heads,
                dropout=dropout,
                residual=residual,
                aggr=aggr,
            ) for _ in range(num_layers)
        ])
        self.decoder = MLP([hidden_channels, 512, 256, 128, out_channels],
                           norm=None, dropout=dropout) 
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.decoder.reset_parameters()

    def encode(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:

        x = self.node_encoder(x)
        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)

        edge_time = time_difference(
            edge_time, edge_index, num_nodes=x.size(0)).squeeze() 

        for conv in self.convs:
            x = conv(x, edge_index, edge_time, edge_attr)
        return x

    def decode(self, src_emb, dst_emb: Optional[Tensor] = None) -> Tensor:
        if dst_emb is not None:
            return self.decoder(src_emb * dst_emb)
        return self.decoder(src_emb)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """"""
        x = self.encode(x, edge_index, edge_time, edge_attr)
        x = self.decode(x)
        return x
    
class TGAT(nn.Module):
    def __init__( 
        self,
        in_channels: int,
        out_channels: int = 1,
        hidden_channels: int = 256,
        edge_dim: Optional[int] = None,
        heads: int = 1,
        num_layers: int = 2,
        dropout: float = 0.,
        residual: bool = True,
    ):
        super().__init__()
        self.dropout = dropout

        self.node_encoder = Linear(in_channels, hidden_channels)
        if edge_dim is not None:
            self.edge_encoder = Linear(edge_dim, hidden_channels)
        else:
            self.edge_encoder = None

        time_encoder = TimeEncoder(dimension=hidden_channels) 
        self.convs = nn.ModuleList([
            TGATConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                time_encoder=time_encoder,
                time_dim=hidden_channels, 
                edge_dim=hidden_channels,
                heads=heads,
                dropout=dropout,
                residual=residual,
            ) for _ in range(num_layers)
        ])
        self.decoder = MLP([hidden_channels, 512, 256, 128, out_channels],
                           norm=None, dropout=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.decoder.reset_parameters()

    def encode(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:

        x = self.node_encoder(x)
        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)

        edge_time = time_difference(
            edge_time, edge_index, num_nodes=x.size(0)).squeeze() 
        for conv in self.convs:
            x = conv(x, edge_index, edge_time, edge_attr) 

        return x

    def decode(self, src_emb, dst_emb: Optional[Tensor] = None) -> Tensor:
        if dst_emb is not None:
            return self.decoder(src_emb * dst_emb)
        return self.decoder(src_emb)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        """"""
        x = self.encode(x, edge_index, edge_time, edge_attr)
        x = self.decode(x)
        return x