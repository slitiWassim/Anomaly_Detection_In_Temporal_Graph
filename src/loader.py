from typing import List, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch_geometric.utils import index_sort
from torch_geometric.utils.sparse import index2ptr

from .temporal_data import TemporalData

class EventLoader(DataLoader):
    r"""A data loader which merges succesive events of a
    :class:`lastgl.data.temporal.TemporalData` to a mini-batch.

    Args:
        data (TemporalData): The temporal data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        data: TemporalData,
        num_neighbors: List[int],
        input_events: Optional[Tensor] = None,
        input_time: Optional[Tensor] = None,
        batch_size: int = 1,
        replace: bool = False,
        **kwargs,
    ):

        if input_events is None:
            src = data.src
            input_events = torch.arange(src.size(0), device=src.device)
            input_time = data.t if input_time is None else input_time
        else:
            input_time = data.t[
                input_events] if input_time is None else input_time
            if input_events.dtype == torch.bool:
                assert input_events.dim() == 1
                input_events = input_events.nonzero().view(-1)

        if not isinstance(input_events, list):
            input_events = [input_events]

        if not isinstance(input_time, list):
            input_time = [input_time]

        assert len(input_events) == len(input_time)
        for i in range(1, len(input_events)):
            assert input_events[i].size() == input_events[0].size()
            assert input_time[i].size() == input_time[0].size()

        self.num_neighbors = num_neighbors

        dataset = EventBasedSamplingDataset(data, nodes_or_events=input_events,
                                            input_time=input_time,
                                            num_neighbors=num_neighbors,
                                            replace=replace)
        super().__init__(dataset=dataset, batch_size=batch_size,
                         collate_fn=Collater(data), **kwargs)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num_neighbors})'


class NodeLoader(DataLoader):
    r"""A data loader which merges succesive events of a
    :class:`lastgl.data.temporal.TemporalData` to a mini-batch.

    Args:
        data (TemporalData): The temporal data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        data: TemporalData,
        num_neighbors: List[int],
        input_nodes: Tensor,
        input_time: Tensor,
        batch_size: int = 1,
        replace: bool = False,
        **kwargs,
    ):

        if not isinstance(input_nodes, list):
            input_nodes = [input_nodes]

        if not isinstance(input_time, list):
            input_time = [input_time]

        assert len(input_nodes) == len(input_time)
        for i in range(1, len(input_nodes)):
            assert input_nodes[i].size() == input_nodes[0].size()
            assert input_time[i].size() == input_time[0].size()

        self.num_neighbors = num_neighbors

        dataset = NodeBasedSamplingDataset(data, nodes_or_events=input_nodes,
                                           input_time=input_time,
                                           num_neighbors=num_neighbors,
                                           replace=replace)
        super().__init__(dataset=dataset, batch_size=batch_size,
                         collate_fn=Collater(data), **kwargs)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num_neighbors})'


class SamplingDataset(Dataset):
    def __init__(
        self,
        data: TemporalData,
        num_neighbors: List[int],
        nodes_or_events: Optional[Tensor] = None,
        input_time: Optional[Tensor] = None,
        replace: bool = False,
    ):
        super().__init__()
        src, perm = index_sort(data.src, max_value=data.num_nodes)
        rowptr = index2ptr(src, data.num_nodes)
        self.src_time = data.t[perm]
        self.perm = perm
        self.rowptr = rowptr

        self.nodes_or_events = nodes_or_events
        self.input_time = input_time
        self.data = data
        self.num_neighbors = num_neighbors
        self.replace = replace

    def subgraph_events(self, root: int, time: float):
        root = [root]
        time = [time]
        event_ids = []
        for num_neighbors in self.num_neighbors:
            event_id = self.sample(root, time, num_neighbors)
            if len(event_id) == 0:
                # stop sampling at the first step with empty neighbors
                break
            root = self.data.dst[event_id]
            event_ids.append(event_id)
            time = self.data.t[event_id]
        if len(event_ids) > 0:
            event_ids = torch.cat(event_ids, dim=0)
        else:
            event_ids = torch.empty(0, dtype=torch.long)
        return event_ids

    def sample(self, src, time, num_neighbors):
        event_ids = []

        rowptr = self.rowptr
        perm = self.perm
        src_time = self.src_time

        for root, root_time in zip(src, time):
            event_id = self.sample_from_ptr(root, rowptr, perm, root_time,
                                            src_time)
            num_events = event_id.size(0)
            if num_events == 0:
                continue
            if num_neighbors < 0:
                ...  # no sampling
            else:
                if num_neighbors <= num_events or not self.replace:
                    # TODO (jintang): sample the time-neast neighbors?
                    random_samples = torch.randperm(num_events)[:num_neighbors]
                    event_id = event_id[random_samples]
                else:
                    random_sampled = torch.randint(
                        0, num_events, size=(num_neighbors - num_events, ))
                    event_id = torch.cat([event_id, event_id[random_sampled]],
                                         dim=0)

            event_ids.append(event_id)

        if len(event_ids) > 0:
            event_ids = torch.cat(event_ids, dim=0)
        return event_ids

    @staticmethod
    def sample_from_ptr(root, ptr, perm, root_time, time):
        if ptr[root + 1] == ptr[root]:
            return ptr.new_empty(0)
        slice_ = slice(ptr[root], ptr[root + 1])
        mask = time[slice_] <= root_time
        event_id = perm[slice_][mask]
        return event_id

    def __len__(self) -> int:
        return self.nodes_or_events[0].size(0)

    def __getitem__(self, index: int) -> Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num_neighbors})'


class EventBasedSamplingDataset(SamplingDataset):
    def __getitem__(self, index: int) -> Tensor:
        event_ids = []
        num_trials = len(self.nodes_or_events)
        for i in range(num_trials):
            time = self.input_time[i][index]
            index = self.nodes_or_events[i][index]
            root = self.data.src[index]
            index = index.unsqueeze(0)
            event_id = self.subgraph_events(root=root, time=time)
            event_id = torch.cat([index, event_id])  # TODO: remove redudant
            event_ids.append(event_id)
        return event_ids


class NodeBasedSamplingDataset(SamplingDataset):
    def __getitem__(self, index: int) -> Tensor:
        event_ids = []
        num_trials = len(self.nodes_or_events)
        for i in range(num_trials):
            time = self.input_time[i][index]
            root = self.nodes_or_events[i][index]
            event_id = self.subgraph_events(root=root, time=time)
            event_ids.append(event_id)
        return event_ids


class Collater:
    def __init__(self, data):
        self.data = data

    def gather(self, batch):
        root_event = []
        event_ids = []
        for event_id in batch:
            if event_id.numel() > 0:
                root_event.append(event_id[:1]) 
                event_ids.append(event_id[1:])
        root_event = torch.cat(root_event)
        event_ids = torch.cat(event_ids)
        event_ids = torch.cat([root_event, event_ids])
        data = self.data[event_ids]

        # Relabel nodes
        unique, (src, dst) = torch.unique(data.edge_index, return_inverse=True)
        data.src, data.dst = src, dst

        for key, val in data.stores[0].items():
            if key in ['n_id', 'e_id', 'input_id']:
                continue
            if data.is_node_attr(key, val):
                data[key] = val[unique]

        data.n_id = unique
        data.e_id = event_ids
        data.input_id = root_event
        data.batch_size = root_event.size(0)
        return data

    def __call__(self, batch):
        first_batch = batch[0]
        outs = []
        for i in range(len(first_batch)):
            outs.append(self.gather([b[i] for b in batch]))
        if len(outs) == 1:
            outs, = outs
        return outs
