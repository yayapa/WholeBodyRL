import random
from collections import defaultdict
from typing import List, Iterator
import numpy as np
import torch
from torch.utils.data import Sampler
import torch.distributed as dist
import math

import random
from collections import defaultdict
from itertools import cycle
from typing import List, Iterator, Union

class EventsBalancedBatchSampler(Sampler[List[int]]):
    """
    Produces batches that contain exactly one sample for every event label.
    Assumes N_events == batch_size.
    """
    def __init__(self, labels: List[int]):
        self.labels = np.asarray(labels)
        self.classes = np.unique(self.labels)
        assert len(self.classes) == 5, "Expecting 5 different events"

        # indices for every class
        self.class_to_indices = {
            c: np.where(self.labels == c)[0].tolist()
            for c in self.classes
        }
        self.batch_size = len(self.classes)              # 5
        self.num_batches = max(len(v) for v in self.class_to_indices.values())

    def __iter__(self) -> Iterator[List[int]]:
        # shuffle every epoch
        for idx_list in self.class_to_indices.values():
            random.shuffle(idx_list)

        batch_id = 0
        while batch_id < self.num_batches:
            batch = []
            for c in self.classes:
                idx_list = self.class_to_indices[c]

                # recycle indices if we already used the whole list
                pos = batch_id % len(idx_list)
                batch.append(idx_list[pos])
            batch_id += 1
            yield batch                                     # length == 5

    def __len__(self):
        return self.num_batches
    

class RandomDistributedSampler(Sampler):
    def __init__(self, dataset, num_samples=None, replacement=True):
        super().__init__(dataset)
        self.dataset = dataset
        self.num_samples = num_samples or len(dataset)
        self.replacement = replacement

        # Check if distributed is available and initialized
        if dist.is_available() and dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            print("distributed is not available and initialized")
            self.num_replicas = 1
            self.rank = 0

        self.num_samples_per_rank = math.ceil(self.num_samples / self.num_replicas)
        self.total_size = self.num_samples_per_rank * self.num_replicas
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)  # Ensuring reproducibility

        # Randomly sample indices from the entire dataset
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        # Ensure each rank gets a non-overlapping subset of indices
        indices = indices[self.rank:self.total_size:self.num_replicas]

        # Truncate to desired number of samples per rank
        return iter(indices[:self.num_samples_per_rank])

    def __len__(self):
        return self.num_samples_per_rank
    
class EventsBalancedSampler(torch.utils.data.Sampler[int]):
    def __init__(self, labels):
        self.labels = np.asarray(labels)
        classes = np.unique(self.labels)
        assert len(classes) == 5
        pools = {c: np.where(self.labels == c)[0].tolist() for c in classes}
        self.order = []
        # build the permutation once per epoch in __iter__
        self.pools = pools
    def __iter__(self):
        for p in self.pools.values():
            random.shuffle(p)
        # interleave class lists: c0_0,c1_0,...,c4_0, c0_1,c1_1,...
        for row in zip(*self.pools.values()):
            self.order.extend(row)
        return iter(self.order)
    def __len__(self):                # just the epoch length
        return len(next(iter(self.pools.values()))) * 5


class AnchorBalancedBatchSampler(Sampler[List[int]]):
    """
    Every batch:
        • contains exactly 1 sample whose label == anchor_class
        • fills the remaining (batch_size-1) slots with the other classes
          so that those classes are seen equally often over one epoch.
    """
    def __init__(
        self,
        labels: Union[List[int], np.ndarray],
        anchor_class: int = 0,
        batch_size: int = 5,
    ):
        self.labels = np.asarray(labels)
        self.anchor_class = anchor_class
        self.batch_size = batch_size
        assert batch_size >= 2, "Need room for anchor + at least one other class"

        # Split indices into “anchor” and “others by class”
        self.anchor_indices = np.where(self.labels == anchor_class)[0].tolist()
        self.other_classes = [c for c in np.unique(self.labels) if c != anchor_class]
        self.class_pools = {
            c: np.where(self.labels == c)[0].tolist() for c in self.other_classes
        }

        # Epoch length = how many anchor samples you have
        self.num_batches = len(self.anchor_indices)

    # ------------------------------------------------------------------ #
    def __iter__(self) -> Iterator[List[int]]:
        # fresh shuffle each epoch
        random.shuffle(self.anchor_indices)
        for idxs in self.class_pools.values():
            random.shuffle(idxs)

        # create infinite round-robin iterator over the non-anchor classes
        round_robin = cycle(self.other_classes)
        class_cursors = defaultdict(int)        # where we are inside each pool

        for b in range(self.num_batches):
            # 1) the anchor
            batch = [self.anchor_indices[b]]

            # 2) fill the rest
            while len(batch) < self.batch_size:
                cls = next(round_robin)
                pool = self.class_pools[cls]
                pos = class_cursors[cls] % len(pool)
                batch.append(pool[pos])
                class_cursors[cls] += 1           # move cursor (wraps naturally)

            yield batch                           # length == batch_size

    # ------------------------------------------------------------------ #
    def __len__(self):
        return self.num_batches



class AnchorBalancedSampler(Sampler[int]):
    """
    Emits a stream of indices such that, when the DataLoader groups them in
    `batch_size` chunks, every chunk contains **one** anchor-class example
    and `batch_size-1` examples from the other classes, which are balanced
    over the whole epoch.

    Example with batch_size=3 and anchor_class=0:
        stream:  0, 1, 2,   0, 3, 4,   0, 1, 2,   0, 3, 4, ...
        batches: [0,1,2], [0,3,4], [0,1,2], [0,3,4], ...
    """

    def __init__(
        self,
        labels: Union[List[int], np.ndarray],
        anchor_class: int = 0,
        batch_size: int = 5,
    ):
        super().__init__(None)
        assert batch_size >= 2, "Need room for anchor + at least one other class"

        self.labels = np.asarray(labels)
        self.anchor_class = anchor_class
        self.batch_size = batch_size

        # split indices into anchor / non-anchor pools
        self.anchor_idx = np.where(self.labels == anchor_class)[0].tolist()
        other_classes = [c for c in np.unique(self.labels) if c != anchor_class]
        self.other_pools = {
            c: np.where(self.labels == c)[0].tolist() for c in other_classes
        }

        # one batch per anchor sample
        self._epoch_batches = len(self.anchor_idx)

    # ------------------------------------------------------------------ #
    def __iter__(self) -> Iterator[int]:
        random.shuffle(self.anchor_idx)
        for pool in self.other_pools.values():
            random.shuffle(pool)

        # infinite round-robin iterator over non-anchor classes
        class_cycle = cycle(self.other_pools.keys())
        # cursor for each class pool
        cursors = {c: 0 for c in self.other_pools}

        # build the epoch stream
        for i in range(self._epoch_batches):
            # 1) yield the anchor
            yield self.anchor_idx[i]

            # 2) fill the rest of the batch
            needed = self.batch_size - 1
            while needed:
                cls = next(class_cycle)
                pool = self.other_pools[cls]
                # wrap around pool if exhausted
                pos = cursors[cls] % len(pool)
                yield pool[pos]
                cursors[cls] += 1
                needed -= 1

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        # total number of indices we yield = num_batches * batch_size
        return self._epoch_batches * self.batch_size


