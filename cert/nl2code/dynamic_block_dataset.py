# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import List
from dataclasses import dataclass
import numpy as np
from datetime import datetime

import torch
from torch.utils.data.dataset import Dataset

@dataclass
class BlockSpan:
    index: int
    start: int
    end: int

    @property
    def length(self):
        return self.end - self.start + 1

@dataclass
class BlockItem:
    spans: List[BlockSpan]

    def __len__(self):
        return len(self.spans)

    def pad(self, item):
        self.spans += item.spans

    @property
    def size(self):
        return sum([x.length for x in self.spans])

class BlockCache:
    def __init__(self, block_size: int, cache_size: int):
        self.cache_size = cache_size
        self.block_size = block_size

        self.len2spans = [[] for _ in range(block_size)]
        self.items_count = 0
        self.length_counts = np.zeros(block_size, dtype=np.int32)

    def add_one(self, span: BlockSpan):
        length = span.length
        if length >= self.block_size:
            raise ValueError("Can't add one item with length {} >= block size {}.".format(length, self.block_size))

        self.len2spans[length].append(span)
        self.items_count += 1
        self.length_counts[length] += 1

    def is_full(self):
        return self.items_count >= self.cache_size

    def __len__(self):
        return self.items_count

    def _pop_one_by_length(self, sel_length: int) -> BlockItem:
        assert len(self.len2spans[sel_length]) == self.length_counts[sel_length]

        if len(self.len2spans[sel_length]) == 0:
            raise ValueError("Pop from empty length spans: {}".format(sel_length))

        self.length_counts[sel_length] -= 1
        sel_span = self.len2spans[sel_length].pop()
        pad_length = self.block_size - sel_length

        while pad_length > 0:
            # Find a perfect one to block
            if len(self.len2spans[pad_length]) > 0:
                pad_span = self.len2spans[pad_length].pop()
                self.length_counts[pad_length] -= 1
                block_item = BlockItem(spans=[sel_span, pad_span])
                self.items_count -= 2
                return block_item

            pad_length -= 1

        # can't find one to pad
        self.items_count -= 1
        return BlockItem(spans=[sel_span])

    def pop_one(self):
        sel_length = np.argmax(self.length_counts)
        return self._pop_one_by_length(sel_length)

    def pop_all(self):
        index = self.block_size - 1
        while index > 0:
            while self.len2spans[index]:
                yield self._pop_one_by_length(index)
            index -= 1

class DynamicBlockDataset(Dataset):
    def __init__(self, src_dataset: Dataset, src_sizes: List[int], block_size: int, dynamic_factor: int=10) -> None:
        super().__init__()
        self.src_dataset = src_dataset
        self.src_sizes = src_sizes
        self.block_size = block_size
        self.dynamic_factor = dynamic_factor

        start = datetime.now()
        self.block_items: List[BlockItem] = self.build_block_index_mappings()
        self._block_sizes = [x.size for x in self.block_items]
        print("DynamicBlockDataset builds block indices over, {} => {} ({:.4f}), avg examples = {:.3f}, cost = {}.".format(
            len(self.src_dataset),
            len(self.block_items),
            self.get_block_ratio(),
            np.mean([len(x) for x in self.block_items]),
            datetime.now() - start
        ))

    @property
    def sizes(self):
        return self._block_sizes

    def size(self, index) -> int:
        return self.block_items[index].size

    def get_block_ratio(self) -> float:
        print(np.mean(self._block_sizes), np.mean(self._block_sizes) / self.block_size)
        return sum(self.sizes) / len(self.block_items) / self.block_size

    def __len__(self):
        return len(self.block_items)

    def __getitem__(self, index) -> torch.Tensor:
        item = self.block_items[index]
        tensors = [self.src_dataset[span.index][span.start:span.end+1] for span in item.spans]
        return torch.cat(tensors, dim=0)

    def build_block_index_mappings(self):
        cache = BlockCache(self.block_size, self.dynamic_factor * self.block_size)
        block_idx_items = []

        for i, size in enumerate(self.src_sizes):
            start = 0
            while start < size:
                end = min(size, start + self.block_size)
                span = BlockSpan(index=i, start=start, end=end-1)

                if span.length == self.block_size:
                    block_idx_items.append(BlockItem([span]))
                else:
                    # Pop one if cache is full
                    if cache.is_full():
                        block_idx_items.append(cache.pop_one())
                    cache.add_one(span)
                start = end

        for item in cache.pop_all():
            block_idx_items.append(item)

        return block_idx_items
