import os
import shutil
import struct
from functools import lru_cache

import numpy as np
import torch

from fairseq.data import TokenBlockDataset
from torch.utils.data.dataset import Dataset

from typing import Union

_code_to_dtype = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float,
    7: np.double,
    8: np.uint16,
    9: np.uint32,
    10: np.uint64,
}

def best_fitting_int_dtype(
    max_int_to_represent,
) -> Union[np.uint16, np.uint32, np.int64]:

    if max_int_to_represent is None:
        return np.uint32  # Safe guess
    elif max_int_to_represent < 65500:
        return np.uint16
    elif max_int_to_represent < 4294967295:
        return np.uint32
    else:
        return np.int64
        # we avoid np.uint64 because it doesn't save space and its type promotion behaves unexpectedly
        # https://github.com/numpy/numpy/issues/5745

def _dtype_header_code(dtype) -> int:
    for k in _code_to_dtype.keys():
        if _code_to_dtype[k] == dtype:
            return k
    raise ValueError(dtype)

def _warmup_mmap_file(path):
    with open(path, "rb") as stream:
        while stream.read(100 * 1024 * 1024):
            pass

def index_file_path(prefix_path):
    return prefix_path + ".idx"


def data_file_path(prefix_path):
    return prefix_path + ".bin"

class MMapIndexedDataset(torch.utils.data.Dataset):
    class Index:
        _HDR_MAGIC = b"MMIDIDX\x00\x00"

        @classmethod
        def writer(cls, path, dtype):
            class _Writer:
                def __enter__(self):
                    self._file = open(path, "wb")

                    self._file.write(cls._HDR_MAGIC)
                    self._file.write(struct.pack("<Q", 1))
                    self._file.write(struct.pack("<B", _dtype_header_code(dtype)))

                    return self

                @staticmethod
                def _get_pointers(sizes):
                    dtype_size = dtype().itemsize
                    address = 0
                    pointers = []

                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size

                    return pointers

                def write(self, sizes):
                    pointers = self._get_pointers(sizes)

                    self._file.write(struct.pack("<Q", len(sizes)))

                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order="C"))
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order="C"))
                    del pointers

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path):
            with open(path, "rb") as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    "Index file doesn't match expected format. "
                    "Make sure that --dataset-impl is configured properly."
                )
                version = struct.unpack("<Q", stream.read(8))
                assert (1,) == version

                (dtype_code,) = struct.unpack("<B", stream.read(1))
                self._dtype = _code_to_dtype[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack("<Q", stream.read(8))[0]
                offset = stream.tell()

            _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode="r", order="C")
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            self._sizes = np.frombuffer(
                self._bin_buffer, dtype=np.int32, count=self._len, offset=offset
            )
            self._pointers = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._len,
                offset=offset + self._sizes.nbytes,
            )

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path):
        self._path = path
        self._index = self.Index(index_file_path(self._path))

        _warmup_mmap_file(data_file_path(self._path))
        self._bin_buffer_mmap = np.memmap(
            data_file_path(self._path), mode="r", order="C"
        )
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        del self._index

    def __len__(self):
        return len(self._index)

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        ptr, size = self._index[i]
        np_array = np.frombuffer(
            self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr
        )
        if self._index.dtype != np.int64:
            np_array = np_array.astype(np.int64)

        return torch.from_numpy(np_array)

    @property
    def sizes(self):
        return self._index.sizes

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return os.path.exists(index_file_path(path)) and \
            os.path.exists(data_file_path(path))

class MMapIndexedDatasetBuilder:
    def __init__(self, out_file, dtype=np.int64):
        self._data_file = open(out_file, "wb")
        self._dtype = dtype
        self._sizes = []

    def add_item(self, tensor):
        if isinstance(tensor, torch.Tensor):
            np_array = np.array(tensor.numpy(), dtype=self._dtype)
        elif isinstance(tensor, list):
            np_array = np.array(tensor, dtype=self._dtype)
        elif isinstance(tensor, np.ndarray):
            np_array = tensor
        else:
            raise NotImplementedError(type(tensor))

        self._data_file.write(np_array.tobytes(order="C"))
        self._sizes.append(np_array.size)

    def merge_file_(self, another_file):
        # Concatenate index
        index = MMapIndexedDataset.Index(index_file_path(another_file))
        assert index.dtype == self._dtype

        for size in index.sizes:
            self._sizes.append(size)

        # Concatenate data
        with open(data_file_path(another_file), "rb") as f:
            shutil.copyfileobj(f, self._data_file)

    def finalize(self, index_file):
        self._data_file.close()

        with MMapIndexedDataset.Index.writer(index_file, self._dtype) as index:
            index.write(self._sizes)

def make_builder(out_file, impl, vocab_size=None):
    if impl == "mmap":
        return MMapIndexedDatasetBuilder(
            out_file, dtype=best_fitting_int_dtype(vocab_size)
        )
    else:
        raise NotImplementedError(impl)


class DatasetForCLM(Dataset):
    def __init__(self, dataset: MMapIndexedDataset, block_size: int) -> None:
        super().__init__()
        self.block_dataset = TokenBlockDataset(
            dataset=dataset,
            sizes=dataset.sizes,
            block_size=block_size,
            pad=None,
            eos=None)

    @property
    def num_tokens(self):
        return np.sum(self.block_dataset.dataset.sizes)

    def __len__(self):
        return len(self.block_dataset)

    def __getitem__(self, idx: int):
        item = self.block_dataset[idx]
        return {
            'input_ids': item,
            'labels': item.clone()
        }

def load_indexed_dataset_for_clm(path: str, block_size: int):
    dataset = MMapIndexedDataset(path)
    return DatasetForCLM(dataset, block_size)