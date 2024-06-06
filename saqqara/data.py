import torch
import numpy as np
from torch.utils.data import Dataset


class NPYDataset(Dataset):
    def __init__(self, file_paths):
        """
        Args:
            file_paths (list of str): List of file paths to the .npy files.
        """
        self.file_paths = file_paths
        self.index_mapping = []
        self.total_length = 0

        # Pre-calculate the cumulative length of the arrays in all .npy files
        for file_path in self.file_paths:
            array_length = np.load(file_path, mmap_mode="r").shape[0]
            self.total_length += array_length
            self.index_mapping.append(
                (
                    file_path,
                    self.total_length - array_length,
                    self.total_length - 1,
                    array_length,
                )
            )
        self._test_array = np.zeros(self.total_length)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx < 0:
                idx = self.total_length + idx
            if idx < 0:
                raise IndexError
            # Find the file that contains the data for the requested index
            for file_path, start_idx, stop_idx, array_length in self.index_mapping:
                if idx >= start_idx and idx < start_idx + array_length:
                    # Calculate the index within the file
                    inner_idx = idx - start_idx
                    data = np.load(file_path, mmap_mode="c")[inner_idx]
                    return torch.from_numpy(data).float()
            # In case of an index out of bounds error
            raise IndexError("Index out of bounds")

        elif isinstance(idx, slice):
            # Accumulate batch data from multiple files if necessary
            start = idx.start or 0
            stop = idx.stop or self.total_length
            step = idx.step or 1
            if start < 0:
                start = self.total_length + start
            if stop < 0:
                stop = self.total_length + stop
            if stop > self.total_length:
                raise IndexError
            if start < 0:
                raise IndexError
            batch_arrays = []
            current_idx = start
            for file_path, start_idx, stop_idx, array_length in self.index_mapping:
                if current_idx >= start_idx and current_idx < start_idx + array_length:
                    inner_start_idx = current_idx - start_idx
                    if stop_idx >= stop:
                        inner_stop_idx = stop - current_idx + inner_start_idx
                        batch_arrays.append(
                            np.load(file_path, mmap_mode="c")[
                                inner_start_idx:inner_stop_idx:step
                            ]
                        )
                        break
                    else:
                        inner_stop_idx = array_length
                        batch_arrays.append(
                            np.load(file_path, mmap_mode="c")[
                                inner_start_idx:inner_stop_idx:step
                            ]
                        )
                    current_idx += step * batch_arrays[-1].shape[0]
            out = np.vstack(batch_arrays)
            if out.shape[0] != self._test_array[idx].shape[0]:
                raise IndexError
            return torch.from_numpy(out).float()

        elif isinstance(idx, tuple):
            batch_arrays = []
            for _idx in idx:
                if isinstance(_idx, int):
                    batch_arrays.append(self.__getitem__(_idx).unsqueeze(0))
                elif isinstance(_idx, slice):
                    batch_arrays.append(self.__getitem__(_idx))
            return torch.from_numpy(np.vstack(batch_arrays)).float()


# TODO: Implement a noise resampling class that grabs same signal but different
# noise each time __getitem__ is callled. Try on_after_load function?


class ResamplingDataset(Dataset, dict):
    def __init__(self, z_store, signal_store, tm_store, oms_store):
        self.datasets = {
            "z": z_store,
            "signal": signal_store,
            "tm": tm_store,
            "oms": oms_store,
        }
        self.z = z_store
        self.signal = signal_store
        self.tm = tm_store
        self.oms = oms_store
        self.total_length = z_store.total_length

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if type(idx) != str:
            d = {k: v[idx] for k, v in self.datasets.items()}
            return d
        else:
            return self.datasets[idx]


class TrainingDataset(Dataset, dict):
    def __init__(self, z_store, data_store):
        self.datasets = {
            "z": z_store,
            "data": data_store,
        }
        self.z = z_store
        self.data = data_store
        self.total_length = z_store.total_length
        if len(self.datasets["z"]) != len(self.datasets["data"]):
            raise ValueError("Stores must be the same size")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if type(idx) != str:
            d = {k: v[idx] for k, v in self.datasets.items()}
            return d
        else:
            return self.datasets[idx]
