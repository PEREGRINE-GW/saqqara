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


class RandomSamplingDataset(Dataset, dict):
    def __init__(self, signal_store, tm_store, oms_store, cross_store, shuffle=True):
        self.datasets = {
            "signal": signal_store,
            "tm": tm_store,
            "oms": oms_store,
        }
        self.signal = signal_store
        self.tm = tm_store
        self.oms = oms_store
        self.cross = cross_store
        self.total_length = signal_store.total_length
        self.shuffle = shuffle

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if self.shuffle:
                signal_idx = np.random.randint(len(self.signal))
                tm_idx = np.random.randint(len(self.tm))
                oms_idx = tm_idx # np.random.randint(len(self.oms))
                cross_idx = tm_idx # np.random.randint(len(self.cross))
            else:
                signal_idx = idx
                tm_idx = idx
                oms_idx = idx
                cross_idx = idx
            signal = self.signal[signal_idx]
            tm = self.tm[tm_idx]
            oms = self.oms[oms_idx]
            cross = self.cross[cross_idx]
            return {
                "signal": np.array(signal),
                "tm": np.array(tm),
                "oms": np.array(oms),
                "cross": np.array(cross),
            }
        elif isinstance(idx, slice):
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
            # work out how many samples there will be
            n_samples = len(range(start, stop, step))
            signal_samples = []
            tm_samples = []
            oms_samples = []
            cross_samples = []
            for d_idx in range(n_samples):
                if self.shuffle:
                    signal_idx = np.random.randint(len(self.signal))
                    tm_idx = np.random.randint(len(self.tm))
                    oms_idx = tm_idx # np.random.randint(len(self.oms))
                    cross_idx = tm_idx # np.random.randint(len(self.cross))
                else:
                    signal_idx = start + d_idx * step
                    tm_idx = start + d_idx * step
                    oms_idx = start + d_idx * step
                    cross_idx = start + d_idx * step
                signal_samples.append(self.signal[signal_idx])
                tm_samples.append(self.tm[tm_idx])
                oms_samples.append(self.oms[oms_idx])
                cross_samples.append(self.cross[cross_idx])
            return {
                "signal": np.array(signal_samples),
                "tm": np.array(tm_samples),
                "oms": np.array(oms_samples),
                "cross": np.array(cross_samples),
            }
        elif isinstance(idx, tuple):
            batch = []
            for _idx in idx:
                if isinstance(_idx, int):
                    batch.append(self.__getitem__(_idx))
                elif isinstance(_idx, slice):
                    batch.append(self.__getitem__(_idx))
            signal_arrs = []
            tm_arrs = []
            oms_arrs = []
            cross_arrs = []
            for b in batch:
                if len(b["signal"].shape) > 2:
                    signal_arrs.append(b["signal"])
                    tm_arrs.append(b["tm"])
                    oms_arrs.append(b["oms"])
                    cross_arrs.append(b["cross"])
                else:
                    signal_arrs.append(b["signal"].unsqueeze(0))
                    tm_arrs.append(b["tm"].unsqueeze(0))
                    oms_arrs.append(b["oms"].unsqueeze(0))
                    cross_arrs.append(b["cross"].unsqueeze(0))
            return {
                "signal": np.vstack(signal_arrs),
                "tm": np.vstack(tm_arrs),
                "oms": np.vstack(oms_arrs),
                "cross": np.vstack(cross_arrs),
            }
        elif isinstance(self, str):
            return self.datasets[idx]
        else:
            raise ValueError("Invalid index type")


class ResamplingTraining(Dataset, dict):
    def __init__(self, sim, resampling_dataset):
        self.prior = sim.prior
        self.f_over_pivot = sim.coarse_grained_f / np.sqrt(sim.f_vec[0] * sim.f_vec[-1])
        self.resampling_dataset = resampling_dataset
        self.total_length = len(self.resampling_dataset)
        self.z_store = self.prior.sample(self.total_length)
        self.shuffle = resampling_dataset.shuffle

    def __len__(self):
        return self.total_length
    
    def sample(self, z, cross=1.0):
        n_sims = z.shape[0] if len(z.shape) > 1 else 1
        if n_sims == 1:
            data = self.resampling_dataset[0]
        else:
            data = self.resampling_dataset[:n_sims]
        if len(z.shape) > 1:
            out = {
                "z": torch.from_numpy(z).float(),
                "data": torch.from_numpy(
                    np.einsum(
                        "ij,ijk->ijk",
                        np.power(self.f_over_pivot[:, None], z[:, 1]).T,
                        np.einsum(
                            "i,ijk->ijk", 10 ** z[:, 0] / 10 ** (-11.0), data["signal"]
                        ),
                    )
                    + np.einsum("i,ijk->ijk", z[:, 2] ** 2, data["tm"])
                    + np.einsum("i,ijk->ijk", z[:, 3] ** 2, data["oms"])
                    + cross * np.einsum("i,ijk->ijk", z[:, 2] * z[:, 3], data["cross"])
                ).numpy(),
            }
        else:
            out = {
                "z": torch.from_numpy(z).float(),
                "data": torch.from_numpy(
                    np.einsum(
                        "i,ij->ij",
                        self.f_over_pivot ** z[1],
                        10 ** z[0] / 10 ** (-11.0) * data["signal"],
                    )
                    + z[2] ** 2 * data["tm"]
                    + z[3] ** 2 * data["oms"]
                    + cross * z[2] * z[3] * data["cross"]
                ).float(),
            }
        return out

    def __getitem__(self, idx):
        data = self.resampling_dataset[idx]
        if len(data["signal"].shape) > 2:
            if self.shuffle:
                z = self.prior.sample(data["signal"].shape[0])
            else:
                z = self.z_store[idx]
            out = {
                "z": torch.from_numpy(z).float(),
                "data": torch.from_numpy(
                    np.einsum(
                        "ij,ijk->ijk",
                        np.power(self.f_over_pivot[:, None], z[:, 1]).T,
                        np.einsum(
                            "i,ijk->ijk", 10 ** z[:, 0] / 10 ** (-11.0), data["signal"]
                        ),
                    )
                    + np.einsum("i,ijk->ijk", z[:, 2] ** 2, data["tm"])
                    + np.einsum("i,ijk->ijk", z[:, 3] ** 2, data["oms"])
                    + np.einsum("i,ijk->ijk", z[:, 2] * z[:, 3], data["cross"])
                ).numpy(),
            }
        else:
            if self.shuffle:
                z = self.prior.sample()
            else:
                z = self.z_store[idx]
            out = {
                "z": torch.from_numpy(z).float(),
                "data": torch.from_numpy(
                    np.einsum(
                        "i,ij->ij",
                        self.f_over_pivot ** z[1],
                        10 ** z[0] / 10 ** (-11.0) * data["signal"],
                    )
                    + z[2] ** 2 * data["tm"]
                    + z[3] ** 2 * data["oms"]
                    + z[2] * z[3] * data["cross"]
                ).float(),
            }
        return out


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
