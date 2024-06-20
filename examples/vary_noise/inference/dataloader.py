# import all the missing packages
import glob
import os
import saqqara
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler


def get_datasets(data_dir):
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} does not exist.")
    print(f"[INFO] Loading data from {data_dir}")
    Z_FILES = glob.glob(data_dir + "/z_*.npy")
    Z_FILES = sorted(Z_FILES, key=lambda x: str(x.split("_")[-1].split(".")[0]))
    DATA_FILES = glob.glob(data_dir + "/cg_data_*.npy")
    DATA_FILES = sorted(DATA_FILES, key=lambda x: str(x.split("_")[-1].split(".")[0]))
    # Check all names match
    for idx in range(len(Z_FILES)):
        assert (
            Z_FILES[idx].split("_")[-1].split(".")[0]
            == DATA_FILES[idx].split("_")[-1].split(".")[0]
        )
    # Compute total number of simulations
    n_simulations = len(Z_FILES) * 128
    print(f"[INFO] Total number of simulations: {n_simulations}")

    # Get data shapes
    z_shape = np.load(Z_FILES[0]).shape[1:]
    data_shape = np.load(DATA_FILES[0]).shape[1:]
    print(f"z shape: {z_shape}")
    print(f"data shape: {data_shape}")

    z_dataset = saqqara.NPYDataset(file_paths=Z_FILES)
    data_dataset = saqqara.NPYDataset(file_paths=DATA_FILES)
    training_dataset = saqqara.TrainingDataset(
        z_store=z_dataset, data_store=data_dataset
    )
    return training_dataset


def get_data_npy_dataset(data_dir):
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory {data_dir} does not exist.")
    print(f"[INFO] Loading data from {data_dir}")
    DATA_FILES = glob.glob(data_dir + "/cg_data_*.npy")
    DATA_FILES = sorted(DATA_FILES, key=lambda x: str(x.split("_")[-1].split(".")[0]))

    # Compute total number of simulations
    n_simulations = len(DATA_FILES) * 128
    print(f"[INFO] Total number of simulations: {n_simulations}")
    data_shape = np.load(DATA_FILES[0]).shape[1:]
    print(f"data shape: {data_shape}")
    data_dataset = saqqara.NPYDataset(file_paths=DATA_FILES)
    return data_dataset


def setup_dataloaders(
    dataset,
    total_size=None,
    train_fraction=0.8,
    val_fraction=0.2,
    num_workers=0,
    batch_size=64,
):
    if total_size is None:
        total_size = len(dataset)
    if total_size > len(dataset):
        raise ValueError(
            f"Total size {total_size} is larger than dataset size {len(dataset)}"
        )
    indices = list(range(len(dataset)))
    train_idx, val_idx = int(np.floor(train_fraction * total_size)), int(
        np.floor((train_fraction + val_fraction) * total_size)
    )
    train_indices, val_indices = indices[:train_idx], indices[train_idx:val_idx]
    # train_sampler, val_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices)
    train_sampler, val_sampler = SequentialSampler(train_indices), SequentialSampler(
        val_indices
    )
    train_dataloader = DataLoader(
        dataset=dataset,
        drop_last=True,
        sampler=train_sampler,
        num_workers=int(num_workers),
        batch_size=int(batch_size),
    )
    val_dataloader = DataLoader(
        dataset=dataset,
        drop_last=True,
        sampler=val_sampler,
        num_workers=int(num_workers),
        batch_size=int(batch_size),
    )
    return train_dataloader, val_dataloader


def get_dataloader(settings):
    training_settings = settings.get("train", {})
    data_dir = training_settings.get("data_dir", "./simulations")
    dataset = get_datasets(data_dir)
    total_size = training_settings.get("total_size", None)
    train_fraction = training_settings.get("train_fraction", 0.8)
    val_fraction = training_settings.get("val_fraction", 0.2)
    num_workers = training_settings.get("num_workers", 0)
    batch_size = training_settings.get("batch_size", 64)
    return setup_dataloaders(
        dataset=dataset,
        total_size=total_size,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        num_workers=num_workers,
        batch_size=batch_size,
    )


def get_resampling_dataloader(sim, settings):
    training_settings = settings.get("train", {})
    if training_settings["type"] != "resampling":
        raise ValueError("Training type must be resampling")
    signal_data_dir = training_settings.get("signal_dir")
    tm_data_dir = training_settings.get("tm_dir")
    oms_data_dir = training_settings.get("oms_dir")
    signal_dataset = get_data_npy_dataset(signal_data_dir)
    tm_dataset = get_data_npy_dataset(tm_data_dir)
    oms_dataset = get_data_npy_dataset(oms_data_dir)
    resampling_dataset = saqqara.RandomSamplingDataset(
        signal_dataset, tm_dataset, oms_dataset
    )
    dataset = saqqara.ResamplingTraining(sim, resampling_dataset)
    total_size = training_settings.get("total_size", None)
    train_fraction = training_settings.get("train_fraction", 0.8)
    val_fraction = training_settings.get("val_fraction", 0.2)
    num_workers = training_settings.get("num_workers", 0)
    batch_size = training_settings.get("batch_size", 64)
    return setup_dataloaders(
        dataset=dataset,
        total_size=total_size,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        num_workers=num_workers,
        batch_size=batch_size,
    )
