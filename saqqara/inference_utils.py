import numpy as np
import os
import torch
from torch import nn
from torch.functional import F
from pathlib import Path

torch.set_float32_matmul_precision("high")
import pickle
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import swyft.lightning as sl
import swyft
from swyft.lightning.estimators import LogRatioEstimator_Autoregressive


class InferenceNetwork(sl.SwyftModule):
    def __init__(self, conf):
        super().__init__()
        self.batch_size = conf["hparams"]["training_batch_size"]
        self.num_params = len(conf["priors"]["sgwb"].keys()) + len(
            conf["priors"]["noise"].keys()
        )
        self.method = conf["tmnre"]["method"]
        self.one_d = conf["tmnre"]["one_d"]
        if self.one_d and self.method == "tmnre":
            self.n_features = 2 * self.num_params
        elif not self.one_d and self.method == "tmnre":
            self.n_features = (
                2
                * len(conf["tmnre"]["marginals"])
                * (len(conf["tmnre"]["marginals"]) - 1)
            )
        elif self.method == "anre":
            self.n_features = 2 * self.num_params * (self.num_params - 1)
            self.param_order = conf["tmnre"]["param_order"]
        self.sampling_frequency = conf["data_options"]["sampling_frequency"]
        self.minimum_frequency = conf["data_options"]["minimum_frequency"]
        self.maximum_frequency = conf["data_options"]["maximum_frequency"]
        self.f_vec = np.arange(
            self.minimum_frequency, self.maximum_frequency, self.sampling_frequency
        )
        self.linear_compression = LinearCompression(
            in_features=len(self.f_vec),
            n_features=self.n_features,
            conf=conf,
        )
        self.unet_f = Unet(
            n_in_channels=1,
            n_out_channels=1,
            sizes=(16, 32, 64, 128, 256),
            down_sampling=(2, 2, 2, 2),
        )
        if self.one_d and self.method == "tmnre":
            self.logratios = sl.LogRatioEstimator_1dim(
                num_features=self.n_features,
                num_params=self.num_params,
                num_blocks=4,
                varnames="z_total",
            )
        elif not self.one_d and self.method == "tmnre":
            self.logratios_1d = sl.LogRatioEstimator_1dim(
                num_features=self.n_features,
                num_params=self.num_params,
                num_blocks=4,
                varnames="z_total",
            )
            self.logratios_2d = sl.LogRatioEstimator_Ndim(
                num_features=self.n_features,
                marginals=conf["tmnre"]["marginals"],
                num_blocks=4,
                varnames="z_total",
            )
        elif self.method == "anre":
            self.lre = LogRatioEstimator_Autoregressive(
                self.n_features, self.num_params, "z_total"
            )
        self.online_normalisation = swyft.networks.OnlineStandardizingLayer(
            shape=torch.Size([len(self.f_vec)])
        )
        self.optimizer_init = sl.AdamOptimizerInit(lr=conf["hparams"]["learning_rate"])

    def forward(self, A, B):
        data = torch.log(A["data"])
        data = self.online_normalisation(data)
        data = self.unet_f(data.unsqueeze(1))
        compression = self.linear_compression(data.squeeze(1))
        if self.method == "tmnre" and self.one_d:
            z_tot = B["z_total"]
            logratios = self.logratios(compression, z_tot)
            return logratios
        if self.method == "tmnre" and not self.one_d:
            z_tot = B["z_total"]
            logratios_1d = self.logratios_1d(compression, z_tot)
            logratios_2d = self.logratios_2d(compression, z_tot)
            return logratios_1d, logratios_2d
        elif self.method == "anre":
            if self.param_order is not None:
                z_tot_A = A["z_total"][:, self.param_order]
                z_tot_B = B["z_total"][:, self.param_order]
            else:
                z_tot_A = A["z_total"]
                z_tot_B = B["z_total"]
            logratios = self.lre(compression, z_tot_A, z_tot_B)
            return logratios


class LinearCompression(nn.Module):
    def __init__(self, in_features, n_features, conf):
        super(LinearCompression, self).__init__()
        if conf["data_options"]["sampling_frequency"] < 1e-4:
            self.sequential = nn.Sequential(
                nn.LazyLinear(in_features),
                nn.ReLU(),
                nn.LazyLinear(n_features * 2),
                nn.ReLU(),
                nn.LazyLinear(n_features),
            )
        else:
            self.sequential = nn.Sequential(
                nn.LazyLinear(in_features * 2),
                nn.ReLU(),
                nn.LazyLinear(in_features * 4),
                nn.ReLU(),
                nn.LazyLinear(in_features),
                nn.ReLU(),
                nn.LazyLinear(n_features * 2),
                nn.ReLU(),
                nn.LazyLinear(n_features),
            )

    def forward(self, x):
        return self.sequential(x)


# 1D Unet implementation below
class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        mid_channels=None,
        padding=1,
        bias=False,
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                mid_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                mid_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, down_sampling=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(down_sampling), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2):
        super().__init__()
        self.up = nn.ConvTranspose1d(
            in_channels, in_channels // 2, kernel_size=kernel_size, stride=stride
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_signal_length = x2.size()[2] - x1.size()[2]

        x1 = F.pad(
            x1, [diff_signal_length // 2, diff_signal_length - diff_signal_length // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(OutConv, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(
        self,
        n_in_channels,
        n_out_channels,
        sizes=(16, 32, 64, 128, 256),
        down_sampling=(2, 2, 2, 2),
    ):
        super(Unet, self).__init__()
        self.inc = DoubleConv(n_in_channels, sizes[0])
        self.down1 = Down(sizes[0], sizes[1], down_sampling[0])
        self.down2 = Down(sizes[1], sizes[2], down_sampling[1])
        self.down3 = Down(sizes[2], sizes[3], down_sampling[2])
        self.down4 = Down(sizes[3], sizes[4], down_sampling[3])
        self.up1 = Up(sizes[4], sizes[3])
        self.up2 = Up(sizes[3], sizes[2])
        self.up3 = Up(sizes[2], sizes[1])
        self.up4 = Up(sizes[1], sizes[0])
        self.outc = OutConv(sizes[0], n_out_channels)
        self.batch_norm = nn.LazyBatchNorm1d(momentum=0.01)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        f = self.outc(x)
        return f


def init_network(conf: dict):
    """
    Initialise the network with the settings given in a loaded config dictionary
    Args:
      conf: dictionary of config options, output of init_config
      data_norm: dictionary of normalisation values for min and max of log10(data)
        if None, skips normalisation step
    Returns:
      Pytorch lightning network class
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> tmnre_parser = read_config(sysargs)
    >>> conf = init_config(tmnre_parser, sysargs, sim=True)
    >>> network = init_network(conf)
    """
    network = InferenceNetwork(conf)
    return network


def setup_zarr_store(
    conf: dict,
    simulator,
    round_id: int = None,
    coverage: bool = False,
    n_sims: int = None,
):
    """
    Initialise or load a zarr store for saving simulations
    Args:
      conf: dictionary of config options, output of init_config
      simulator: simulator object, output of init_simulator
      round_id: specific round id for store name
      coverage: specifies if store should be used for coverage sims
      n_sims: number of simulations to initialise store with
    Returns:
      Zarr store object
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> tmnre_parser = read_config(sysargs)
    >>> conf = init_config(tmnre_parser, sysargs, sim=True)
    >>> simulator = init_simulator(conf)
    >>> store = setup_zarr_store(conf, simulator)
    """
    zarr_params = conf["zarr_params"]
    if zarr_params["use_zarr"]:
        chunk_size = zarr_params["chunk_size"]
        if n_sims is None:
            if "nsims" in zarr_params.keys():
                n_sims = zarr_params["nsims"]
            else:
                n_sims = zarr_params["sim_schedule"][round_id - 1]
        shapes, dtypes = simulator.get_shapes_and_dtypes()
        store_path = zarr_params["store_path"]
        if round_id is not None:
            if coverage:
                store_dir = f"{store_path}/coverage_simulations_{zarr_params['run_id']}_R{round_id}"
            else:
                store_dir = (
                    f"{store_path}/simulations_{zarr_params['run_id']}_R{round_id}"
                )
        else:
            if coverage:
                store_dir = (
                    f"{store_path}/coverage_simulations_{zarr_params['run_id']}_R1"
                )
            else:
                store_dir = f"{store_path}/simulations_{zarr_params['run_id']}_R1"

        store = sl.ZarrStore(f"{store_dir}")
        store.init(N=n_sims, chunk_size=chunk_size, shapes=shapes, dtypes=dtypes)
        return store
    else:
        return None


def setup_dataloader(store, simulator, conf: dict, round_id: int = None):
    """
    Initialise a dataloader to read in simulations from a zarr store
    Args:
      store: zarr store to load from, output of setup_zarr_store
      conf: dictionary of config options, output of init_config
      simulator: simulator object, output of init_simulator
      round_id: specific round id for store name
    Returns:
      (training dataloader, validation dataloader), trainer directory
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> tmnre_parser = read_config(sysargs)
    >>> conf = init_config(tmnre_parser, sysargs, sim=True)
    >>> simulator = init_simulator(conf)
    >>> store = setup_zarr_store(conf, simulator)
    >>> train_data, val_data, trainer_dir = setup_dataloader(store, simulator, conf)
    """
    if round_id is not None:
        trainer_dir = f"{conf['zarr_params']['store_path']}/trainer_{conf['zarr_params']['run_id']}_R{round_id}"
    else:
        trainer_dir = f"{conf['zarr_params']['store_path']}/trainer_{conf['zarr_params']['run_id']}_R1"
    if not os.path.isdir(trainer_dir):
        os.mkdir(trainer_dir)
    hparams = conf["hparams"]
    if conf["tmnre"]["resampler"]:
        resampler = simulator.get_resampler(targets=conf["tmnre"]["noise_targets"])
    else:
        resampler = None
    train_data = store.get_dataloader(
        num_workers=int(hparams["num_workers"]),
        batch_size=int(hparams["training_batch_size"]),
        idx_range=[0, int(hparams["train_data"] * len(store))],
        on_after_load_sample=resampler,
    )
    val_data = store.get_dataloader(
        num_workers=int(hparams["num_workers"]),
        batch_size=int(hparams["validation_batch_size"]),
        idx_range=[
            int(hparams["train_data"] * len(store)),
            len(store) - 1,
        ],
        on_after_load_sample=None,
    )
    return train_data, val_data, trainer_dir


def setup_trainer(trainer_dir: str, conf: dict, round_id: int):
    """
    Initialise a pytorch lightning trainer and relevant directories
    Args:
      trainer_dir: location for the training logs
      conf: dictionary of config options, output of init_config
      round_id: specific round id for store name
    Returns:
      Swyft lightning trainer instance
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> tmnre_parser = read_config(sysargs)
    >>> conf = init_config(tmnre_parser, sysargs, sim=True)
    >>> simulator = init_simulator(conf)
    >>> store = setup_zarr_store(conf, simulator, 1)
    >>> train_data, val_data, trainer_dir = setup_dataloader(store, simulator, conf, 1)
    >>> trainer = setup_trainer(trainer_dir, conf, 1)
    """
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=conf["hparams"]["early_stopping"],
        verbose=False,
        mode="min",
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{trainer_dir}",
        filename="{epoch}_{val_loss:.2f}_{train_loss:.2f}" + f"_R{round_id}",
        mode="min",
    )
    logger_tbl = pl_loggers.TensorBoardLogger(
        save_dir=f"{trainer_dir}",
        name=f"{conf['zarr_params']['run_id']}_R{round_id}",
        version=None,
        default_hp_metric=False,
    )

    device_params = conf["device_params"]
    hparams = conf["hparams"]
    trainer = sl.SwyftTrainer(
        accelerator=device_params["device"],
        gpus=device_params["n_devices"],
        min_epochs=hparams["min_epochs"],
        max_epochs=hparams["max_epochs"],
        logger=logger_tbl,
        callbacks=[lr_monitor, early_stopping_callback, checkpoint_callback],
    )
    return trainer


def save_logratios(logratios, conf, round_id):
    """
    Save logratios from a particular round
    Args:
      logratios: swyft logratios instance
      conf: dictionary of config options, output of init_config
      round_id: specific round id for store name
    """
    if not os.path.isdir(
        f"{conf['zarr_params']['store_path']}/logratios_{conf['zarr_params']['run_id']}"
    ):
        os.mkdir(
            f"{conf['zarr_params']['store_path']}/logratios_{conf['zarr_params']['run_id']}"
        )
    with open(
        f"{conf['zarr_params']['store_path']}/logratios_{conf['zarr_params']['run_id']}/logratios_R{round_id}",
        "wb",
    ) as p:
        pickle.dump(logratios, p)


def save_coverage(coverage, conf: dict, round_id: int):
    """
    Save coverage samples from a particular round
    Args:
      coverage: swyft coverage object instance
      conf: dictionary of config options, output of init_config
      round_id: specific round id for store name
    """
    if not os.path.isdir(
        f"{conf['zarr_params']['store_path']}/coverage_{conf['zarr_params']['run_id']}"
    ):
        os.mkdir(
            f"{conf['zarr_params']['store_path']}/coverage_{conf['zarr_params']['run_id']}"
        )
    with open(
        f"{conf['zarr_params']['store_path']}/coverage_{conf['zarr_params']['run_id']}/coverage_R{round_id}",
        "wb",
    ) as p:
        pickle.dump(coverage, p)


def save_bounds(bounds, conf: dict, round_id: int):
    """
    Save bounds from a particular round
    Args:
      bounds: unpacked swyft bounds object
      conf: dictionary of config options, output of init_config
      round_id: specific round id for store name
    """
    np.savetxt(
        f"{conf['zarr_params']['store_path']}/bounds_{conf['zarr_params']['run_id']}_R{round_id}.txt",
        bounds,
    )


def load_bounds(conf: dict, round_id: int):
    """
    Load bounds from a particular round
    Args:
      conf: dictionary of config options, output of init_config
      round_id: specific round id for store name
    Returns:
      Bounds object with ordering defined by the param idxs in the config
    """
    if round_id == 1:
        return None
    else:
        bounds = np.loadtxt(
            f"{conf['zarr_params']['store_path']}/bounds_{conf['zarr_params']['run_id']}_R{round_id - 1}.txt"
        )
        return bounds


def linear_rescale(v, v_ranges, u_ranges):
    """
    Rescales a tensor in its last dimension from v_ranges to u_ranges
    """
    device = v.device

    # Move points onto hypercube
    v_bias = v_ranges[:, 0].to(device)
    v_width = (v_ranges[:, 1] - v_ranges[:, 0]).to(device)

    # Move points onto hypercube
    u_bias = u_ranges[:, 0].to(device)
    u_width = (u_ranges[:, 1] - u_ranges[:, 0]).to(device)

    t = (v - v_bias) / v_width
    u = t * u_width + u_bias  # (..., N)
    return u


class BaseTorchDataset(torch.utils.data.Dataset):
    """Simple torch dataset class for nested sampling samples"""

    def __init__(self, root_dir, n_max=None):
        self.root_dir = root_dir
        self.X = torch.load(root_dir)
        self.n = len(self.X)
        if n_max is not None:
            self.n = min(self.n, n_max)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        z = self.X[idx]
        if idx >= self.n or idx < 0:
            raise IndexError()
        return torch.as_tensor(z)


def load_constrained_samples(conf, round_id):
    """
    Load constrained samples from a given round
    Args:
      conf: dictionary of config options, output of init_config
      round_id: round id to load samples from
    Returns:
      Torch dataset of constrained samples from slice sampler
    """
    if round_id > 1:
        constrained_samples = BaseTorchDataset(
            conf["zarr_params"]["store_path"]
            + f"/prior_samples/constrained_samples_R{round_id - 1}.pt"
        )
    else:
        constrained_samples = None
    return constrained_samples


def save_constrained_samples(samples, conf, round_id):
    """
    Load constrained samples from a given round
    Args:
      conf: dictionary of config options, output of init_config
      round_id: round id to load samples from
    Returns:
      Torch dataset of constrained samples from slice sampler
    """
    Path(conf["zarr_params"]["store_path"] + f"/prior_samples/").mkdir(
        parents=True, exist_ok=True
    )
    torch.save(
        samples,
        conf["zarr_params"]["store_path"]
        + f"/prior_samples/constrained_samples_R{round_id}.pt",
    )


def get_data_norm(A):
    return {
        "min": torch.log10(torch.tensor(np.array(A["data"]))).min(),
        "max": torch.log10(torch.tensor(np.array(A["data"]))).max(),
    }
