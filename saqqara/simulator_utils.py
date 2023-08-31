import numpy as np
import swyft
import swyft.lightning as sl
import bilby
from scipy.interpolate import interp1d
import sys
sys.path.append("../likelihood-based")
from mcmc_utils import coarse_grain_data
import pandas as pd
import glob
import torch
import os
from itertools import cycle

bilby.core.utils.logger.setLevel("WARNING")
Hubble_over_h = 3.24e-18


class Simulator(sl.Simulator):
    def __init__(self, conf, bounds=None, prior_samples=None):
        super().__init__()
        self.injection_parameters = conf["injection"].copy()
        self.priors = conf["priors"].copy()
        self.sgwb_priors = self.priors["sgwb"].copy()
        self.noise_priors = self.priors["noise"].copy()
        self.n_chunks = conf["data_options"]["n_chunks"]
        self.n_grid = conf["data_options"]["n_grid"]
        self.sampling_frequency = conf["data_options"]["sampling_frequency"]
        self.minimum_frequency = conf["data_options"]["minimum_frequency"]
        self.maximum_frequency = conf["data_options"]["maximum_frequency"]
        self.f_vec = np.arange(
            self.minimum_frequency, self.maximum_frequency, self.sampling_frequency
        )
        self.p_transient = conf["data_options"]["p_transient"]
        if self.p_transient > 0.0:
            self.transient_store = (
                conf["data_options"]["transient_store"]
                if conf["data_options"]["transient_store"][-1] == "/"
                else conf["data_options"]["transient_store"] + "/"
            )
        else:
            self.transient_store = None
        self.rescaling = 1e32
        self.bounds = bounds
        self.prior_samples = prior_samples
        if self.bounds is not None:
            for idx, key in enumerate(self.sgwb_priors.keys()):
                self.sgwb_priors[key].minimum = self.bounds[idx, 0]
                self.sgwb_priors[key].maximum = self.bounds[idx, 1]
            for idx, key in enumerate(self.noise_priors.keys()):
                self.noise_priors[key].minimum = self.bounds[
                    idx + len(self.sgwb_priors.keys()), 0
                ]
                self.noise_priors[key].maximum = self.bounds[
                    idx + len(self.sgwb_priors.keys()), 1
                ]
        elif self.prior_samples is not None:
            self.prior_samples = (
                {"z_total": cycle(prior_samples)}
                if prior_samples is not None
                else None
            )
            self.ns_bounds = conf["ns_bounds"].copy()
        self.LISA_data = np.loadtxt(os.path.dirname(os.path.abspath(__file__)) + "/../utils/LISA_strain.txt")
        self.TM_noise = interp1d(self.LISA_data[:, 0], self.LISA_data[:, 1])
        self.OMS_noise = interp1d(self.LISA_data[:, 0], self.LISA_data[:, 2])
        self.transform_samples = swyft.to_numpy32

    def sgwb_template(self, f, *args, **kwargs):
        conversion = 4 * np.pi**2 * f**3 / 3 / Hubble_over_h**2
        return (
            10 ** args[0]
            * self.rescaling
            * (f / np.sqrt(f[0] * f[-1])) ** args[1]
            / conversion
        )

    def noise_template(self, f, TM, OMS):
        return self.rescaling * (
            TM**2 * self.TM_noise(f) + OMS**2 * self.OMS_noise(f)
        )

    def generate_observation(self):
        params = self.injection_parameters.copy()
        z_sgwb = np.array([params["sgwb"][key] for key in self.sgwb_priors.keys()])
        z_noise = np.array([params["noise"][key] for key in self.noise_priors.keys()])
        z_total = np.concatenate((z_sgwb, z_noise))
        return self.sample(
            conditions={
                "z_total": z_total,
            },
            targets=["data"],
        )

    def generate_z_total(self):
        if self.prior_samples is None:
            z_sgwb = np.array(
                [self.sgwb_priors[key].sample() for key in self.sgwb_priors.keys()]
            )
            z_noise = np.array(
                [self.noise_priors[key].sample() for key in self.noise_priors.keys()]
            )
            return np.concatenate((z_sgwb, z_noise))
        else:
            return next(self.prior_samples["z_total"]).numpy()

    def generate_sgwb_template(self, z_total):
        z_sgwb = z_total[: len(self.sgwb_priors.keys())]
        return self.sgwb_template(self.f_vec, *z_sgwb)

    def generate_noise_template(self, z_total):
        z_noise = z_total[len(self.sgwb_priors.keys()) :]
        return self.noise_template(self.f_vec, *z_noise)

    def compute_exposure(self):
        yr = 365.25 * 24 * 3600
        Tyrs = self.n_chunks / (yr * np.diff(self.f_vec)[0])
        return Tyrs

    def get_transient(self):
        files = glob.glob(self.transient_store + "*.npz")
        file = np.load(files[np.random.choice(len(files))])
        array = file[file.files[0]]
        return array[np.random.choice(array.shape[0]), :]

    def get_array_mask(self):
        arr_mask = np.random.uniform(0.0, 1.0, self.n_chunks) < self.p_transient
        return arr_mask

    def generate_data(self, temp_sgwb, temp_noise, arr_mask):
        shape = (self.n_chunks, len(self.f_vec))
        d_sgwb = (
            np.random.normal(0.0, np.sqrt(temp_sgwb), shape)
            + 1j * np.random.normal(0.0, np.sqrt(temp_sgwb), shape)
        ) / np.sqrt(2)
        d_noise = (
            np.random.normal(0.0, np.sqrt(temp_noise), shape)
            + 1j * np.random.normal(0.0, np.sqrt(temp_noise), shape)
        ) / np.sqrt(2)
        if self.p_transient > 0.0:
            d_transient = np.zeros_like(d_noise)
            d_transient[arr_mask] = (
                1e-3
                * np.sqrt(self.rescaling)
                * np.vstack(
                    [
                        self.get_transient()
                        for _ in range(len(np.where(arr_mask == True)))
                    ]
                )
            )
        else:
            d_transient = np.zeros_like(d_noise)
        return np.mean(np.abs(d_sgwb + d_noise + d_transient) ** 2, axis=0)

    def generate_MCMC_observation(self, obs):
        data = obs["data"]
        temp_noise = obs["temp_noise"]
        cg_data = pd.DataFrame(
            coarse_grain_data(self.f_vec, {"XX": data}, {"XX": temp_noise}, self.n_grid)
        )
        return cg_data

    def build(self, graph):
        """
        Define the computational graph, which allows us to sample specific targets efficiently
        """
        z_total = graph.node("z_total", self.generate_z_total)
        arr_mask = graph.node("arr_mask", self.get_array_mask)
        temp_sgwb = graph.node("temp_sgwb", self.generate_sgwb_template, z_total)
        temp_noise = graph.node("temp_noise", self.generate_noise_template, z_total)
        data = graph.node("data", self.generate_data, temp_sgwb, temp_noise, arr_mask)


def init_simulator(conf: dict, bounds=None, prior_samples=None):
    """
    Initialise the swyft simulator
    Args:
      conf: dictionary of config options, output of init_config
      bounds: Array of prior bounds
    Returns:
      Swyft lightning simulator instance
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> tmnre_parser = read_config(sysargs)
    >>> conf = init_config(tmnre_parser, sysargs, sim=True)
    >>> simulator = init_simulator(conf)
    """
    simulator = Simulator(conf, bounds, prior_samples)
    return simulator


def simulate(simulator, store, conf, max_sims=None):
    """
    Run a swyft simulator to save simulations into a given zarr store
    Args:
      simulator: swyft simulator object
      store: initialised zarr store
      conf: dictionary of config options, output of init_config
      max_sims: maximum number of simulations to perform (otherwise will fill store)
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> tmnre_parser = read_config(sysargs)
    >>> conf = init_config(tmnre_parser, sysargs, sim=True)
    >>> simulator = init_simulator(conf)
    >>> store = setup_zarr_store(conf, simulator)
    >>> simulate(simulator, store, conf)
    """
    store.simulate(
        sampler=simulator,
        batch_size=int(conf["zarr_params"]["chunk_size"]),
        max_sims=max_sims,
    )
