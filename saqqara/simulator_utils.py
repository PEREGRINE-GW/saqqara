import numpy as np
import swyft
import swyft.lightning as sl
import bilby
from scipy.interpolate import interp1d
import sys
sys.path.append("../likelihood-based/")
from mcmc_utils import coarse_grain_data
import pandas as pd
import glob
import os
import torch
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
        self.n_bins = len(self.sgwb_priors.keys()) - 1
        self.f_bins = np.geomspace(self.f_vec[0], self.f_vec[-1], self.n_bins + 1)
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

    def sgwb_template(self, f, z_sgwb):
        conversion = 4 * np.pi**2 * f**3 / 3 / Hubble_over_h**2
        amp0 = 10 ** z_sgwb[0]
        tilts = z_sgwb[1:]
        pivots = np.sqrt(self.f_bins[:-1] * self.f_bins[1:])
        amplitude_factors = np.append(
            np.array([1.0]),
            self.f_bins[1:-1] ** (-np.diff(tilts))
            * (pivots[1:] ** tilts[1:] / pivots[:-1] ** tilts[:-1]),
        )
        amps = amp0 * np.cumprod(amplitude_factors)
        index_list = [
            np.where((f >= self.f_bins[i]) & (f < self.f_bins[i + 1]))
            for i in range(self.n_bins)
        ]
        if index_list[-1][0][-1] != len(f) - 1:
            index_list[-1] = np.append(index_list[-1], np.array([len(f) - 1]))
        result = np.array([])
        for amp, tilt, pivot, indices in zip(amps, tilts, pivots, index_list):
            result = np.append(result, amp * (f[indices] / pivot) ** tilt)
        return self.rescaling * result / conversion

    def noise_template(self, f, z_noise):
        TM, OMS = z_noise[0], z_noise[1]
        return self.rescaling * (
            TM**2 * self.TM_noise(f) + OMS**2 * self.OMS_noise(f)
        )

    def generate_observation(self):
        params = self.injection_parameters.copy()
        z_sgwb = np.array([params["sgwb"][key] for key in self.sgwb_priors.keys()])
        z_noise = np.array([params["noise"][key] for key in self.noise_priors.keys()])
        return self.sample(
            conditions={
                "z_total": np.concatenate((z_sgwb, z_noise))
            },
            targets=["data"],
        )

    def sample_sgwb_prior(self):
        amplitude = self.sgwb_priors["amplitude"].sample()
        tilts = np.array(
            [self.sgwb_priors[f"tilt_{bin + 1}"].sample() for bin in range(self.n_bins)]
        )
        z_sgwb = np.array([amplitude, *tilts])
        return z_sgwb

    def sample_noise_prior(self):
        z_noise = np.array(
            [self.noise_priors[key].sample() for key in self.noise_priors.keys()]
        )
        return z_noise

    def generate_z_total(self):
        if self.prior_samples is None:
            z_sgwb = self.sample_sgwb_prior()
            z_noise = self.sample_noise_prior()
            return np.concatenate((z_sgwb, z_noise))
        else:
            return next(self.prior_samples["z_total"]).numpy()

    def generate_sgwb_template(self, z_total):
        z_sgwb = z_total[:-2]
        return self.sgwb_template(self.f_vec, z_sgwb)

    def generate_noise_template(self, z_total):
        z_noise = z_total[-2:]
        return self.noise_template(self.f_vec, z_noise)

    def generate_data(self, temp_sgwb, temp_noise):
        shape = (self.n_chunks, len(self.f_vec))
        d_sgwb = (
            np.random.normal(0.0, np.sqrt(temp_sgwb), shape)
            + 1j * np.random.normal(0.0, np.sqrt(temp_sgwb), shape)
        ) / np.sqrt(2)
        d_noise = (
            np.random.normal(0.0, np.sqrt(temp_noise), shape)
            + 1j * np.random.normal(0.0, np.sqrt(temp_noise), shape)
        ) / np.sqrt(2)
        return np.mean(np.abs(d_sgwb + d_noise) ** 2, axis=0)

    def compute_exposure(self):
        yr = 365.25 * 24 * 3600
        Tyrs = self.n_chunks / (yr * np.diff(self.f_vec)[0])
        return Tyrs

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
        temp_sgwb = graph.node("temp_sgwb", self.generate_sgwb_template, z_total)
        temp_noise = graph.node("temp_noise", self.generate_noise_template, z_total)
        data = graph.node("data", self.generate_data, temp_sgwb, temp_noise)


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
