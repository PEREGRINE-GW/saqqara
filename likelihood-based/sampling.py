print(
    r"""
             /'{>           Initialising SAQQARA
         ____) (____        -----------------------
       //'--;   ;--'\\      Type: MCMC Sampling
      ///////\_/\\\\\\\     
             m m            
"""
)

import sys
from datetime import datetime
import numpy as np
import pickle
import sys
sys.path.append("../saqqara")
from config_utils import read_config, init_config, info
from simulator_utils import init_simulator
from mcmc_utils import get_R
import emcee
from multiprocess import Pool
from pathlib import Path
import subprocess
import logging

if __name__ == "__main__":
    args = sys.argv[1:]
    info(f"Reading config file: {args[0]}")
    tmnre_parser = read_config(args)
    conf = init_config(tmnre_parser, args)
    logging.basicConfig(
        filename=f"{conf['zarr_params']['store_path']}/mcmc_log_{conf['zarr_params']['run_id']}.log",
        filemode="w",
        format="%(asctime)s | %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )
    simulator = init_simulator(conf)
    bounds = None
    observation_path = conf["sampling"]["mcmc_obs_path"]
    with open(observation_path, "rb") as f:
        mcmc_obs = pickle.load(f)
    mcmc_path = f"{conf['zarr_params']['store_path']}/mcmc_chains/"
    info(f"Creating MCMC chains directory: {mcmc_path}")
    Path(mcmc_path).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        f"cp {observation_path} {conf['zarr_params']['store_path']}/mcmc_observation_{conf['zarr_params']['run_id']}",
        shell=True,
    )
    logging.info(
        f"Observation loaded and saved in {conf['zarr_params']['store_path']}/mcmc_observation_{conf['zarr_params']['run_id']}"
    )

    logging.info(f"Starting MCMC chain")

    info("Setting up likelihood")

    def mcmc_model(frequency, theta):
        return simulator.sgwb_template(
            frequency, theta[:-2]
        ) + simulator.noise_template(frequency, theta[-2:])

    def log_prior(theta):
        sgwb_keys = simulator.sgwb_priors.keys()
        noise_keys = simulator.noise_priors.keys()
        for val, key in zip(theta[:-2], sgwb_keys):
            if (
                simulator.sgwb_priors[key].minimum > val
                or simulator.sgwb_priors[key].maximum < val
            ):
                return -np.inf

        for val, key in zip(theta[-2:], noise_keys):
            if (
                simulator.noise_priors[key].minimum > val
                or simulator.noise_priors[key].maximum < val
            ):
                return -np.inf
        return 0.0

    def log_likelihood(n_chunks, cg_obs, pars, model):
        return (
            -n_chunks
            / 2
            * np.sum(
                np.array(cg_obs["weight_XX"])
                * (
                    1.0
                    / 3.0
                    * (
                        1
                        - np.array(cg_obs["XX"])
                        / model(np.array(cg_obs["frequency_XX"]), pars)
                    )
                    ** 2
                    + 2.0
                    / 3.0
                    * np.log(
                        model(np.array(cg_obs["frequency_XX"]), pars)
                        / np.array(cg_obs["XX"])
                    )
                    ** 2
                )
            )
        )

    def parallel_run(n_chunks, cg_data, nburn, nsteps, R_conv, iter_max):
        def log_posterior(theta):
            lp = log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(n_chunks, cg_data, theta, mcmc_model)

        nwalkers = 2 * (len(simulator.sgwb_priors) + len(simulator.noise_priors))
        initial_sample = simulator.sample(nwalkers, targets=["z_total"])
        initial = initial_sample["z_total"]

        nwalkers, ndim = initial.shape

        info("Starting initial burn in phase")
        logging.info(f"Starting initial burn in phase")
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, pool=pool)
            sampler.run_mcmc(initial, nburn, progress=True)

        state = sampler.get_chain()[-1, :, :]
        info("Dropping burn in samples")
        logging.info(f"Dropping burn in samples")
        R = 1e100
        i = 1

        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, pool=pool)

            while np.abs(R - 1) > R_conv and i < iter_max:
                sampler.run_mcmc(state, nsteps, progress=True)
                R = get_R(sampler.get_chain())
                state = None
                logging.info(f"After {i} iterations: R = {R:.4f}")
                np.savez(
                    mcmc_path + "mcmc_chain.npz",
                    samples=sampler.get_chain(),
                    pdfs=sampler.lnprobability,
                )
                i += 1

        samples = sampler.get_chain()
        pdfs = sampler.lnprobability
        np.savez(
            mcmc_path + "mcmc_chain.npz",
            samples=sampler.get_chain(),
            pdfs=sampler.lnprobability,
        )
        return samples, pdfs

    samples, pdfs = parallel_run(
        simulator.n_chunks,
        mcmc_obs,
        nburn=conf["sampling"]["n_burn"],
        nsteps=conf["sampling"]["n_steps"],
        R_conv=conf["sampling"]["r_conv"],
        iter_max=conf["sampling"]["iter_max"],
    )
    info(f"MCMC chains saved in {mcmc_path + 'mcmc_chain.npz'}")
