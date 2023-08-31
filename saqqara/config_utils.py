import configparser
import subprocess
from distutils.util import strtobool
from pathlib import Path

import numpy as np
import os
import sys
import bilby
from datetime import datetime

bilby.core.utils.logger.setLevel("WARNING")
from bilby.core import prior as prior_core


def read_config(sysargs: list):
    """
    Load the config using configparser, returns a parser object that can be accessed as
    e.g. tmnre_parser['FIELD']['parameter'], this will always return a string, so must be
    parsed for data types separately, see init_config
    Args:
      sysargs: list of command line arguments (i.e. strings) containing path to config in position 0
    Returns:
      Config parser object containing information in configuration file sections
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> tmnre_parser = read_config(sysargs)
    >>> tmnre_parser['TMNRE']['num_rounds'] etc.
    """
    tmnre_config_file = sysargs[0]
    tmnre_parser = configparser.ConfigParser()
    tmnre_parser.read_file(open(tmnre_config_file))
    return tmnre_parser


def init_config(tmnre_parser, sysargs: list, sim: bool = False) -> dict:
    """
    Initialise the config dictionary, this is a dictionary of dictionaries obtaining by parsing
    the relevant config file. A copy of the config file is stored along with the eventual simulations.
    All parameters are parsed to the correct data type from strings, including lists and booleans etc.
    Args:
      tmnre_parser: config parser object, output of read_config
      sysargs: list of command line arguments (i.e. strings) containing path to config in position 0
      sim: boolean to choose whether to include config copying features etc. if False, will create a
           copy of the config in the store directory and generate the param idxs file
    Returns:
      Dictionary of configuration options with all data types explicitly parsed
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> tmnre_parser = read_config(sysargs)
    >>> conf = init_config(tmnre_parser, sysargs, )
    """
    tmnre_config_file = sysargs[0]
    store_path = tmnre_parser["ZARR PARAMS"]["store_path"]
    if not sim:
        Path(store_path).mkdir(parents=True, exist_ok=True)
    run_id = tmnre_parser["ZARR PARAMS"]["run_id"]
    if not sim:
        subprocess.run(
            f"cp {tmnre_config_file} {store_path}/config_{run_id}.ini", shell=True
        )
    conf = {}

    injection = {"sgwb": {}, "noise": {}}
    for key in tmnre_parser["SGWB INJECTION"]:
        if key in [None]:
            continue
        else:
            injection["sgwb"][key] = np.float64(tmnre_parser["SGWB INJECTION"][key])
    for key in tmnre_parser["NOISE INJECTION"]:
        if key in [None]:
            continue
        else:
            injection["noise"][key] = np.float64(tmnre_parser["NOISE INJECTION"][key])
    conf["injection"] = injection.copy()

    data_options = {}
    for key in tmnre_parser["DATA OPTIONS"]:
        if key in ["n_chunks", "n_grid", "transient_store_size"]:
            data_options[key] = int(tmnre_parser["DATA OPTIONS"][key])
        elif key in ["transient_store"]:
            data_options[key] = str(tmnre_parser["DATA OPTIONS"][key])
        elif key in ["p_transient"]:
            data_options[key] = float(tmnre_parser["DATA OPTIONS"][key])
        else:
            data_options[key] = float(tmnre_parser["DATA OPTIONS"][key])

    conf["data_options"] = data_options.copy()

    sgwb_priors = populate_sgwb_priors(tmnre_parser)
    noise_priors = populate_noise_priors(tmnre_parser)
    priors = {"sgwb": sgwb_priors, "noise": noise_priors}
    conf["priors"] = priors.copy()
    conf["ns_bounds"] = [
        *[
            [sgwb_priors[key].minimum, sgwb_priors[key].maximum]
            for key in sgwb_priors.keys()
        ],
        *[
            [noise_priors[key].minimum, noise_priors[key].maximum]
            for key in noise_priors.keys()
        ],
    ]
    conf["num_params"] = len(sgwb_priors.keys()) + len(noise_priors.keys())

    param_idxs = {}
    param_names = {}
    if not sim:
        with open(
            (
                f"{tmnre_parser['ZARR PARAMS']['store_path']}/param_idxs_{tmnre_parser['ZARR PARAMS']['run_id']}.txt"
            ),
            "w",
        ) as f:
            for idx, key in enumerate(sgwb_priors.keys()):
                param_idxs[key] = idx
                param_names[idx] = key
                f.write(f"{key} {idx} sgwb\n")
            for idx, key in enumerate(noise_priors.keys()):
                param_idxs[key] = idx + len(sgwb_priors.keys())
                param_names[idx + len(sgwb_priors.keys())] = key
                f.write(f"{key} {idx + len(sgwb_priors.keys())} noise\n")
            f.close()
    else:
        for idx, key in enumerate(sgwb_priors.keys()):
            param_idxs[key] = idx
            param_names[idx] = key
        for idx, key in enumerate(noise_priors.keys()):
            param_idxs[key] = idx + len(sgwb_priors.keys())
            param_names[idx + len(sgwb_priors.keys())] = key
    conf["param_idxs"] = param_idxs
    conf["param_names"] = param_names

    tmnre = {}
    tmnre["infer_only"] = False
    tmnre["marginals"] = None
    tmnre["param_order"] = None
    for key in tmnre_parser["TMNRE"]:
        if key in ["num_rounds", "num_batch_samples", "samples_per_slice", "num_steps"]:
            tmnre[key] = int(tmnre_parser["TMNRE"][key])
        elif key in ["one_d", "generate_obs", "resampler", "infer_only", "skip_ns"]:
            tmnre[key] = bool(strtobool(tmnre_parser["TMNRE"][key]))
        elif key in ["obs_path", "method"]:
            tmnre[key] = str(tmnre_parser["TMNRE"][key])
        elif key in ["epsilon", "logl_th_max", "alpha"]:
            tmnre[key] = float(tmnre_parser["TMNRE"][key])
        elif key in ["noise_targets"]:
            tmnre[key] = [
                str(target) for target in tmnre_parser["TMNRE"][key].split(",")
            ]
        elif key in ["param_order"]:
            tmnre[key] = [
                int(p) for p in tmnre_parser["TMNRE"][key].split(",")
            ]
        elif key in ["marginals"]:
            marginals_list = []
            if tmnre_parser["TMNRE"][key] != "all":
                marginals_string = tmnre_parser["TMNRE"][key]
                marginals_list = []
                for marginal in marginals_string.split("("):
                    split_marginal = marginal.split(")")
                    if split_marginal[0] != "":
                        indices = split_marginal[0].split(",")
                        mg = []
                        for index in indices:
                            mg.append(int(index.strip(" ")))
                        marginals_list.append(tuple(mg))
                tmnre["marginals"] = tuple(marginals_list)
            else:
                for idx1 in range(conf["num_params"]):
                    for idx2 in range(idx1 + 1, conf["num_params"]):
                        marginals_list.append(tuple([idx1, idx2]))
                tmnre["marginals"] = tuple(marginals_list)
    conf["tmnre"] = tmnre

    sampling = {}
    for key in tmnre_parser["SAMPLING"]:
        if key in ["mcmc_obs_path"]:
            sampling[key] = str(tmnre_parser["SAMPLING"][key])
        elif key in ["n_burn", "n_steps"]:
            sampling[key] = int(tmnre_parser["SAMPLING"][key])
        elif key in ["r_conv", "iter_max"]:
            sampling[key] = float(tmnre_parser["SAMPLING"][key])
    conf["sampling"] = sampling

    zarr_params = {}
    for key in tmnre_parser["ZARR PARAMS"]:
        if key in ["run_id", "store_path"]:
            zarr_params[key] = tmnre_parser["ZARR PARAMS"][key]
        elif key in ["use_zarr", "run_parallel"]:
            zarr_params[key] = bool(strtobool(tmnre_parser["ZARR PARAMS"][key]))
        elif key in ["nsims", "chunk_size", "njobs"]:
            zarr_params[key] = int(tmnre_parser["ZARR PARAMS"][key])
        elif key in ["targets"]:
            zarr_params[key] = [
                target
                for target in tmnre_parser["ZARR PARAMS"][key].split(",")
                if target != ""
            ]
        elif key in ["sim_schedule"]:
            zarr_params[key] = [
                int(nsims) for nsims in tmnre_parser["ZARR PARAMS"][key].split(",")
            ]
    if "sim_schedule" in zarr_params.keys():
        if len(zarr_params["sim_schedule"]) != tmnre["num_rounds"]:
            print(
                f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [config_utils.py] | WARNING: Error in sim scheduler, setting to default n_sims = 30_000"
            )
            zarr_params["nsims"] = 30_000
        elif "nsims" in zarr_params.keys():
            zarr_params.pop("nsims")
    conf["zarr_params"] = zarr_params

    hparams = {}
    for key in tmnre_parser["HYPERPARAMS"].keys():
        if key in [
            "min_epochs",
            "max_epochs",
            "early_stopping",
            "num_workers",
            "training_batch_size",
            "validation_batch_size",
        ]:
            hparams[key] = int(tmnre_parser["HYPERPARAMS"][key])
        elif key in ["learning_rate", "train_data", "val_data"]:
            hparams[key] = float(tmnre_parser["HYPERPARAMS"][key])
    conf["hparams"] = hparams

    device_params = {}
    for key in tmnre_parser["DEVICE PARAMS"]:
        if key in ["device"]:
            device_params[key] = str(tmnre_parser["DEVICE PARAMS"][key])
        elif key in ["n_devices"]:
            device_params[key] = int(tmnre_parser["DEVICE PARAMS"][key])
    conf["device_params"] = device_params

    return conf


def populate_sgwb_priors(tmnre_parser):
    """
    Construct the prior dictionary taking into account the relevant bilby defaults
    Args:
      tmnre_parser: config parser object, output of read_config
    Returns:
      Prior dictionary
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> tmnre_parser = read_config(sysargs)
    >>> sgwb_priors = populate_sgwb_priors(tmnre_parser)
    """
    # Initialise prior dictionaries
    prior_dict = {}
    for key in tmnre_parser[f"SGWB PRIORS"].keys():
        prior_dict[key] = getattr(
            prior_core,
            "Uniform",
        )(
            minimum=float(tmnre_parser[f"SGWB PRIORS"][key].split(",")[0]),
            maximum=float(tmnre_parser[f"SGWB PRIORS"][key].split(",")[1]),
            boundary=None,
        )

    return prior_dict


def populate_noise_priors(tmnre_parser):
    """
    Construct the prior dictionary taking into account the relevant bilby defaults
    Args:
      tmnre_parser: config parser object, output of read_config
    Returns:
      Prior dictionary
    Examples
    --------
    >>> sysargs = ['/path/to/config/file.txt']
    >>> tmnre_parser = read_config(sysargs)
    >>> noise_priors = populate_noise_priors(tmnre_parser)
    """
    # Initialise prior dictionaries
    prior_dict = {}
    for key in tmnre_parser[f"NOISE PRIORS"].keys():
        prior_dict[key] = getattr(
            prior_core,
            "Uniform",
        )(
            minimum=float(tmnre_parser[f"NOISE PRIORS"][key].split(",")[0]),
            maximum=float(tmnre_parser[f"NOISE PRIORS"][key].split(",")[1]),
            boundary=None,
        )

    return prior_dict


def info(msg):
    print(
        f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [{os.path.basename(sys.argv[0])}] | {msg}"
    )
