print(
    r"""
             /'{>           Initialising SAQQARA
         ____) (____        --------------------
       //'--;   ;--'\\      Type: TMNRE Inference
      ///////\_/\\\\\\\     Args: config
             m m            
"""
)

import sys
import glob
import pickle
from datetime import datetime
import swyft.lightning as sl
from config_utils import read_config, init_config, info
from simulator_utils import init_simulator, simulate
from inference_utils import (
    setup_zarr_store,
    setup_dataloader,
    setup_trainer,
    init_network,
    load_bounds,
    save_bounds,
    linear_rescale,
    load_constrained_samples,
    save_constrained_samples,
    save_logratios,
)
from swyft.utils.ns import SwyftSimpleSliceSampler

import subprocess
import psutil
import logging
import torch
import numpy as np
import swyft


if __name__ == "__main__":
    args = sys.argv[1:]
    info(f"Reading config file: {args[0]}")
    tmnre_parser = read_config(args)
    conf = init_config(tmnre_parser, args)
    logging.basicConfig(
        filename=f"{conf['zarr_params']['store_path']}/log_{conf['zarr_params']['run_id']}.log",
        filemode="w",
        format="%(asctime)s | %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )
    simulator = init_simulator(conf)
    if conf["tmnre"]["method"] == "anre":
        ns_bounds = conf["ns_bounds"].copy()
    observation_path = conf["tmnre"]["obs_path"]
    with open(observation_path, "rb") as f:
        obs = pickle.load(f)
    subprocess.run(
        f"cp {observation_path} {conf['zarr_params']['store_path']}/observation_{conf['zarr_params']['run_id']}",
        shell=True,
    )
    info(
        f"Observation loaded and saved in {conf['zarr_params']['store_path']}/observation_{conf['zarr_params']['run_id']}"
    )
    logging.info(
        f"Observation loaded and saved in {conf['zarr_params']['store_path']}/observation_{conf['zarr_params']['run_id']}"
    )
    obs = sl.Sample({key: obs[key] for key in ["data"]})
    for round_id in range(1, int(conf["tmnre"]["num_rounds"]) + 1):
        start_time = datetime.now()
        info(f"Initialising zarrstore for round {round_id}")
        store = setup_zarr_store(conf, simulator, round_id=round_id)
        logging.info(f"Starting simulations for round {round_id}")
        info(f"Simulating data for round {round_id}")
        if conf["zarr_params"]["run_parallel"]:
            info("Running in parallel - spawning processes")
            processes = []
            if conf["zarr_params"]["njobs"] == -1:
                njobs = psutil.cpu_count(logical=True)
            elif conf["zarr_params"]["njobs"] > psutil.cpu_count(logical=False):
                njobs = psutil.cpu_count(logical=True)
            else:
                njobs = conf["zarr_params"]["njobs"]
            if store.sims_required > 0:
                for job in range(njobs):
                    if conf["tmnre"]["method"] == "tmnre":
                        p = subprocess.Popen(
                            [
                                "python",
                                "run_parallel.py",
                                f"{conf['zarr_params']['store_path']}/config_{conf['zarr_params']['run_id']}.ini",
                                str(round_id),
                            ]
                        )
                    elif conf["tmnre"]["method"] == "anre":
                        p = subprocess.Popen(
                            [
                                "python",
                                "run_parallel.py",
                                f"{conf['zarr_params']['store_path']}/config_{conf['zarr_params']['run_id']}.ini",
                                str(round_id),
                                str(job),
                                str(njobs),
                            ]
                        )
                    processes.append(p)
                for p in processes:
                    p.wait()
        else:
            info("WARNING: Running in serial mode")
            if conf["tmnre"]["method"] == "tmnre":
                bounds = load_bounds(conf, round_id)
                simulator = init_simulator(conf, bounds)
            elif conf["tmnre"]["method"] == "anre":
                prior_samples = load_constrained_samples(conf, round_id)
                simulator = init_simulator(conf, prior_samples=prior_samples)
            simulate(simulator, store, conf)
        logging.info(f"Simulations for round {round_id} completed")

        info(f"Setting up dataloaders for round {round_id}")
        train_data, val_data, trainer_dir = setup_dataloader(
            store, simulator, conf, round_id
        )

        info(f"Setting up trainer for round {round_id}")
        trainer = setup_trainer(trainer_dir, conf, round_id)

        info(f"Initialising network for round {round_id}")
        network = init_network(conf)
        if (
            not conf["tmnre"]["infer_only"]
            or len(glob.glob(f"{trainer_dir}/epoch*_R{round_id}.ckpt")) == 0
        ):
            print(
                f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [tmnre.py] | Training network for round {round_id}"
            )
            trainer.fit(network, train_data, val_data)
            logging.info(
                f"Training completed for round {round_id}, checkpoint available at {glob.glob(f'{trainer_dir}/epoch*_R{round_id}.ckpt')[0]}"
            )
        if conf["tmnre"]["method"] == "tmnre":
            info("Generate prior samples")
            prior_sim = init_simulator(conf, load_bounds(conf, round_id))
            prior_samples = prior_sim.sample(100_000, targets=["z_total"])

            info("Generate posterior samples")
            trainer.test(
                network,
                val_data,
                glob.glob(f"{trainer_dir}/epoch*_R{round_id}.ckpt")[0],
            )
            logratios = trainer.infer(
                network, obs, prior_samples.get_dataloader(batch_size=2048)
            )
            logging.info(f"Logratios saved for round {round_id}")
            info(f"Saving logratios from round {round_id}")
            save_logratios(logratios, conf, round_id)
            info(f"Update bounds from round {round_id}")
            if conf["tmnre"]["one_d"]:
                bounds = (
                    sl.bounds.get_rect_bounds(
                        logratios, threshold=conf["tmnre"]["alpha"]
                    )
                    .bounds.squeeze(1)
                    .numpy()
                )
                save_bounds(bounds, conf, round_id)
            else:
                bounds = (
                    sl.bounds.get_rect_bounds(
                        logratios, threshold=conf["tmnre"]["alpha"]
                    )[0]
                    .bounds.squeeze(1)
                    .numpy()
                )
                save_bounds(bounds, conf, round_id)
        elif conf["tmnre"]["method"] == "anre":

            def log_likelihood(net, z_total):
                z_total = linear_rescale(
                    z_total,
                    torch.tensor([0, 1]).unsqueeze(0),
                    torch.tensor(ns_bounds),
                )
                B = dict(z_total=z_total.to(net.device))
                A = dict(
                    data=torch.tensor(obs["data"]).unsqueeze(0).to(net.device),
                    z_total=B["z_total"],
                )
                with torch.no_grad():
                    predictions = net(A, B)
                logl = predictions["lrs_total"].logratios.squeeze(-1)
                return logl

            trainer.test(
                network,
                val_data,
                glob.glob(f"{trainer_dir}/epoch*_R{round_id}.ckpt")[0],
            )

            if (
                not conf["tmnre"]["skip_ns"]
                or len(
                    glob.glob(
                        f"{conf['zarr_params']['store_path']}/logratios*/logratios_R{round_id}"
                    )
                )
                == 0
                or len(
                    glob.glob(
                        f"{conf['zarr_params']['store_path']}/prior_samples*/constrained_samples_R{round_id}*"
                    )
                )
                == 0
            ):
                info(f"Starting nested sampling exploration for round {round_id}")
                network.eval() if str(
                    network.device
                ) == "cpu" else network.cuda().eval()
                sample = swyft.to_torch(simulator.sample(500, targets=["z_total"]))
                X_init = sample["z_total"]
                X_init = linear_rescale(
                    X_init,
                    torch.tensor(ns_bounds),
                    torch.tensor([0, 1]).unsqueeze(0),
                )
                X_init.max()
                ssss = SwyftSimpleSliceSampler(X_init)
                ssss.nested_sampling(
                    lambda z_total: log_likelihood(network, z_total),
                    epsilon=conf["tmnre"]["epsilon"],
                    logl_th_max=conf["tmnre"]["logl_th_max"],
                    num_batch_samples=conf["tmnre"]["num_batch_samples"],
                    samples_per_slice=conf["tmnre"]["samples_per_slice"],
                    num_steps=conf["tmnre"]["num_steps"],
                )
                dimensions = X_init.size(1)
                logl_th = ssss.get_threshold(1e-3)
                X_cp, L_cp = ssss.generate_constrained_prior_samples(
                    lambda z_total: log_likelihood(network, z_total),
                    N=10000,
                    min_logl=logl_th,
                    batch_size=100,
                )
                ssss = SwyftSimpleSliceSampler(X_cp)
                ssss.nested_sampling(
                    lambda z_total: log_likelihood(network, z_total),
                    epsilon=conf["tmnre"]["epsilon"],
                    logl_th_max=conf["tmnre"]["logl_th_max"],
                    num_batch_samples=conf["tmnre"]["num_batch_samples"],
                    samples_per_slice=conf["tmnre"]["samples_per_slice"],
                    num_steps=conf["tmnre"]["num_steps"],
                )
                X_post, L_post = ssss.get_posterior_samples()
                X_post = linear_rescale(
                    X_post,
                    torch.tensor([0, 1]).unsqueeze(0),
                    torch.tensor(ns_bounds),
                )
                lrs = swyft.LogRatioSamples(
                    L_post.unsqueeze(-1) * 0,
                    X_post.unsqueeze(-2),
                    np.array([["z_total[%i]" % i for i in range(dimensions)]]),
                )
                save_logratios(lrs, conf, round_id)
                alpha = conf["tmnre"]["alpha"]
                logl_th = ssss.get_threshold(alpha)
                info(
                    f"Log-likelihood threshold for p = {alpha}: {logl_th:.2f} in round {round_id}"
                )
                logging.info(
                    f"Log-likelihood threshold for p = {alpha}: {logl_th:.2f} in round {round_id}"
                )
                if round_id != conf["tmnre"]["num_rounds"]:
                    X_cp, L_cp = ssss.generate_constrained_prior_samples(
                        lambda z_total: log_likelihood(network, z_total),
                        N=4 * conf["zarr_params"]["sim_schedule"][round_id],
                        min_logl=logl_th,
                        batch_size=100,
                    )
                    X_cp = linear_rescale(
                        X_cp,
                        torch.tensor([0, 1]).unsqueeze(0),
                        torch.tensor(ns_bounds),
                    )
                    save_constrained_samples(X_cp, conf, round_id)
                    info(f"Constrained samples saved for round {round_id}")
                    logging.info(f"Constrained samples saved for round {round_id}")
        end_time = datetime.now()
        logging.info(f"Completed round {round_id}")
        info(f"Completed round {round_id} in {end_time - start_time}.")
