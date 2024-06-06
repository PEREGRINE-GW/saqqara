import numpy as np
import matplotlib.pyplot as plt
import saqqara
import sys
import random
import string
import os
import tqdm
import glob
import argparse
import shutil

sys.path.insert(0, os.path.dirname(__file__) + "/../simulator/")
from simulator import LISA_AET


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


def total_sims(directory):
    count = 0
    files = glob.glob(f"{directory}/cg_data_*.npy")
    for fl in files:
        data = np.load(fl)
        count += data.shape[0]
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Noise variation observation data generation"
    )
    parser.add_argument("-c", type=str, help="config path")
    parser.add_argument(
        "-vary",
        metavar="vary",
        help="varying noise or fixed noise",
        action="store_const",
        const=True,
        default=False,
    )
    parser.add_argument(
        "-q",
        metavar="quiet",
        help="quiet evalulation mode",
        action="store_const",
        const=True,
        default=False,
    )

    args = parser.parse_args()

    print(f"\n[INFO] Loading data from config: {args.c}")
    config = saqqara.load_settings(args.c)
    sim = LISA_AET(config)
    Nsamples_per_chunk = config["simulate"].get("chunk_size", 128)
    Nsims = config["simulate"].get("store_size", 100_000)
    save_directory = config["simulate"].get("store_name", "simulations")
    save_directory = (
        save_directory if save_directory[-1] != "/" else save_directory[:-1]
    )
    print(f"[INFO] Saving data to: {os.path.dirname(__file__) + '/' + save_directory}")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory, exist_ok=True)
    shutil.copy(args.c, save_directory + "/observation_config.yaml")
    if total_sims(save_directory) != 0:
        print("[WARNING] Directory not empty, config may match all simulations")
    print(f"[INFO] Generating {Nsims} simulations")
    print("[WARNING] This script overwrites the TM and OMS priors")
    try:
        while total_sims(save_directory) < Nsims:
            print("[INFO] Total simulations so far: ", total_sims(save_directory))
            z_out = []
            data_out = []
            for _ in tqdm.tqdm(range(Nsamples_per_chunk), disable=args.q):
                z_in = sim.sample(targets=["z"])["z"]
                z_noise = (
                    np.array([3.0, 15.0])
                    if not args.vary
                    else np.array(
                        [np.random.normal(3.0, 0.6), np.random.normal(15.0, 3.0)]
                    )
                )
                sample = sim.sample(
                    conditions={
                        "z": np.array([z_in[0], z_in[1], z_noise[0], z_noise[1]])
                    }
                )
                z_out.append(sample["z"])
                data_out.append(sample["coarse_grained_data"])
            rid = id_generator(6)
            if len(glob.glob(f"{save_directory}/z_{rid}")) != 0:
                rid = id_generator(6)
            np.save(f"{save_directory}/z_{rid}", np.array(z_out, dtype=np.float32))
            np.save(
                f"{save_directory}/cg_data_{rid}", np.array(data_out, dtype=np.float32)
            )
        print(f"[INFO] Simulations complete in {save_directory}")

    except KeyboardInterrupt:
        print("[INFO] Total simulations: ", total_sims())
        print("[INFO] Exiting...")
        exit(0)
