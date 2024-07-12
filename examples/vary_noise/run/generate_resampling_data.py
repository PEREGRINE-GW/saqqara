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
        description="Noise variation training data generation"
    )
    parser.add_argument("-c", type=str, help="config path")
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
    save_directory = config["simulate"].get("store_name", "resampling_simulations")
    save_directory = (
        save_directory if save_directory[-1] != "/" else save_directory[:-1]
    )

    # tm_save_directory = config["simulate"].get("tm_store", "tm_store")
    # tm_save_directory = (
    #     tm_save_directory if tm_save_directory[-1] != "/" else tm_save_directory[:-1]
    # )
    # oms_save_directory = config["simulate"].get("oms_store", "oms_store")
    # oms_save_directory = (
    #     oms_save_directory if oms_save_directory[-1] != "/" else oms_save_directory[:-1]
    # )
    # tm_oms_cross_save_directory = config["simulate"].get(
    #     "tm_oms_cross_store", "tm_oms_cross_store"
    # )
    # tm_oms_cross_save_directory = (
    #     tm_oms_cross_save_directory
    #     if tm_oms_cross_save_directory[-1] != "/"
    #     else tm_oms_cross_save_directory[:-1]
    # )
    # tm_signal_cross_save_directory = config["simulate"].get(
    #     "tm_signal_cross_store", "tm_signal_cross_store"
    # )
    # tm_signal_cross_save_directory = (
    #     tm_signal_cross_save_directory
    #     if tm_signal_cross_save_directory[-1] != "/"
    #     else tm_signal_cross_save_directory[:-1]
    # )
    # oms_signal_cross_save_directory = config["simulate"].get(
    #     "oms_signal_cross_store", "oms_signal_cross_store"
    # )
    # oms_signal_cross_save_directory = (
    #     oms_signal_cross_save_directory
    #     if oms_signal_cross_save_directory[-1] != "/"
    #     else oms_signal_cross_save_directory[:-1]
    # )
    # signal_store = config["simulate"].get("signal_store", "signal_store")
    # signal_store = signal_store if signal_store[-1] != "/" else signal_store[:-1]

    print(f"[INFO] Saving data to: {save_directory}")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory, exist_ok=True)

    # if not os.path.exists(save_directory + "/" + tm_save_directory):
    #     os.makedirs(save_directory + "/" + tm_save_directory, exist_ok=True)
    # if not os.path.exists(save_directory + "/" + oms_save_directory):
    #     os.makedirs(save_directory + "/" + oms_save_directory, exist_ok=True)
    # if not os.path.exists(save_directory + "/" + tm_oms_cross_save_directory):
    #     os.makedirs(save_directory + "/" + tm_oms_cross_save_directory, exist_ok=True)
    # if not os.path.exists(save_directory + "/" + tm_signal_cross_save_directory):
    #     os.makedirs(
    #         save_directory + "/" + tm_signal_cross_save_directory, exist_ok=True
    #     )
    # if not os.path.exists(save_directory + "/" + oms_signal_cross_save_directory):
    #     os.makedirs(
    #         save_directory + "/" + oms_signal_cross_save_directory, exist_ok=True
    #     )
    # if not os.path.exists(save_directory + "/" + signal_store):
    #     os.makedirs(save_directory + "/" + signal_store, exist_ok=True)

    shutil.copy(args.c, save_directory + "/resampling_config.yaml")
    if (
        total_sims(save_directory)
        != 0
        # total_sims(save_directory + "/" + tm_save_directory) != 0
        # or total_sims(save_directory + "/" + oms_save_directory) != 0
        # or total_sims(save_directory + "/" + tm_oms_cross_save_directory) != 0
        # or total_sims(save_directory + "/" + tm_signal_cross_save_directory) != 0
        # or total_sims(save_directory + "/" + oms_signal_cross_save_directory) != 0
        # or total_sims(save_directory + "/" + signal_store) != 0
    ):
        print("[WARNING] Directory not empty, config may not match all simulations")
    print(f"[INFO] Generating {Nsims} simulations")
    try:
        while total_sims(save_directory) < Nsims:
            # while total_sims(save_directory + "/" + tm_save_directory) < Nsims:
            print(
                "[INFO] Total simulations so far: ",
                total_sims(save_directory),
                # total_sims(save_directory + "/" + tm_save_directory),
            )
            z_out = []
            cg_data_out = []
            # tm_data_out = []
            # oms_data_out = []
            # tm_oms_cross_data_out = []
            # tm_signal_cross_data_out = []
            # oms_signal_cross_data_out = []
            # signal_data_out = []
            for _ in tqdm.tqdm(range(Nsamples_per_chunk), disable=args.q):
                sample = sim.sample(
                    conditions={"z": np.array([-11.0, 0.0, 1.0, 1.0])},
                    targets=[
                        "linear_TM_noise_AET",
                        "linear_OMS_noise_AET",
                        "linear_signal_AET",
                    ],
                )
                z_out.append(sample["z"])
                cg_tm_data = sim.generate_coarse_grained_data_from_sum(
                    sim.generate_quadratic_data(sample["linear_TM_noise_AET"])
                )
                cg_oms_data = sim.generate_coarse_grained_data_from_sum(
                    sim.generate_quadratic_data(sample["linear_OMS_noise_AET"])
                )
                cg_signal_data = sim.generate_coarse_grained_data_from_sum(
                    sim.generate_quadratic_data(sample["linear_signal_AET"])
                )
                cg_tm_oms_cross_data = sim.generate_coarse_grained_data_from_sum(
                    2
                    * np.real(
                        np.einsum(
                            "...i,...i->...i",
                            sample["linear_TM_noise_AET"],
                            np.conj(sample["linear_OMS_noise_AET"]),
                        )
                    )
                )
                cg_tm_signal_cross_data = sim.generate_coarse_grained_data_from_sum(
                    2
                    * np.real(
                        np.einsum(
                            "...i,...i->...i",
                            sample["linear_TM_noise_AET"],
                            np.conj(sample["linear_signal_AET"]),
                        )
                    )
                )
                cg_oms_signal_cross_data = sim.generate_coarse_grained_data_from_sum(
                    2
                    * np.real(
                        np.einsum(
                            "...i,...i->...i",
                            sample["linear_OMS_noise_AET"],
                            np.conj(sample["linear_signal_AET"]),
                        )
                    )
                )
                cg_data_out.append(
                    np.array(
                        [
                            cg_tm_data,
                            cg_oms_data,
                            cg_signal_data,
                            cg_tm_oms_cross_data,
                            cg_tm_signal_cross_data,
                            cg_oms_signal_cross_data,
                        ]
                    )
                )
                # tm_data_out.append(cg_tm_data)
                # oms_data_out.append(cg_oms_data)
                # signal_data_out.append(cg_signal_data)
                # tm_oms_cross_data_out.append(cg_tm_oms_cross_data)
                # tm_signal_cross_data_out.append(cg_tm_signal_cross_data)
                # oms_signal_cross_data_out.append(cg_oms_signal_cross_data)
            rid = id_generator(6)
            if len(glob.glob(f"{save_directory}/z_{rid}")) != 0:
                rid = id_generator(6)
            # if (
            #     len(glob.glob(f"{save_directory + '/' + tm_save_directory}/z_{rid}"))
            #     != 0
            # ):
            #     rid = id_generator(6)

            np.save(
                f"{save_directory}/z_{rid}",
                np.array(z_out, dtype=np.float32),
            )
            np.save(
                f"{save_directory}/cg_data_{rid}",
                np.array(cg_data_out, dtype=np.float32),
            )
            # np.save(
            #     f"{save_directory + '/' + tm_save_directory}/z_{rid}",
            #     np.array(z_out, dtype=np.float32),
            # )
            # np.save(
            #     f"{save_directory + '/' + tm_save_directory}/cg_data_{rid}",
            #     np.array(tm_data_out, dtype=np.float32),
            # )
            # np.save(
            #     f"{save_directory + '/' + oms_save_directory}/z_{rid}",
            #     np.array(z_out, dtype=np.float32),
            # )
            # np.save(
            #     f"{save_directory + '/' + oms_save_directory}/cg_data_{rid}",
            #     np.array(oms_data_out, dtype=np.float32),
            # )
            # np.save(
            #     f"{save_directory + '/' + signal_store}/z_{rid}",
            #     np.array(z_out, dtype=np.float32),
            # )
            # np.save(
            #     f"{save_directory + '/' + signal_store}/cg_data_{rid}",
            #     np.array(signal_data_out, dtype=np.float32),
            # )
            # np.save(
            #     f"{save_directory + '/' + tm_oms_cross_save_directory}/z_{rid}",
            #     np.array(z_out, dtype=np.float32),
            # )
            # np.save(
            #     f"{save_directory + '/' + tm_oms_cross_save_directory}/cg_data_{rid}",
            #     np.array(tm_oms_cross_data_out, dtype=np.float32),
            # )
            # np.save(
            #     f"{save_directory + '/' + tm_signal_cross_save_directory}/z_{rid}",
            #     np.array(z_out, dtype=np.float32),
            # )
            # np.save(
            #     f"{save_directory + '/' + tm_signal_cross_save_directory}/cg_data_{rid}",
            #     np.array(tm_signal_cross_data_out, dtype=np.float32),
            # )
            # np.save(
            #     f"{save_directory + '/' + oms_signal_cross_save_directory}/z_{rid}",
            #     np.array(z_out, dtype=np.float32),
            # )
            # np.save(
            #     f"{save_directory + '/' + oms_signal_cross_save_directory}/cg_data_{rid}",
            #     np.array(oms_signal_cross_data_out, dtype=np.float32),
            # )
        print(f"[INFO] Simulations complete in {save_directory}")

    except KeyboardInterrupt:
        print("[INFO] Total simulations: ", total_sims())
        print("[INFO] Exiting...")
        exit(0)
