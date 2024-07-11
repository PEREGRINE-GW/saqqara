import numpy as np
import saqqara
import sys
import os
import argparse
import shutil
import glob

sys.path.insert(0, os.path.dirname(__file__) + "/../simulator/")
from simulator import LISA_AET

sys.path.insert(0, os.path.dirname(__file__) + "/../inference/")
from networks import SignalAET
from dataloader import get_resampling_dataloader


def get_network(id, sim):
    config = glob.glob(
        os.path.dirname(__file__) + f"/../training_dir/training_config_id={id}.yaml"
    )[0]
    ckpt = glob.glob(
        os.path.dirname(__file__) + f"/../training_dir/saqqara-*_id={id}.ckpt"
    )[0]
    settings = saqqara.load_settings(config_path=config)
    network = SignalAET(settings=settings, sim=sim)
    network = saqqara.load_state(network=network, ckpt=ckpt)
    return network


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Noise variation model training")
    parser.add_argument("-c", type=str, help="config path")

    args = parser.parse_args()

    print(f"\n[INFO] Loading data from config: {args.c}")
    config = saqqara.load_settings(args.c)
    print("[INFO] Training settings:\n")
    for key, value in config["train"].items():
        print(f"{key}: {value}")
    print("[INFO] Loading simulator")
    sim = LISA_AET(config)
    print("[INFO] Loading network instance")
    network = SignalAET(settings=config, sim=sim)
    restart_id = config["train"].get("restart_id", None)
    if restart_id is not None:
        print(f"[INFO] Restarting training from id={restart_id}")
        network = get_network(restart_id, sim)
    print("[INFO] Loading datasets")
    train_dl, val_dl = get_resampling_dataloader(sim, config)
    if not os.path.exists(config["train"]["trainer_dir"]):
        os.makedirs(config["train"]["trainer_dir"])
    shutil.copy(
        args.c,
        config["train"]["trainer_dir"] + f"/training_config_id={network.rid}.yaml",
    )
    try:
        logger = saqqara.setup_logger(config, rid=network.rid)
        trainer = saqqara.setup_trainer(config, logger=logger)
        trainer.fit(network, train_dl, val_dl)

    except KeyboardInterrupt:
        print("[INFO] Exiting...")
        exit(0)
