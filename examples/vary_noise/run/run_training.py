import numpy as np
import saqqara
import sys
import os
import argparse
import shutil

sys.path.insert(0, os.path.dirname(__file__) + "/../simulator/")
from simulator import LISA_AET

sys.path.insert(0, os.path.dirname(__file__) + "/../inference/")
from networks import SignalAET
from dataloader import get_dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Noise variation model training")
    parser.add_argument("-c", type=str, help="config path")

    args = parser.parse_args()

    print(f"\n[INFO] Loading data from config: {args.c}")
    print("[INFO] Training settings:\n", args.c["train"])
    config = saqqara.load_settings(args.c)
    print("[INFO] Loading simulator")
    sim = LISA_AET(config)
    print("[INFO] Loading network instance")
    network = SignalAET(settings=config, sim=sim)
    print("[INFO] Loading datasets")
    train_dl, val_dl = get_dataloader(config)
    shutil.copy(args.c, config["train"]["trainer_dir"] + "/training_config.yaml")
    try:
        logger = saqqara.setup_logger(config)
        trainer = saqqara.setup_trainer(config, logger=logger)
        trainer.fit(network, train_dl, val_dl)

    except KeyboardInterrupt:
        print("[INFO] Exiting...")
        exit(0)
