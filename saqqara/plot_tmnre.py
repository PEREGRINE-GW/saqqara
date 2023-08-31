print(
    r"""
             /'{>           Initialising SAQQARA
         ____) (____        --------------------
       //'--;   ;--'\\      Type: Plot TMNRE
      ///////\_/\\\\\\\     
             m m            
"""
)

import sys
import swyft.lightning as sl
from datetime import datetime
from config_utils import read_config, init_config
from simulator_utils import init_simulator
import numpy as np
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

if __name__ == "__main__":
    args = sys.argv[1:]
    print(
        f"{datetime.now().strftime('%a %d %b %H:%M:%S')} | [plot_both.py] | Reading config file"
    )
    # Load and parse config file
    tmnre_parser = read_config(args)
    conf = init_config(tmnre_parser, args, sim=True)
    simulator = init_simulator(conf)

    Path(f"{conf['zarr_params']['store_path']}/plots").mkdir(
        parents=True, exist_ok=True
    )

    tmnre_path = f"{conf['zarr_params']['store_path']}/logratios_{conf['zarr_params']['run_id']}/"
    with open(tmnre_path + f"logratios_R{conf['tmnre']['num_rounds']}", "rb") as f:
        logratios = pickle.load(f)

    import swyft

    values = []
    for key in conf["injection"].keys():
        for pkey in conf["injection"][key].keys():
            values.append(conf["injection"][key][pkey])

    if conf["tmnre"]["method"] == "tmnre":
        fig = plt.figure(figsize=(12, 12))
        for i in range(4):
            ax = plt.subplot(2, 2, i + 1)
            swyft.plot_1d(
                logratios,
                bins=100,
                parname=f"z_total[{i}]",
                smooth=1.0,
                ax=ax,
            )
            ax.axvline(values[i], color="r", linestyle="--")
    elif conf["tmnre"]["method"] == "anre":
        fig = plt.figure(figsize=(16, 16))
        swyft.corner(
            logratios,
            bins=100,
            smooth=1.0,
            parnames=["z_total[0]", "z_total[1]", "z_total[2]", "z_total[3]"],
            fig=plt.gcf(),
        )
    plt.tight_layout()
    plt.savefig(f"{conf['zarr_params']['store_path']}/plots/plot_tmnre.png")
