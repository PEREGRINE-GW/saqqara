print(
    r"""
             /'{>           Initialising SAQQARA
         ____) (____        --------------------
       //'--;   ;--'\\      Type: Generate Observation
      ///////\_/\\\\\\\     Args: config
             m m            
"""
)

import sys
import pickle
from config_utils import read_config, init_config, info
from simulator_utils import init_simulator

if __name__ == "__main__":
    args = sys.argv[1:]
    info(msg=f"Reading config file: {args[0]}")
    tmnre_parser = read_config(args)
    conf = init_config(tmnre_parser, args)
    simulator = init_simulator(conf)
    obs = simulator.generate_observation()
    mcmc_obs = simulator.generate_MCMC_observation(obs)
    with open(
        f"{conf['zarr_params']['store_path']}/observation_{conf['zarr_params']['run_id']}",
        "wb",
    ) as f:
        pickle.dump(obs, f)

    mcmc_obs.to_pickle(
        f"{conf['zarr_params']['store_path']}/mcmc_observation_{conf['zarr_params']['run_id']}"
    )
    info(msg=f"Generated observation")
    info(
        msg=f"Swyft observation Path: {conf['zarr_params']['store_path']}/observation_{conf['zarr_params']['run_id']}"
    )
    info(
        msg=f"MCMC observation Path: {conf['zarr_params']['store_path']}/mcmc_observation_{conf['zarr_params']['run_id']}"
    )
