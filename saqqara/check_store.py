print(
    r"""
             /'{>           Initialising SAQQARA
         ____) (____        --------------------
       //'--;   ;--'\\      Type: Check Store
      ///////\_/\\\\\\\     Args: config, round
             m m            
"""
)

from config_utils import read_config, init_config, info
from simulator_utils import init_simulator
from inference_utils import setup_zarr_store
import sys

if __name__ == "__main__":
    args = sys.argv[1:]
    round_id = int(args[1])
    tmnre_parser = read_config(args)
    info(msg=f"Reading config file: {args[0]}")
    conf = init_config(tmnre_parser, args, sim=True)
    simulator = init_simulator(conf)
    store = setup_zarr_store(conf, simulator, round_id=round_id)
    info(
        msg=f"{store.sims_required} simulations required to fill store in round {round_id}"
    )
