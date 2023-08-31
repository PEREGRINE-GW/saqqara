print(
    r"""
             /'{>           Initialising SAQQARA
         ____) (____        --------------------
       //'--;   ;--'\\      Type: Load Simulator
      ///////\_/\\\\\\\     
             m m            
"""
)

import sys
import swyft.lightning as sl
from datetime import datetime
from config_utils import read_config, init_config, info
from simulator_utils import init_simulator

if __name__ == "__main__":
    args = sys.argv[1:]
    info(msg=f"Reading config file: {args[0]}")
    tmnre_parser = read_config(args)
    conf = init_config(tmnre_parser, args, sim=True)
    simulator = init_simulator(conf)
