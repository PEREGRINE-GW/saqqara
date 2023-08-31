print(
    r"""
             /'{>           Initialising SAQQARA
         ____) (____        --------------------
       //'--;   ;--'\\      Type: Load Network
      ///////\_/\\\\\\\     Args: config, round
             m m            
"""
)

import sys
import glob
from config_utils import read_config, init_config, info
from simulator_utils import init_simulator
from inference_utils import (
    setup_trainer,
    init_network,
    load_bounds,
)

if __name__ == "__main__":
    args = sys.argv[1:]
    info(f"Reading config file: {args[0]}")
    tmnre_parser = read_config(args)
    conf = init_config(tmnre_parser, args, sim=True)
    hparams = conf["hparams"]
    store_path = (
        conf["zarr_params"]["store_path"]
        if conf["zarr_params"]["store_path"][-1] != "/"
        else conf["zarr_params"]["store_path"][:-1]
    )
    run_id = conf["zarr_params"]["run_id"]
    round_id = int(args[1])
    simulator = init_simulator(conf, bounds=load_bounds(conf, round_id))
    val_data_store = simulator.sample(hparams["validation_batch_size"])
    val_data = val_data_store.get_dataloader(
        num_workers=int(hparams["num_workers"]),
        batch_size=int(hparams["validation_batch_size"]),
        on_after_load_sample=None,
    )
    trainer_dir = f"{conf['zarr_params']['store_path']}/trainer_{conf['zarr_params']['run_id']}_R{round_id}"
    trainer = setup_trainer(trainer_dir, conf, round_id)
    network = init_network(conf)
    trainer.test(
        network, val_data, glob.glob(f"{trainer_dir}/epoch*_R{round_id}.ckpt")[0]
    )
    info(
        f"Loaded network from checkpoint: {glob.glob(f'{trainer_dir}/epoch*_R{round_id}.ckpt')[0]}"
    )
