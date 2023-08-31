print(
    r"""
             /'{>           Initialising SAQQARA
         ____) (____        --------------------
       //'--;   ;--'\\      Type: Generate Transients
      ///////\_/\\\\\\\     Args: config
             m m            
"""
)
import sys
from ripple.waveforms import IMRPhenomXAS
from pathlib import Path
import time
import jax
import jax.numpy as jnp
from tqdm import tqdm
import numpy as np
from config_utils import read_config, init_config, info
from simulator_utils import init_simulator


@jax.jit
def get_params(key):
    keys = jax.random.split(key, 5)
    Mc = jax.random.uniform(keys[0], minval=8e5, maxval=9e5)
    eta = jax.random.uniform(keys[1], minval=0.16, maxval=0.25)
    chi1 = jax.random.uniform(keys[2], minval=-1.0, maxval=1.0)
    chi2 = jax.random.uniform(keys[3], minval=-1.0, maxval=1.0)
    dist_mpc = jax.random.uniform(keys[4], minval=5e4, maxval=1e5)
    tc = 0.0
    phic = 0.0
    return jnp.array(
        [Mc, eta, chi1, chi2, dist_mpc, tc, phic],
        dtype=jnp.float64,
    )


def get_param_batch(N):
    return jnp.array([get_params(jax.random.PRNGKey(time.time_ns())) for _ in range(N)])


gen = jax.jit(IMRPhenomXAS.gen_IMRPhenomXAS)
gen_vmap = jax.vmap(gen, (None, 0, None))

if __name__ == "__main__":
    args = sys.argv[1:]
    tmnre_parser = read_config(args)
    info(msg=f"Reading config file: {args[0]}")
    conf = init_config(tmnre_parser, args, sim=True)
    simulator = init_simulator(conf)
    _f_arr = jnp.array(simulator.f_vec)
    _f_ref = simulator.f_vec[0]
    store_path = conf["data_options"]["transient_store"]
    store_path = store_path if store_path[-1] == "/" else store_path + "/"
    num_sims = conf["data_options"]["transient_store_size"]
    info(f"Creating transient store ({store_path}) for {num_sims} simulations")
    Path(store_path).mkdir(parents=True, exist_ok=True)
    batch_size = 1000
    iterations = num_sims // batch_size
    with tqdm(total=iterations, desc="Generating Transients", ncols=100) as pbar:
        for iteration in range(iterations):
            sample = gen_vmap(_f_arr, get_param_batch(batch_size), _f_ref)
            np.savez(store_path + f"transient_{iteration}.npz", np.array(sample))
            pbar.update(1)
