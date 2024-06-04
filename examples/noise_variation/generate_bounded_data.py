import numpy as np
import matplotlib.pyplot as plt
import saqqara
from simulator import LISA_AET
import random
import string
import os
import tqdm


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


def total_sims():
    import glob

    count = 0
    files = glob.glob("new_bounded_simulations/cg_data_*.npy")
    for fl in files:
        data = np.load(fl)
        count += data.shape[0]
    return count


if __name__ == "__main__":
    config = saqqara.load_settings("bounded_config.yaml")
    sim = LISA_AET(config)
    Nsamples_per_chunk = 128
    try:
        while total_sims() < 1_000_000:
            print("Total simulations so far: ", total_sims())
            z_out = []
            data_out = []
            for _ in tqdm.tqdm(range(Nsamples_per_chunk)):
                sample = sim.sample()
                z_out.append(sample["z"])
                data_out.append(sample["coarse_grained_data"])
            if not os.path.exists("./new_bounded_simulations/"):
                os.mkdir("./new_bounded_simulations")
            rid = id_generator(4)
            np.save(f"new_bounded_simulations/z_{rid}", np.array(z_out, dtype=np.float32))
            np.save(f"new_bounded_simulations/cg_data_{rid}", np.array(data_out, dtype=np.float32))

    except KeyboardInterrupt:
        print("Total simulations: ", total_sims())
        print("Exiting...")
        exit(0)
