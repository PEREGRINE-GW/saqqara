from config_utils import read_config, init_config
from simulator_utils import init_simulator, simulate
from inference_utils import setup_zarr_store, load_bounds, load_constrained_samples
import sys

if __name__ == "__main__":
    args = sys.argv[1:]
    round_id = int(args[1])
    if "coverage" not in args:
        coverage = False
    else:
        coverage = True
    if len(args) >= 3:
        job_id = int(args[2])
        njobs = int(args[3])
    tmnre_parser = read_config(args)
    conf = init_config(tmnre_parser, args, sim=True)
    if coverage:
        conf["zarr_params"]["chunk_size"] = 50
    if conf["tmnre"]["method"] == "tmnre":
        bounds = load_bounds(conf, round_id)
        simulator = init_simulator(conf, bounds=bounds)
    elif conf["tmnre"]["method"] == "anre":
        prior_samples = load_constrained_samples(conf, round_id)
        if prior_samples is not None:
            idx_range = prior_samples.X.size(0) // njobs
            X = prior_samples.X[job_id * idx_range : (job_id + 1) * idx_range]
            prior_samples.X = X
            prior_samples.n = int(X.size(0))
        simulator = init_simulator(conf, prior_samples=prior_samples)
    store = setup_zarr_store(conf, simulator, round_id=round_id, coverage=coverage)
    while store.sims_required > 0:
        simulate(
            simulator, store, conf, max_sims=int(conf["zarr_params"]["chunk_size"])
        )
