<img align="center" height="200" src="./images/saqqara_logo.png">


[![version](https://img.shields.io/badge/version-0.0.1-blue)](https://github.com/PEREGRINE-GW/peregrine) [![DOI](https://img.shields.io/badge/DOI-arXiv.2309.07954-brightgreen)](https://arxiv.org/abs/2309.07954)
## Description

###### *"Discovered during the 1898 excavation of the tomb of Pa-di-Imen in Saqqara, Egypt, the ***SAQQARA*** bird artifact is dated to about 200 BCE and is of unresolved origin."*

- **SAQQARA** is a Simulation-based Inference (SBI) library designed to perform analysis on stochastic gravitational wave (background) signals (SGWB). It is built on top of the [swyft](https://swyft.readthedocs.io/en/lightning/) code, which implements neural ratio estimation to efficiently access marginal posteriors for all parameters of interest.
- **Related paper:** The details regarding the implementation of the TMNRE algorithm and the application to agnostic and template-based SGWB searches (in the presence of sub-threshold transients) is in: [arxiv:2309.07954](https://arxiv.org/abs/2309.07954).
- **Key benefits:** We show in the above paper a proof-of-principle for simulation-based inference combined with implicit marginalisation (over nuisance parameters) to be very well suited for SGWB data analysis. Our results are additionally validated via comparison to traditional, likelihood-based algorithms.

- **Contacts:** For questions and comments on the code, please contact either [James Alvey](mailto:j.b.g.alvey@uva.nl), [Uddipta Bhardwaj](mailto:u.bhardwaj@uva.nl), or [Mauro Pieroni](mailto:mauro.pieroni@cern.ch). Alternatively feel free to open a github issue.

- **Citation:** If you use SAQQARA in your analysis, or find it useful, we would ask that you please use the following citation.
```
@article{Alvey:2023npw,
    author = "Alvey, James and Bhardwaj, Uddipta and Domcke, Valerie and Pieroni, Mauro and Weniger, Christoph",
    title = "{Simulation-based inference for stochastic gravitational wave background data analysis}",
    eprint = "2309.07954",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    reportNumber = "CERN-TH-2023-167",
    month = "9",
    year = "2023"
}
```

## Recommended Installation Instructions

### Environment Setup
The safest way to install the dependencies for `saqqara` is to create a virtual environment from `python>=3.8`

**Option 1 (venv):**
```
python3 -m venv /your/choice/of/env/path/
```
- Source the new environment
```
source /your/choice/of/env/path/bin/activate
```

**Option 2 (conda):**
```
conda create -n your_env_name python=3.x (python>=3.8 required)
conda activate your_env_name
```

### Code Installation
- Clone the peregrine repo into location of choice
```
cd /path/to/your/code/store/
git clone git@github.com:PEREGRINE-GW/saqqara.git
```
- Install the relevant packages including e.g. `swyft` and GW specific analysis tools
- **PS**: Some features of the `swyft` version in use are in active development. We are happy to resolve issues (if any) that arise during installation.
```
pip install git+https://github.com/undark-lab/swyft.git@f036b15dab0664614b3e3891dd41e25f6f0f230f
pip install tensorboard psutil gwpy lalsuite bilby chainconsumer multiprocess
pip install ripplegw
```

## Running saqqara

Key run files:
- `generate_observation.py` - Generates a test observation from a configuration file given a set of injection parameters
- `generate_transients.py` - (Optional) Generates a store of transients using the `IMRPhenomXAS` waveform approximant. NOTE: Must have `ripplegw` installed.
- `tmnre.py` - Runs the TMNRE algorithm given the parameters in the specified configuration file
- `coverage.py` - Runs coverage tests on the logratio estimators that have been generated by `tmnre.py`

Example Run Scheme:
- Step 1: Generate a configuration file following the instructions in the [examples directory](./examples/config_files). To just do a test run, you will only need to change the `store_path` and `obs_path` options to point to the desired location in which you want to save your data (`mcmc_obs_path` as well if you want to run corresponding likelihood-based analysis for comparison).
- Step 2: Change directory to `saqqara/saqqara` where the run scripts are stored
- Optional step: Generate a transient store using `python generate_transients.py /path/to/config/file.ini`
- Step 3: Generate an observation using `python generate_observation.py /path/to/config/file.ini` or point to a desired observation in the configuration file
- Step 4: Run the inference algorithm using `python tmnre.py /path/to/config/file.ini`, this will produce a results directory as described below
- Optional step: Run the coverage tests using `python coverage.py /path/to/config/file.ini n_coverage_samples` (`n_coverage_samples = 1000` is usually a good start)

Result output:
- `config_[run_id].txt` - copy of the config file used to generate the run
- `bounds_[run_id]_R[k].txt` - bounds on the individual parameters from Round `k` of the algorithm
- `coverage_[run_id]/` - directory containing the coverage samples if `coverage.py` has been run
- `logratios_[run_id]/` - directory containing the logratios and samples for each round of inference (stored in files `logratios_R[k]` for each round `k`. These can be loaded using the `pickle` python library)
- `observation_[run_id]` - `pickle` file containing the observation used for this run as a `swyft.Sample` object. The same observation is used for both the TMNRE algorithm and any traditional sampling approach.
- `param_idxs_[run_id].txt` - A list of parameter IDs that can be matched to the logratios results files and used for plotting purposes.
- `simulations_[run_id]_R[k]/` - `Zarrstore` directory containing the simulations for Round `k` of inference
- `trainer_[run_id]_R[k]/` - directory containing the information and checkpoints for Round `k` of training the inference network. This directory can also be passed to `tensorboard` as `tensorboard --logdir trainer_[run_id]_R[k]` to investigate the training and validation performance.
- `log_[run_id].log` - Log file containing timing details of run and any errors that were raised

<img align="center" height="250" src="./images/agnostic_10b_samples.png">

- e.g. reconstruction of SGWB using SAQQARA.

## Available Branches:
- `template-powerlaw` - SGWB search using a powerlaw template
- `agnostic` - agnostic SGWB search

## Release Details:
- v0.0.1 | *September 2023* | Public release matching companion paper: 
    - [Simulation-based inference for stochastic gravitational wave background data analysis](https://arxiv.org/abs/2309.07954)
