# Setting up a *SAQQARA* configuration file

### **Data representation:** 
```
[DATA OPTIONS]
n_chunks = 100
n_grid = 1000
sampling_frequency = 1e-4
minimum_frequency = 1e-4
maximum_frequency = 5e-2
transient_store = ../transient_store
transient_store_size = 200_000
p_transient = 0.0
```
- `n_chunks` | Type: `int` | Number of chunks the data is divided into
- `n_grid` | Type: `int` | Gridding/binning rate for MCMC
- `sampling_frequency` | Type: `float` [Hz]| Sampling frequency of the data
- `minimum_frequency` | Type: `float` [Hz] | Low frequency limit of the data to be analysed
- `maximum_frequency` | Type: `float` [Hz] | High frequency limit of the data to be analysed
- `transient_store` | Type: `str` | Path to store transient data (generated using *ripple*)
- `transient_store_size` | Type: `int` | Number of transients to be simulated
- `p_transient` | Type: `float` | Probability of transient injection in the data stream

### **SGWB injection parameters:**
```
amplitude = -11.
tilt = 0.
```
- `amplitude` | Type: `int` | Amplitude of the power law template (log scale)
- `tilt` | Type:`float` | Tilt of the power law template
- If you want to generate the observation, these parameters will be your ground truth.
- If you want to analyse a pre-existing GW observation, set `generate_obs` in `[TMNRE]` to `False` (In this case, the injection values are irrelevant).
  
### **Prior limits:**
```
[SGWB PRIORS]
amplitude = -20.,-5.
tilt = -10.,10.
```
- Parameter ordering in final results will follow the ordering in this section of the config. (In this particular example, `parameter_0 = amplitude` and `parameter_1 = tilt`)
- Follows `lower_bound,upper_bound` format
- **Please do not leave spaces after the commas to avoid parsing errors**

### **Injected detector noise parameter values**
```
[NOISE INJECTION]
TM = 3.
OMS = 15.
```

### **Detector noise prior limits**
```
[NOISE PRIORS]
TM = 0.,6.
OMS = 0.,30.
```
- Follows `lower_bound,upper_bound` format
- **Please do not leave spaces after the commas to avoid parsing errors**

### **Parameters defining the (`zarr`) store for the waveform simulations**
```
[ZARR PARAMS]
run_id = sgwb_powerlaw
use_zarr = True
sim_schedule = 20_000,20_000
chunk_size = 250
run_parallel = True
njobs = 16
store_path = sgwb_powerlaw
run_description = Implementation of SGWB analysis for power law templates
```
- `run_id` | Type: `str` | Unique identifier for the **saqqara** run (names the output directory and result files)
- `use_zarr` | Type: `bool` | Option to use a zarr store for storing simulations (recommended setting: `True`)
- `sim_schedule` | Type: `int` | Schedule for number of simulations per round of **saqqara**-tmnre 
    - Follows : `n_sims_R1`,`n_sims_R2`,...,`n_sims_RN` where `N` is the number of rounds
- `chunk_size` | Type: `int` | Number of simulations to generate per batch
- `run_parallel` | Type: `bool` | Option to simulate in parallel across cpus
- `njobs` | Type: `int` | number of parallel simulation threads (Defaults to `n_cpus` if `njobs` > `n_cpus` of your machine)
- `store_path` | Type: `str` | Path to the directory to store the simulations
- `run_description` | Type: `str` | Description for the specific **saqqara** run
- **Please do not leave spaces after the commas to avoid parsing errors**

### **TMNRE parameters**
```
[TMNRE]
method = tmnre
one_d = False
marginals = all
num_rounds = 2
infer_only = False
skip_ns = False
resampler = False
noise_targets = data
generate_obs = False
epsilon = 1e-6
logl_th_max = 500.
num_batch_samples = 10
samples_per_slice = 20
num_steps = 4
alpha = 1e-5
obs_path = sgwb_agnostic/observation_sgwb_agnostic
```
- `method` | Type: `str` | Choice between TMNRE/ANRE
- `one_d` | Type: `bool` | Choice of training only the 1D marginals (if `True`, neglects the `marginals` argument)
- `marginals` | Type: `tuple`/`str` | If `one_d` is set to `False`, specify the 2D (or ND) marginals that you want to train or set to `all`
- `num_rounds` | Type: `int` | Number of TMNRE rounds to be executed
- `infer_only` | Type: `bool` | Choice for running only inference if you have a pretrained NN
- `skip_ns` | Type: `bool` | Skip the nested sampling step (only relevant for ANRE method)
- `resampler` | Type: `bool` | Choice for resampling the noise realizations at each training iteration (slow!)
- `noise_targets` | Type: `str` | Noise targets to be used - should comply with the data strains used for training (Default: `data`)
- `generate_obs` | Type: `bool` | Choice to generate the observation before training
- `epsilon` | Type: `float` | Threshold determining the stopping criteria for the ANRE algorithm
- `logl_th_max`| Type: `int` | Threshold determining the maximum log-likelihood for ANRE algorithm
- `num_batch_samples`| Type: `int` | Number of samples per batch in ANRE sampler
- `samples_per_slice`| Type: `int` | Number of samples per slice in ANRE sampler
- `num_steps`| Type: `int` | Number of steps in ANRE sampler
- `alpha` | Type: `int` | Bounds threshold for TMNRE and ANRE truncation step
- `obs_path` | Type: `str` | Path to observation file (loaded as a pickle object) when `generate_obs` is `False`

### **Standard likelihood based sampler parameters (for comparison tests)**
```
[SAMPLING]
mcmc_obs_path = sgwb_agnostic/mcmc_observation_sgwb_agnostic
n_burn = 500
n_steps = 1000
r_conv = 7e-3
iter_max = 50
```
- Requires the `emcee` sampler to be installed on your system
- `mcmc_obs_path` | Type: `str` | Path to mcmc observation file
- `emcee` params with deafult choices
  
### **Hyperparameters for training the NN**
```
[HYPERPARAMS]
min_epochs = 1
max_epochs = 50
early_stopping = 7
learning_rate = 2e-5
num_workers = 0
training_batch_size = 512
validation_batch_size = 512
train_data = 0.9
val_data = 0.1
```
- `min_epochs` | Type: `int` | Minimum number of epochs to train for
- `max_epochs` | Type: `int` | Maximum number of epochs to train for
- `early_stopping` | Type: `int` | Number of training epochs to wait before stopping training in case of overfitting (reverts to the last minimum validation loss epoch)
- `learning_rate` | Type: `float` | The initial learning rate of the trainer
- `num_workers` | Type: `int` | Number of worker processes for loading training and validation data
- `training_bath_size` | Type: `int` | Batch size of the training data to be passed on to the dataloader
- `validation_bath_size` | Type: `int` | Batch size of the validation data to be passed on to the dataloader
- `train_data` | Type: `float` | $\in$ [0,1], fraction of simulation data to be used for training
- `min_epochs` | Type: `float` | $\in$ [0,1], fraction of simulation data to be used for validation/testing

### **Device parameters for training the NN**
```
[DEVICE PARAMS]
device = gpu
n_devices = 1
```
- `device` | Type: `str` | Device on which training is executed (Choice between `gpu` or `cpu`)
- `n_devices` | Type: `int` | Number of devices that the training can be parallelized over
---
