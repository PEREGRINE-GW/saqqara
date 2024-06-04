#!/bin/bash
cd /home/alveyjbg/Code/public_releases/saqqara/examples/noise_variation
mamba activate default
python single_target_training.py 10000 1e-5 64
python single_target_training.py 10000 5e-5 64
python single_target_training.py 10000 1e-4 64
python single_target_training.py 10000 1e-5 32
python single_target_training.py 10000 1e-5 64
python single_target_training.py 10000 1e-5 128
python single_target_training.py 20000 1e-5 64
python single_target_training.py 20000 5e-5 64
python single_target_training.py 20000 1e-4 64
python single_target_training.py 20000 1e-5 32
python single_target_training.py 20000 1e-5 64
python single_target_training.py 20000 1e-5 128
python single_target_training.py 50000 1e-5 64
python single_target_training.py 50000 5e-5 64
python single_target_training.py 50000 1e-4 64
python single_target_training.py 50000 1e-5 32
python single_target_training.py 50000 1e-5 64
python single_target_training.py 50000 1e-5 128
python single_target_training.py 100000 1e-5 128
python single_target_training.py 100000 5e-5 128
python single_target_training.py 100000 1e-4 128
python single_target_training.py 100000 1e-5 64
python single_target_training.py 100000 1e-5 128
python single_target_training.py 100000 1e-5 256
python single_target_training.py 250000 1e-5 128
python single_target_training.py 250000 5e-5 128
python single_target_training.py 250000 1e-4 128
python single_target_training.py 250000 1e-5 64
python single_target_training.py 250000 1e-5 128
python single_target_training.py 250000 1e-5 256