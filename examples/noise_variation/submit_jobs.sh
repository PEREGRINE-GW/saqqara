#!/bin/bash
cd /home/alveyjbg/Code/public_releases/saqqara/examples/noise_variation
mamba activate default
INSTANCES=8
for ((i=0; $i<$INSTANCES; ++i))
do
    python generate_bounded_data.py &
done