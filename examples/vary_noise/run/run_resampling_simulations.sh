#!/bin/bash
config=''
jobs=1
rundir='./'
script='generate_resampling_data.py'

while getopts 'c:j:d:' flag; do
  case "${flag}" in
    c) config="${OPTARG}" ;;
    j) jobs=${OPTARG} ;;
    d) rundir="${OPTARG}" ;;
  esac
done
for ((i=0; $i<$jobs; ++i))
do
    python $rundir$script -c $config &
done
