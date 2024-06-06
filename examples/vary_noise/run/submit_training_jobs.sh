#!/bin/bash
config=''
jobs=1
q_flag=''
rundir='./'
script='generate_training_data.py'

while getopts 'c:j:qd:' flag; do
  case "${flag}" in
    c) config="${OPTARG}" ;;
    j) jobs=${OPTARG} ;;
    q) q_flag='-q' ;;
    d) rundir="${OPTARG}" ;;
  esac
done
for ((i=0; $i<$jobs; ++i))
do
    python $rundir$script -c $config $q_flag &
done
