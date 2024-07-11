#!/bin/bash
jobs=1
script='generate_logratios.py'

while getopts 'c:j:qd:' flag; do
  case "${flag}" in
    j) jobs=${OPTARG} ;;
  esac
done
for ((i=0; $i<$jobs; ++i))
do
    python $script &
done
