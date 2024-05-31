#!/usr/bin/env bash

CONFIG=$1
# CONFIG=cora_norm
# GRID=${GRID:-example}
# REPEAT=${REPEAT:-3}
# MAX_JOBS=${MAX_JOBS:-8}
# SLEEP=${SLEEP:-1}
# MAIN=${MAIN:-main}
GRID=grid
REPEAT=5 
MAX_JOBS=1
SLEEP=1
MAIN=main
OUTDIR=grid_search

# generate configs (after controlling computational budget)
# please remove --config_budget, if don't control computational budget
python configs_gen.py --config configs/${CONFIG}.yaml \
  --grid grids/grid.txt \
  --out_dir configs/${OUTDIR}
#python configs_gen.py --config configs/ChemKG/${CONFIG}.yaml --config_budget configs/ChemKG/${CONFIG}.yaml --grid grids/ChemKG/${GRID}.txt --out_dir configs
# run batch of configs
# Args: config_dir, num of repeats, max jobs running, sleep time
bash parallel.sh configs/${OUTDIR}/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN
# rerun missed / stopped experiments
# bash parallel.sh configs/grid_search/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN
# rerun missed / stopped experiments
bash parallel.sh configs/${OUTDIR}/${CONFIG}_grid_${GRID} $REPEAT $MAX_JOBS $SLEEP $MAIN

# aggregate results for the batch
python agg_batch.py --dir results/${CONFIG}_grid_${GRID}
