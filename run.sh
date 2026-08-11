#!/usr/bin/env bash
set -e

mkdir -p data/input data/output artifacts

python3 scripts/generate_dataset.py \
  --output data/input \
  --count 1000 \
  --width 128 \
  --height 128

if [ ! -f cuda_batch_processor ]; then
  make
fi

./cuda_batch_processor \
  --input data/input \
  --output data/output \
  --mode grayscale-invert \
  --limit 1000 \
  --threads 256 | tee artifacts/execution_log.txt

cp data/input/image_0000.ppm artifacts/input_sample.ppm
cp data/output/image_0000.ppm artifacts/output_sample.ppm
