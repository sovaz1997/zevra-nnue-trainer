#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 base_output_filename"
    exit 1
fi

base_output_csv="$1"

input_csv="./billion-dataset.fen"

startPoint=0
lastPoint=10000000
threads=12
batchSize=$(( (lastPoint - startPoint) / threads ))

for ((i=1; i<=threads; i++)); do
  start=$(( (i - 1) * batchSize + startPoint ))
  end=$(( i * batchSize + startPoint ))

  echo "Processing lines: $start - $end"

  output_csv="${base_output_csv}_${start}_${end}.csv"


  python3 reevaluate.py "$input_csv" "$output_csv" "$start" "$end" &
done

wait
