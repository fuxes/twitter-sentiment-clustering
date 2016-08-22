#!/bin/bash

function run_pipeline {
  output_base="oscars/solutions/"
  output_file="$output_base$2-$3-$4-$5${6:+-stem}${7:+-tfidf}${8:+-bin}.out"

  echo "Writing to $output_file"
  time python preprocessing.py \
    --data $1 \
    --min_df $2  \
    --max_df $3 \
    --n_clusters $4 \
    --n_grams $5 \
    ${6:+--stem} \
    ${7:+--idf} \
    ${8:+--bin} \
    --
    # > $output_file
}

function run_experiments {
  data="oscars/oscars_trim.json"
  min_df=$1
  max_df=$2
  n_clusters="(4,9)"
  stem=
  idf=
  bin=
  n_grams="(1,1)"
  run_pipeline "$data" "$min_df" "$max_df" "$n_clusters" "$n_grams" "$stem" "$idf" "$bin"

  n_grams="(1,2)"
  run_pipeline "$data" "$min_df" "$max_df" "$n_clusters" "$n_grams" "$stem" "$idf" "$bin"

  n_grams="(1,3)"
  run_pipeline "$data" "$min_df" "$max_df" "$n_clusters" "$n_grams" "$stem" "$idf" "$bin"

  n_grams="(2,3)"
  run_pipeline "$data" "$min_df" "$max_df" "$n_clusters" "$n_grams" "$stem" "$idf" "$bin"

  stem=true
  idf=true
  n_grams="(1,1)"
  run_pipeline "$data" "$min_df" "$max_df" "$n_clusters" "$n_grams" "$stem" "$idf" "$bin"

  n_grams="(1,2)"
  run_pipeline "$data" "$min_df" "$max_df" "$n_clusters" "$n_grams" "$stem" "$idf" "$bin"

  n_grams="(1,3)"
  run_pipeline "$data" "$min_df" "$max_df" "$n_clusters" "$n_grams" "$stem" "$idf" "$bin"

  n_grams="(2,3)"
  run_pipeline "$data" "$min_df" "$max_df" "$n_clusters" "$n_grams" "$stem" "$idf" "$bin"

  bin=true
  n_grams="(1,1)"
  run_pipeline "$data" "$min_df" "$max_df" "$n_clusters" "$n_grams" "$stem" "$idf" "$bin"

  n_grams="(1,2)"
  run_pipeline "$data" "$min_df" "$max_df" "$n_clusters" "$n_grams" "$stem" "$idf" "$bin"

  n_grams="(1,3)"
  run_pipeline "$data" "$min_df" "$max_df" "$n_clusters" "$n_grams" "$stem" "$idf" "$bin"

  n_grams="(2,3)"
  run_pipeline "$data" "$min_df" "$max_df" "$n_clusters" "$n_grams" "$stem" "$idf" "$bin"
}

function run_all_experiments {
  max_df=0.01
  for min_df in `seq 1 5`;
  do
    time run_experiments "$min_df" "$max_df"
  done

  max_df=1.0
  for min_df in `seq 1 5`;
  do
    time run_experiments "$min_df" "$max_df"
  done

  min_df=1
  for max_df in `seq 5 10`;
  do
    time run_experiments "$min_df" "$max_df"
  done
}

function run_best {
  data="oscars/oscars_trim.json"
  min_df=4
  max_df=1.0
  n_clusters="8"
  stem=true
  idf=true
  bin=
  n_grams="(1,3)"
  run_pipeline "$data" "$min_df" "$max_df" "$n_clusters" "$n_grams" "$stem" "$idf" "$bin"
}

time run_best
# time run_all_experiments