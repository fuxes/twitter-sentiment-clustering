#!/bin/bash

for file in *
do
  echo $file
  # sed -n '/^done/,$p' $file > $file.tmp; mv $file.tmp $file
  # awk '/Cluster /' $file | cat - $file > $file.tmp; mv $file.tmp $file
done
