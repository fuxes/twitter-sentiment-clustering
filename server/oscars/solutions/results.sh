#!/bin/bash

for file in *.out
do
  awk '/Cluster /' $file | cat - $file > $file.tmp; mv $file.tmp $file
  awk '/Cluster 0/ {$0="\n"$0} 1' $file > $file.tmp; mv $file.tmp $file
  awk "/\('For clusters /" $file | cat - $file > $file.tmp; mv $file.tmp $file
done
  # sed -n '/^done/,$p' $file > $file.tmp; mv $file.tmp $file

 # sed "/'For clusters ='\, 8\, 'The average silhouette_score/q" $file  > $file.tmp; mv $file.tmp $file