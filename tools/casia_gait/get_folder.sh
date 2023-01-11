#!/bin/bash

tmp='*'
mask_dir="mask/"
mkdir $mask_dir
file_dirs="$(find -maxdepth 3 -type d)"

for file_dir in $file_dirs; do
  num="$(echo $file_dir | awk -F '/' '{print $NF}')"
  if [[ $num == "090" ]]; then
    cp $file_dir/$tmp $mask_dir
  fi
done
