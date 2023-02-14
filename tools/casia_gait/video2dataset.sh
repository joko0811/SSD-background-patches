#!/bin/bash
# Step 1

video_dir="video/"
image_dir="image/"

mkdir $1$image_dir

tmp='*'
file_pathes="$(find $1$video_dir$tmp -maxdepth 0 -type f)"

for file_path in $file_pathes; do
  file_name="$(basename $file_path)"
  image_path="$1$image_dir${file_name%.*}"

  ffmpeg -i $file_path -vf fps=25 $image_path-%03d.png
done

