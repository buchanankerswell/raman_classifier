#!/bin/zsh
# Change directory
cd training_images
# Pull random spectra images and save to file
find * -type f | sort -R | tail -n 30 > tmp.txt
# Use magick montage to montage images
montage -label '%d' -geometry +0+0 -tile 6x5 @tmp.txt ../montage.png
# Actinolite
find actinolite -type f | sort -R | tail -n 30 > tmp.txt
# Use magick montage to montage images
montage -geometry +0+0 -tile 6x5 @tmp.txt ../actinolite_montage.png
# Remove temp file
rm tmp.txt
# Open file
open ../montage.png
open ../actinolite_montage.png