#!/bin/bash

# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <directory-name> <file-name-with-extension>"
    exit 1
fi

# Assign arguments to variables
new_dir_name=$1
file_with_ext=$2

# Extract the file name and extension
file_name="${file_with_ext%.*}"
file_ext="${file_with_ext##*.}"

# Define the results directory relative to the script's execution location
results_dir="./results"

# Create the new directory in the results folder
mkdir -p "$results_dir/$new_dir_name"

# Find all subfolders in the results directory ending with an underscore and a number
for folder in $(find "$results_dir" -type d -regex ".*/.*_[0-9]+$")
do
    # Extract the number from the folder name
    num=$(echo $folder | grep -o '[0-9]*$')
    # Check if the file exists in the subfolder, then copy it
    if [[ -f "$folder/$file_with_ext" ]]; then
        cp "$folder/$file_with_ext" "$results_dir/$new_dir_name/${file_name}_$num.$file_ext"
    fi
done
