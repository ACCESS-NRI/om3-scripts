#!/usr/bin/bash
# Copyright 2024 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0.
#
# Standardise file naming for MOM6 output files in access-om3 by removing the underscore before the four-digit year, i.e., replacing '_YYYY' with 'YYYY'
# This was written assuming it would be used as a payu "userscript" at the "archive" stage, but alternatively a path to an "archive" directory can be provided.
# For more details, see https://github.com/COSIMA/om3-scripts/issues/32

Help()
{
    # Display help
    echo -e "Standardise file naming for MOM6 output files.\n"
    echo "Syntax: scriptTemplate [-h|d DIRECTORY]"
    echo "options:"
    echo "h    Print this help message."
    echo -e "d    Process files in the specified 'DIRECTORY'."
}

while getopts ":hd:" option; do
    case $option in
        h) # display help
           Help
           exit;;
        d) # Enter a directory
            out_dir=$OPTARG
            if [ ! -d $out_dir ]; then
                echo $out_dir Does not exist
                exit
            fi;;
        \?) # Invalid option
            echo "Error: Invalid option"
            exit;;
    esac
done

# if no directory was specified, collect all directories from 'archive'
if [ -z $out_dir ]; then
    out_dirs=$(ls -rd archive/output*[0-9] 2>/dev/null)
else
    out_dirs=$out_dir
fi

# process each output directory
for dir in ${out_dirs[@]}; do
    # process each mom6 file
    for current_file in $dir/access-om3.mom6.*.nc*; do
       if [ -f $current_file ]; then
            new_filename=$(echo $current_file | sed -E 's/_([0-9]{4})/\1/')
                # rename the file without overwriting existing files
                mv -n $current_file $new_filename
        fi
    done
done
