#!/bin/bash

usage()
{
    echo "usage: resize_src_imgs [[[-sd src_dirname ] [-dd dest_dirname ] [-f sizes_filename ] [-h]]"
}

while [ "$1" != "" ]; do
    case $1 in
        -sd | --source_dir )    shift
                                source_dirname=$1
                                ;;
        -dd | --dest_dir )      shift
                                dest_dirname=$1
                                ;;
        -f | --file )           shift
                                sizes_file=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
        * )                     usage
                                exit 1
    esac
    shift
done


while read p; do
    for img in ./${source_dirname}/*; do
       filename=$(basename $img)
       ext=${filename##*.}
       filename=${filename%.*}
       convert $img -interpolate Nearest -filter point -resize ${p}x${p} ./${dest_dirname}/${filename}_${p}.${ext}
    done 
done < ${sizes_file}
