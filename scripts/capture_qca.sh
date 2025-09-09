#!/bin/bash

repo_path=$1
file=$2

sudo "${repo_path}"/extractor/build/csi-extractor "${file}"
