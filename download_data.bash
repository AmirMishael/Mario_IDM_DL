#!/bin/bash
# Download the dataset from github to mario_dataset folder
pip install kornia
mkdir "./tmp_download"
wget "https://github.com/rafaelcp/smbdataset/raw/main/data-smb.7z" -O "./tmp_download/data-smb.7z"
7za x "./tmp_download/data-smb.7z" -o./mario_dataset -y -mmt=8
rm -rf "./tmp_download"
