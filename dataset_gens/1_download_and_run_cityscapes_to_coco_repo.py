#!/usr/bin/env python3
"""
Extracting CoCo-style labels from original Cityscapes labels using code from a public GitHub repository.

Usage:
    python3 1_download_and_run_cityscapes_to_coco_repo.py

Requirements:
    - The Cityscapes dataset path:
      `../datasets/cityscapes`
Output:
    - The script will generate output annotations in the following directory:
      `../datasets/cityscapes/annotations`

References:
    https://github.com/TillBeemelmanns/cityscapes-to-coco-conversion
"""

import argparse
import os
import requests
import zipfile
import subprocess
import sys
import io

def download_and_run_github_repo(github_url, main_file_path, script_args):
    zip_url = f"{github_url}"
    response = requests.get(zip_url, stream=True)
    response.raise_for_status()

    zip_file_path = "temp_repo.zip"

    with open(zip_file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        temp_dir = "."
        zip_ref.extractall(temp_dir)

    extracted_dir = os.path.join(temp_dir, os.listdir(temp_dir)[0])

    main_file_full_path = os.path.join(extracted_dir, main_file_path)

    if os.path.isfile(main_file_full_path):

        with open(main_file_full_path, 'r') as file:
            content = file.read()

        # Make it less verbose
        modified_content = content.replace("print('Warning: invalid contours.')", "# print('Warning: invalid contours.')")
        with open(main_file_full_path, 'w') as file:
            file.write(modified_content)
        subprocess.run([sys.executable, main_file_full_path] + script_args)
    else:
        print(f"Main file not found at {main_file_full_path}")

    os.remove(zip_file_path)
    print(f"ZIP file {zip_file_path} has been removed.")

if __name__ == '__main__':

    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(os.path.dirname(script_dir), 'datasets')
    cityscapes_dir = os.path.join(datasets_dir, 'cityscapes')
    foggy_cityscapes_dir = os.path.join(datasets_dir, 'foggy_cityscapes')

    github_repo_url = "https://github.com/TillBeemelmanns/cityscapes-to-coco-conversion/archive/refs/heads/master.zip"
    main_file = "main.py"  

    script_args = [
        '--dataset', 'cityscapes',
        '--outdir', os.path.join(cityscapes_dir, 'annotations'),
        '--datadir', cityscapes_dir
    ]

    download_and_run_github_repo(github_repo_url, main_file, script_args)
