#!/usr/bin/env python3
"""
This script automates the downloading and extraction of the KITTI dataset.
It requires authentication via a user cookie, which can be copied from a browser session.

Usage:
    python3 0_download_kitti_dataset.py --kitti_user KITTI_USER

Arguments:
    --kitti_user (str): The KITTI_USER cookie value for authentication. This argument is required.

Output:
    The KITTI dataset files are downloaded and extracted into the following directory structure:
    - datasets/kitti/images: Contains image data.
    - datasets/kitti/labels: Contains label data.

References:
    - https://www.cvlibs.net/datasets/kitti/
"""

import argparse
import os
import requests
import tempfile
import zipfile

def download_and_unpack_zip(url, cookies, output_folder, fname=None):
    cookies = cookies if cookies else {}

    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
        save_path = tmp_file.name
    try:
        response = requests.get(url, cookies=cookies, stream=True)
        response.raise_for_status()

        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=128):
                file.write(chunk)

        print(f"ZIP file successfully downloaded to {save_path}")

        if not zipfile.is_zipfile(save_path):
            raise zipfile.BadZipFile("The downloaded file is not a valid ZIP file.")

        os.makedirs(output_folder, exist_ok=True)
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(output_folder)
        print(f"ZIP file successfully unpacked to {output_folder}")

        if fname:
            final_path = os.path.join(output_folder, fname)
            os.rename(save_path, final_path)
            print(f"ZIP file moved to {final_path}")

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except zipfile.BadZipFile as e:
        print(f"Error with ZIP file: {e}")
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f"Temporary ZIP file {save_path} has been deleted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and unpack the KITTI dataset."
    )
    parser.add_argument("--kitti_user", type=str, required=True, help="The KITTI_USER cookie value for authentication. This argument is required.")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(os.path.dirname(script_dir), 'datasets')
    kitti_dir = os.path.join(datasets_dir, 'kitti')

    args = parser.parse_args()

    kitti_zip_urls = [
        # (url, subfolder)
        ("https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip", "images"),
        ("https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip", "labels"),
    ]

    cookies = {'KITTI_USER': args.kitti_user}
    
    for url, subfolder in kitti_zip_urls:
        print(f"Downloading and unpacking {url} into {os.path.join(kitti_dir, subfolder)} ...")
        dest_path = os.path.join(kitti_dir, subfolder)
        download_and_unpack_zip(url, cookies, dest_path)

    print("All KITTI downloads complete.")
