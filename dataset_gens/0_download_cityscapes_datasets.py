#!/usr/bin/env python3
"""
This script automates the process of downloading and extracting the Cityscapes dataset,
including its variants like Foggy Cityscapes and Rainy Cityscapes. It requires a
PHP session ID for authentication, which can be obtained from the browser's developer
tools after logging into the Cityscapes website.

Usage:
    python3 0_download_cityscapes_datasets.py <session_id>

Arguments:
    session_id (str): The PHP session ID used for authentication during download.

Output:
    The datasets are downloaded and extracted into the following directories:
    - datasets/cityscapes: Original Cityscapes dataset.
    - datasets/foggy_cityscapes: Foggy Cityscapes dataset.
    - datasets/rainy_cityscapes: Rainy Cityscapes dataset.

References:
    - https://www.cityscapes-dataset.com
"""

import argparse
import os
import requests
import tempfile
import zipfile

def download_and_unpack_zip(url, session_id, output_folder):
    headers = {
        "Cookie": f"PHPSESSID={session_id}", # can get it from browser dev tools
    }

    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
        save_path = tmp_file.name
    try:
        response = requests.get(url, headers=headers, stream=True)
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

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except zipfile.BadZipFile as e:
        print(f"Error with ZIP file: {e}")
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f"Temporary ZIP file {save_path} has been deleted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and unpack a ZIP file from the Cityscapes dataset downloads page.")
    parser.add_argument("session_id", type=str, required=True, help="The PHP session ID to use for downlading the original Cityscapes dataset")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(os.path.dirname(script_dir), 'datasets')
    cityscapes_dir = os.path.join(datasets_dir, 'cityscapes')
    foggy_cityscapes_dir = os.path.join(datasets_dir, 'foggy_cityscapes')
    rainy_cityscapes_dir = os.path.join(datasets_dir, 'rainy_cityscapes')

    args = parser.parse_args()
    
    # Download and unpack the Cityscapes dataset
    for url in [
        "https://www.cityscapes-dataset.com/file-handling/?packageID=1",
        "https://www.cityscapes-dataset.com/file-handling/?packageID=3"
    ]:
        print(f"Downloading and unpacking {url}...")
        download_and_unpack_zip(url, args.session_id, cityscapes_dir)

    # Download and unpack the foggy Cityscapes dataset
    for url in [
        "https://www.cityscapes-dataset.com/file-handling/?packageID=31"
    ]:
        print(f"Downloading and unpacking {url}...")
        download_and_unpack_zip(url, args.session_id, foggy_cityscapes_dir)

    # Download and unpack the rainy Cityscapes dataset
    for url in [
        "https://www.cityscapes-dataset.com/file-handling/?packageID=33"
    ]:
        print(f"Downloading and unpacking {url}...")
        download_and_unpack_zip(url, args.session_id, rainy_cityscapes_dir)
