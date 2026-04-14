import argparse
import os
import re
import tarfile
import urllib.request
from pathlib import Path


def natural_sort_key(path):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(path))]


def collect_jpgs(root_dir):
    jpgs = []
    for root, _, files in os.walk(root_dir):
        for filename in files:
            if filename.lower().endswith(".jpg"):
                jpgs.append(os.path.join(root, filename))
    jpgs.sort(key=natural_sort_key)
    return jpgs


def write_list_file(paths, list_path):
    with open(list_path, "w", encoding="utf-8") as file:
        for path in paths:
            file.write(path + "\n")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Download a pix2pix dataset and generate train/val list files."
    )
    parser.add_argument(
        "--dataset",
        default="facades",
        help="pix2pix dataset name, for example: facades, maps, cityscapes, edges2shoes",
    )
    parser.add_argument(
        "--base-url",
        default="http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/",
        help="base URL for pix2pix datasets",
    )
    parser.add_argument(
        "--datasets-dir",
        default="./datasets",
        help="directory used to store downloaded archives and extracted datasets",
    )
    parser.add_argument(
        "--write-default-lists",
        action="store_true",
        help="also overwrite train_list.txt and val_list.txt with the selected dataset",
    )
    return parser


def main():
    args = build_parser().parse_args()

    dataset_name = args.dataset
    datasets_dir = Path(args.datasets_dir)
    url = f"{args.base_url.rstrip('/')}/{dataset_name}.tar.gz"
    tar_file = datasets_dir / f"{dataset_name}.tar.gz"
    target_dir = datasets_dir / dataset_name

    datasets_dir.mkdir(parents=True, exist_ok=True)

    if tar_file.exists():
        print(f"Compressed file {tar_file} already exists. Skipping download.")
    else:
        print(f"Downloading {url} to {tar_file}")
        try:
            urllib.request.urlretrieve(url, tar_file)
            print("Download completed.")
        except Exception as exc:
            print(f"Download failed: {exc}")
            return

    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"Target directory {target_dir} already exists and is not empty. Skipping extraction.")
    else:
        print(f"Extracting {tar_file} to {datasets_dir}")
        with tarfile.open(tar_file, "r:gz") as tar:
            try:
                tar.extractall(path=datasets_dir, filter="data")
            except TypeError:
                tar.extractall(path=datasets_dir)
        print("Extraction completed.")

    for split in ("train", "val"):
        split_dir = target_dir / split
        if not split_dir.is_dir():
            print(f"Warning: {split_dir} does not exist, skipping {split} list generation.")
            continue

        image_paths = collect_jpgs(split_dir)
        dataset_specific_list_path = Path(f"./{dataset_name}_{split}_list.txt")

        write_list_file(image_paths, dataset_specific_list_path)

        if args.write_default_lists:
            default_list_path = Path(f"./{split}_list.txt")
            write_list_file(image_paths, default_list_path)
            print(
                f"Generated {default_list_path.name} and {dataset_specific_list_path.name} "
                f"with {len(image_paths)} images."
            )
        else:
            print(f"Generated {dataset_specific_list_path.name} with {len(image_paths)} images.")


if __name__ == "__main__":
    main()
