import kagglehub
import os
import shutil

target_path = "src/data/vehicle-dataset"

def main() -> None:
    if os.path.exists(target_path) and os.listdir(target_path):
        print("Dataset already exists. Skipping download.")
    else:
        print("Downloading dataset from kagglehub...")
        download_path = kagglehub.dataset_download("nehalbirla/vehicle-dataset-from-cardekho")
        
        os.makedirs(target_path, exist_ok=True)
        shutil.copytree(download_path, target_path, dirs_exist_ok=True)
        print("Dataset downloaded and extracted to:", target_path)

