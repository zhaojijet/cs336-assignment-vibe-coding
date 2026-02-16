import os
import zipfile
from huggingface_hub import hf_hub_download, list_repo_files


def download_resources():
    repo_id = "hendrycks/competition_math"
    repo_type = "dataset"

    print(f"Listing files in {repo_id}...")
    try:
        files = list_repo_files(repo_id, repo_type=repo_type)
        print(f"Files: {files}")
    except Exception as e:
        print(f"Error listing files: {e}")
        return

    print(f"Downloading README.md from {repo_id}...")
    try:
        path = hf_hub_download(
            repo_id=repo_id, filename="README.md", repo_type=repo_type, local_dir="data"
        )
        print(f"Successfully downloaded README.md to {path}")
    except Exception as e:
        print(f"Error downloading README.md: {e}")

    if "data/MATH.zip" in files:
        print(f"Downloading data/MATH.zip from {repo_id}...")
        try:
            zip_path = hf_hub_download(
                repo_id=repo_id,
                filename="data/MATH.zip",
                repo_type=repo_type,
                local_dir="data",
            )
            print(f"Successfully downloaded MATH.zip to {zip_path}")

            # Unzip
            print("Extracting MATH.zip...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall("data")
            print("Successfully extracted MATH dataset.")

        except Exception as e:
            print(f"Error downloading/extracting MATH dataset: {e}")
    else:
        print("data/MATH.zip not found in repo files.")


if __name__ == "__main__":
    download_resources()
