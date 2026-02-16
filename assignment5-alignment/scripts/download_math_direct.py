import requests
import os
import zipfile


def download_file(url, local_filename):
    print(f"Downloading from {url} to {local_filename}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded {local_filename}")


def main():
    url = "https://huggingface.co/datasets/hendrycks/competition_math/resolve/main/data/MATH.zip"
    zip_path = "data/MATH.zip"

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    try:
        download_file(url, zip_path)

        print("Extracting MATH.zip...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("data")
        print("Successfully extracted MATH dataset.")

    except Exception as e:
        print(f"Error downloading/extracting MATH dataset: {e}")


if __name__ == "__main__":
    main()
