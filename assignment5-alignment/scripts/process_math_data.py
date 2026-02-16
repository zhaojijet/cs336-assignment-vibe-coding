import json
import os
import glob
import pandas as pd
from pathlib import Path


def process_math_data(input_path: str, output_file: str):
    """
    Process MATH dataset from parquet file to a single JSONL file.
    Use last 1000 examples as validation set if no explicit test set found.
    """
    print(f"Processing MATH data from {input_path} to {output_file}...")

    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        # Read parquet file
        df = pd.read_parquet(input_path)
        print(f"Read {len(df)} rows from parquet file.")

        # Take last 1000 rows as validation set
        validation_df = df.iloc[-1000:]
        print(f"Selected {len(validation_df)} examples for validation.")

        # Convert to list of dicts
        examples = validation_df.to_dict(orient="records")

    except Exception as e:
        print(f"Error reading/processing parquet file: {e}")
        return

    # Write to JSONL
    with open(output_file, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    print(f"Successfully wrote to {output_file}")


if __name__ == "__main__":
    # Default paths based on previous download commands

    base_dir = (
        "data/competition_math/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet"
    )
    output_path = "data/MATH/validation.jsonl"

    process_math_data(base_dir, output_path)
