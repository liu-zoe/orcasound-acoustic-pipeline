import os
import sys
from pathlib import Path
import argparse
from orcasound_noise.utils.hydrophone import Hydrophone
import boto3
from botocore import UNSIGNED
from botocore.config import Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Download .ts files that falls within a time window"
    )
    parser.add_argument(
        "--hydrophone", 
        type=str, 
        choices=["bush_point", "orcasound_lab", "port_townsend", "sunset_bay", "sandbox"],
        default="orcasound_lab", 
    )
    parser.add_argument(
        "--input_dir",
        help="Specific directory name to download (e.g., '1756796419'). If not provided, downloads the latest directory from latest.txt.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output directory for spectrograms. Default is `input_dir`.",
    )

    args = parser.parse_args()
    # now = dt.now(pytz.timezone('US/Pacific'))

    #Set up parameters based on hydrophone 
    # Mapping from user-friendly names to enum values 
    hydrophone_map = {
        'bush_point': Hydrophone.BUSH_POINT,
        'orcasound_lab': Hydrophone.ORCASOUND_LAB,
        'port_townsend': Hydrophone.PORT_TOWNSEND,
        'sunset_bay': Hydrophone.SUNSET_BAY,
        'sandbox': Hydrophone.SANDBOX
    }
    hydrophone = hydrophone_map[args.hydrophone]
    hydrophone_name = hydrophone.value.name 
    hydrophone_bucket = hydrophone.value.bucket
    hydrophone_reffolder= hydrophone.value.ref_folder
    latest_filename = "latest.txt"

    print(hydrophone_bucket)
    print(hydrophone_name)
    print(hydrophone_reffolder)

    output_path=args.output

    # Determine which directory to download
    if args.input_dir:
        # Use the specified input directory
        latest_dir_name = args.input_dir
        print(f"Using specified directory: {latest_dir_name}")
    else:
        # Download and read latest.txt to get the latest directory
        latest_filename = "latest.txt"
        latest_aws_path = hydrophone_reffolder + "/" + latest_filename
        latest_dl_path = output_path + "/" + latest_filename

        # Create S3 client with no signature (for public buckets)
        s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        
        try:
            s3_client.download_file(hydrophone_bucket, latest_aws_path, latest_dl_path)
            print(f"File '{latest_aws_path}' downloaded to '{latest_dl_path}' successfully.")
        except Exception as e:
            print(f"Error downloading file: {e}")
            sys.exit(1)

        with open(latest_dl_path, mode='r') as f:
            latest_dir_name = f.readline().strip()
        print(f"Latest directory from latest.txt: {latest_dir_name}")

    # Create S3 client (reuse or create if not already created)
    if not args.input_dir:
        # S3 client already created above
        pass
    else:
        s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    # path to the latest directory 
    dir_aws_path=hydrophone_reffolder+"/hls/"+latest_dir_name
    dir_dl_path=output_path+"/"+latest_dir_name
    os.makedirs(dir_dl_path, exist_ok=True)
    print(dir_aws_path)

    # Download the latest directory
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=hydrophone_bucket, Prefix=dir_aws_path)

    print(paginator)

    for page in pages:
        if "Contents" in page:
            for obj in page['Contents']:
                s3_key = obj['Key']
                # Skip if it's a "directory" placeholder
                if s3_key.endswith('/'):
                    continue
                # Construct the local file path
                relative_path = os.path.relpath(s3_key, dir_aws_path)
                local_file_path = os.path.join(dir_dl_path, relative_path)
                print(relative_path)
                print(local_file_path)

                # Create any necessary subdirectories locally
                Path(local_file_path).parent.mkdir(parents=True, exist_ok=True)

                print(f"Downloading {s3_key} to {local_file_path}")
                s3_client.download_file(hydrophone_bucket, s3_key, local_file_path)

# Example usage:
# Download latest directory:
# python 01_download.py --hydrophone orcasound_lab -o data/streaming-orcasound-net/orcasound_lab
 
# Download specific directory:
# python 01_download.py --hydrophone port_townsend --input_dir 1760166020 -o output/port_townsend/test
