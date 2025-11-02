"""
Test script for AudioCPAExtractor.process_cpa_dataframe()

This script demonstrates how to use the process_cpa_dataframe() function
with the test AIS data containing CPA timestamps.
"""

import os 
import pandas as pd
import numpy as np
from pathlib import Path
import wave
import struct
from datetime import datetime, timedelta
import sys
import glob 
from audio_cpa_extractor import AudioCPAExtractor

def test_process_cpa_dataframe():
    """
    Main test function demonstrating the process_cpa_dataframe() usage.
    """
    print("=" * 80)
    print("AudioCPAExtractor Test - process_cpa_dataframe()")
    print("=" * 80)
    ais_data=pd.read_csv(os.path.join(output_dir, "test_ais_raw.csv"))
    audio_filenames=glob.glob(os.path.join(output_dir,"**", "*.wav"), recursive=True)
    for x in audio_filenames[:3]:
        print(x)
    print("\n3. Initializing AudioCPAExtractor...")
    extractor = AudioCPAExtractor(
        audio_folder=str(audio_folder),
        audio_filenames=audio_filenames
    )
    print(f"   Parsed {len(extractor.audio_metadata)} audio file timestamps")
    output_audio_folder=os.path.join(output_dir, "test_cpa_extractor")
    os.mkdir(output_audio_folder)
    results_df = extractor.process_cpa_dataframe(
            df=ais_data,
            timestamp_column='t_cpa',
            output_folder=output_audio_folder,
            clip_duration=30.0  # 30-second clips
        )
    return results_df


if __name__ == "__main__":
    # Run the main test
    output_dir="output/port_townsend"
    audio_folder = Path('output/port_townsend/1760166020/wav')

    results = test_process_cpa_dataframe()
    results.to_csv(os.path.join(output_dir, "test_cpa_extractor_res.csv"), index=False)
    