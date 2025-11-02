import argparse
import glob
import logging
import sys
import datetime 
from datetime import datetime as dt 
import os
from os import path
from os.path import join as pjoin 

from pathlib import Path
import numpy as np
import pandas as pd 
import pyarrow
import kaleido 

import ffmpeg
import m3u8
from create_spectrogram import create_spec_name

import librosa
from skimage.restoration import denoise_wavelet

from orcasound_noise.pipeline.acoustic_util import wav_to_array, array_resampler_bands, spec_to_bands
from orcasound_noise.utils.hydrophone import Hydrophone

def convert_with_ffmpeg(input_file, output_file):
    """Converts input file using ffmpeg."""
    try:
        ffmpeg_input = ffmpeg.input(input_file)
        ffmpeg_output = ffmpeg.output(ffmpeg_input, output_file)
        ffmpeg_output.run(capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        logging.error(e.stdout.decode("utf8"))
        logging.error(e.stderr.decode("utf8"))
        raise e

def create_readable_name(directory, timestamp):
    """
    Creates human readable `.wav` file name from `output_dir` and Unix timestamp.

    Resulting name will look like `directory/%Y-%m-%dT%H-%M-%S.wav`
    """
    return path.join(
        directory,
        f"{dt.fromtimestamp(timestamp, datetime.UTC).strftime('%Y-%m-%dT%H-%M-%S-%f')[:-3]}.wav",
    )


def convert2wav(input_dir, output_dir):
    """
    Converts all `.ts` files from `live.m3u8` to `.wav`.

    All files will have the following format: `%Y-%m-%dT%H-%M-%S.wav`

    Args:
        `input_dir`: Path to the input directory with `.m3u8` playlist and `.ts` files. Should contain Unix timestamp of the stream start.
        `output_dir`: Path to the output directory.
    Returns:
        None
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    playlist = m3u8.load(path.join(input_dir, "live.m3u8"))
    timestamp = float(path.basename(input_dir))
    segments = playlist.data["segments"]
    # print(timestamp, segments[0]["uri"])
    # old_name = path.join(input_dir, segments[0]["uri"])
    # convert_with_ffmpeg(old_name, create_readable_name(output_dir, timestamp))
    for idx, segment in enumerate(segments[1:], start=1):
        print(idx)
        print(segment)
        timestamp += segments[idx - 1]["duration"]
        old_name = path.join(input_dir, segment["uri"])
        print(old_name, segment["uri"])        
        convert_with_ffmpeg(old_name, create_readable_name(output_dir, timestamp))


def wavelet_denoising(spectrogram):
    """In this step we would apply Wavelet-denoising.

    Wavelet denoising is an effective method for SNR improvement
    in environments with wide range of noise types competing for the
    same subspace.

    Wavelet denoising relies on the wavelet representation of
    the image. Gaussian noise tends to be represented by small values in the
    wavelet domain and can be removed by setting coefficients below
    a given threshold to zero (hard thresholding) or
    shrinking all coefficients toward zero by a given
    amount (soft thresholding).

    Args:
        data:Spectrogram data in the form of numpy array.

    Returns:
        Denoised spectrogram data in the form of numpy array.
    """
    im_bayes = denoise_wavelet(spectrogram,
                               convert2ycbcr=False,
                               method="BayesShrink",
                               mode="soft")
    return im_bayes

def array_resampler(df, delta_t=1):
    """
    This function takes in the data frame of spectrogram data, converts it to amplitude, averages over time frame, and converts it back to db.

    Args:
        df: data frame of spectrogram data
        delta_t: Int, number of seconds per sample

    Returns:
        resampled_df: data frame of spectrogram data.
    """
    # Save columns and index for later Dataframe construction
    cols = df.columns
    ind = df.index
    resampled_df = df.to_numpy()
    # Convert back to amplitude for averaging
    resampled_df = librosa.db_to_amplitude(resampled_df)
    resampled_df = pd.DataFrame(resampled_df, columns=cols)
    resampled_df['ind'] = ind
    resampled_df = resampled_df.set_index(pd.DatetimeIndex(resampled_df['ind']))

    sample_length = str(delta_t) + 's'

    # Average over given time span
    resampled_df = resampled_df.resample(sample_length).mean()
    resampledIndex = resampled_df.index

    resampled_df = resampled_df.select_dtypes(include=[np.number]).to_numpy()
    # Convert back to decibels
    resampled_df = librosa.amplitude_to_db(resampled_df, ref=1)
    # Reconstruct Dataframe
    resampled_df = pd.DataFrame(resampled_df, index=resampledIndex)

    return resampled_df

def wav_to_array(filepath, t0=dt.now(), delta_t=1, delta_f=10, 
                 transforms=[wavelet_denoising], ref=1, bands=None):
    
    # Load the .wav file
    y, sr = librosa.load(filepath, sr=None)
    # print(y, sr)

    # Set FFT parameters
    n_fft = int(sr / delta_f)  # FFT size based on frequency resolution
    hop_length = int(n_fft / 2)

    # Apply the STFT (Short-Time Fourier Transform)
    D_highres = librosa.stft(y, hop_length=hop_length, n_fft=n_fft)
    
    # Convert from amplitude to decibels
    spec = librosa.amplitude_to_db(np.abs(D_highres), ref=ref)
    
    # Save the frequencies and time for DataFrame construction
    freqs = librosa.core.fft_frequencies(sr=sr, n_fft=n_fft)
    secs = librosa.core.frames_to_time(np.arange(spec.shape[1]), 
                                       sr=sr, n_fft=n_fft, hop_length=hop_length)
    times = [t0 + datetime.timedelta(seconds=x) for x in secs]

    # Apply transforms (PCEN, wavelet denoising, etc.)
    for transform_func in transforms:
        spec = transform_func(spec)

    # Calculate broadband RMS levels
    # Broadband values reflect raw acoustic energy including noise - why not use cleaned data spec? 
    rms = []
    delta_f = sr / n_fft
    DT = D_highres.transpose()
    for i in range(len(DT)):
        rms.append(delta_f * np.sum(np.abs(DT[i, :])))  # Sum across frequencies

    # Create the PSD Dataframe (cleaned frequency data)
    df = pd.DataFrame(spec.transpose(), columns=freqs, index=times)
    df = df.astype(float).round(2)
    df.columns = df.columns.map(str)

    # Create the broadband dataframe
    rms_df = pd.DataFrame(rms, index=times)
    rms_df = rms_df.astype(float).round(2)
    rms_df.columns = rms_df.columns.map(str)
    # Average over desired time and convert to decibels for the broadband
    rms_df = array_resampler_bands(df=rms_df, delta_t=delta_t)

    # Calculate bands if specified
    if bands is not None:
        # Convert to bands
        oct_unscaled, fm = spec_to_bands(np.abs(DT), bands, delta_f, freqs=freqs, ref=ref)
        oct_df = pd.DataFrame(oct_unscaled, columns=fm, index=times).astype(float).round(2)
        # Average over desired time and convert to decibels for bands
        oct_df = array_resampler_bands(df=oct_df, delta_t=delta_t)
        return oct_df, rms_df

    else:
        # Convert PSD back to amplitude, average over time period, and convert back to decibels
        df = array_resampler(df=df, delta_t=delta_t)
        return df, rms_df

if __name__ == "__main__":
    logging.basicConfig(
        format="%(levelname)s:%(message)s", stream=sys.stdout, level=logging.INFO
    )
    parser = argparse.ArgumentParser(
        description="Creates spectrogram for each .ts file in the input directory."
    )
    parser.add_argument(
        "--input_dir",
        help="Path to the input directory with `.m3u8` playlist and `.ts` files. Should contain Unix timestamp of the stream start.",
    )
    parser.add_argument(
        "--hydrophone", 
        type=str, 
        choices=["bush_point", "orcasound_lab", "port_townsend", "sunset_bay", "sandbox"],
        default="orcasound_lab", 
        help="Name of the hydrophone node", 
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Path to the output directory. Default is `input_dir`.",
    )
    parser.add_argument(
        "-n",
        "--nfft",
        type=int,
        default=256,
        help="The number of data points used in each block for the FFT. A power 2 is most efficient. Default is %(default)s.",
    )

    args = parser.parse_args()
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
    hydrophone_ref = hydrophone.value.bb_ref

    input_parent_dir=pjoin("data/streaming-orcasound-net", hydrophone_name)
    output_parent_dir=pjoin("output", hydrophone_name)
    
    #find latest.txt
    try:
        latest_file=input_parent_dir+"/latest.txt"
        with open(latest_file, "r") as f: 
            latest_dir_name=f.readline().strip()
        print(f"--> Latest Directory is: {latest_dir_name}")    
    except:
        latest_dir_name=None

    # Define Input dir
    input_dir = args.input_dir
    if input_dir is None: 
        input_dir=pjoin(input_parent_dir,latest_dir_name)        
    print(f"--> Input direcotry resolved to: {input_dir}")

    # Define Output dir 
    output_dir=args.output_dir 
    if output_dir is None:
        output_dir=pjoin(output_parent_dir, latest_dir_name) 
    print(f"--> Output directory resolved to {output_dir}")
        
    wav_out=pjoin(output_dir, "wav")
    spec_out=pjoin(output_dir, "png")

    pkl_out=pjoin(output_dir, "pkl")
    pkl_psd_out=pjoin(pkl_out, "psd")
    pkl_bb_out=pjoin(pkl_out, "bb")

    os.makedirs(wav_out, exist_ok=True)    
    os.makedirs(spec_out, exist_ok=True)    
    os.makedirs(pkl_bb_out, exist_ok=True)
    os.makedirs(pkl_psd_out, exist_ok=True)

    # print("strart convert2wav")
    # convert2wav(path.normpath(input_dir), wav_out)
    # print("finished convert2wav")

    for input_wav in sorted(glob.glob(pjoin(wav_out, "*.wav")))[:5]:
        print(f"input_wav: {input_wav}")    
        spec_fname = create_spec_name(input_wav, spec_out)
        print(f"spec_fname: {spec_fname}")
        spec_name=spec_fname.split(sep="/")[-1].split(sep=".")[0]
 
        format_string = "%Y-%m-%dT%H-%M-%S-%f"
        t0 = dt.strptime(spec_name, format_string)
        
        psd_df, bb_df= wav_to_array(input_wav, t0=t0)
        print(psd_df.head())
        print(bb_df.head())
        
        # Rescaling 
        # bb_df = bb_df - hydrophone_ref

        psd_df.to_csv(pjoin(pkl_psd_out, spec_name+".csv"))
        bb_df.to_csv(pjoin(pkl_bb_out, spec_name+".csv"))

        psd_df.to_pickle(pjoin(pkl_psd_out, spec_name+".pickle"))
        bb_df.to_pickle(pjoin(pkl_bb_out, spec_name+".pickle"))
    

#python 02_preprocess.py --input_dir data/streaming-orcasound-net/orcasound_lab/1756796419/ --hydrophone orcasound_lab -o output/orcasound_lab/1756796419
#python 02_preprocess.py --input_dir data/streaming-orcasound-net/port_townsend/1760166020/ --hydrophone port_townsend -o output/port_townsend/1760166020

#python 02_preprocess.py --hydrophone orcasound_lab
