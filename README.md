# Orcasound Acoustic Pipeline

## What This Project Does

This toolkit enables researchers to download, process, and analyze underwater acoustic data from the Orcasound hydrophone network in the Pacific Northwest. It automates the entire pipeline from raw streaming audio to analyzable spectrograms and noise level measurements.

## Key Capabilities

### 1. Data Acquisition
- Download streaming audio (`.ts` files) from AWS S3 buckets
- Access data from 5 hydrophone locations
- Retrieve latest data or historical archives

### 2. Audio Processing
- Convert streaming formats to standard WAV files
- Apply signal processing: FFT, STFT, wavelet denoising
- Generate power spectral density (PSD) representations
- Calculate broadband noise levels

### 3. Event-Based Extraction
- Extract audio clips around specific timestamps
- Process vessel Closest Point of Approach (CPA) events
- Handle multi-file clip extraction for longer events
- Batch process multiple events from dataframes

## File Structure

### Processing Scripts
- `01_download.py` - Download audio from S3
- `02_preprocess.py` - Convert and process audio
- `audio_cpa_extractor.py` - Timestamp-based audio clip extraction

### Supporting Modules
- `create_spectrogram.py` - Generate spectrogram images
- `hydrophone.py` - Hydrophone configurations (Bush Point, Orcasound Lab, Port Townsend, Sunset Bay, Sandbox)
- `orcasound_noise/pipeline/acoustic_util.py` - Signal processing utilities (FFT, denoising, band analysis)
- `orcasound_noise/pipeline/pipeline.py` - End-to-end processing pipeline
- `orcasound_noise/utils/file_connector.py` - AWS S3 operations and file management

### Test scripts
- `test_audio_cpa_extractor.py` - Test CPA extraction

## Typical Use Cases

### Research Applications
1. **Ambient Noise Monitoring** - Track long-term noise level trends
2. **Vessel Noise Analysis** - Extract and analyze ship passage sounds
3. **Marine Mammal Studies** - Analyze frequency bands relevant to whale calls
4. **Anthropogenic Impact** - Quantify human-generated underwater noise
5. **Habitat Acoustics** - Characterize soundscapes in marine environments

### Analysis Types
- Spectral analysis (frequency content over time)
- Broadband level trends (overall sound intensity)
- Octave band analysis (ISO standard frequency groupings)
- Event detection and extraction
- Baseline ambient noise calculations

## Technology Stack

**Languages:** Python 3.8+

**Key Libraries:**
- **Audio:** librosa, soundfile, ffmpeg-python
- **Signal Processing:** scipy, numpy, scikit-image, PyWavelets
- **Data:** pandas, pyarrow
- **Cloud:** boto3 (AWS S3)
- **Visualization:** matplotlib, plotly

**External Tools:**
- FFmpeg (command-line audio processing)

## Data Flow

```
AWS S3 Buckets
    ↓
[01_download.py]
    ↓
.ts files (MPEG transport stream)
    ↓
[02_preprocess.py]
    ↓
.wav files → [acoustic_util.py] → PSD data
                                 → Broadband data
    ↓
CSV/Pickle/Parquet files
    ↓
[audio_cpa_extractor.py] → Event clips
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data
python 01_download.py --hydrophone orcasound_lab -o data/orcasound_lab

# 3. Process audio
python 02_preprocess.py --hydrophone orcasound_lab -o output/orcasound_lab

```
## Credits
This toolkit utilizes code from the [ambient-sound-analysis repo](https://github.com/orcasound/ambient-sound-analysis), specifically the the [pipeline module](https://github.com/orcasound/ambient-sound-analysis/tree/main/src/orcasound_noise/pipeline) and the [utils module] (https://github.com/orcasound/ambient-sound-analysis/tree/main/src/orcasound_noise/utils).

## Acknowledgments

Developed for the Orcasound project, which provides open access to underwater acoustic data from hydrophones in the Pacific Northwest. The network monitors marine soundscapes and supports research on marine mammals, vessel noise, and ocean acoustics.

## Related Resources

- [Orcasound](https://www.orcasound.net/) - Live hydrophone streams
- [Orcasound GitHub](https://github.com/orcasound) - Project repositories
- [AWS Open Data](https://registry.opendata.aws/orcasound/) - Orcasound data on AWS

---
