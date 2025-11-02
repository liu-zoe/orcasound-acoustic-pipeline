"""
Audio CPA Extractor
Extracts audio clips around CPA (Closest Point of Approach) timestamps from AIS data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional
import wave
import struct


class AudioCPAExtractor:
    """Extract audio clips around CPA timestamps from audio files."""
    
    def __init__(self, audio_folder: str, audio_filenames: List[str]):
        """
        Initialize the extractor.
        
        Args:
            audio_folder: Path to folder containing .wav files
            audio_filenames: List of audio filenames
        """
        self.audio_folder = Path(audio_folder)
        self.audio_filenames = audio_filenames
        self.audio_metadata = self._parse_audio_files()
    
    def _parse_audio_files(self) -> pd.DataFrame:
        """
        Parse audio filenames to extract timestamps and create metadata.
        
        Returns:
            DataFrame with columns: filename, timestamp, filepath
        """
        metadata = []
        
        for filename in self.audio_filenames:
            timestamp = self._parse_filename_timestamp(filename)
            if timestamp:
                # Handle both full paths and just filenames
                filepath = Path(filename)
                if not filepath.is_absolute() and not str(filename).startswith('output'):
                    # If it's just a filename, prepend the audio folder
                    filepath = self.audio_folder / filename
                
                metadata.append({
                    'filename': Path(filename).name,  # Store just the filename
                    'full_path': str(filename),  # Store the original path
                    'timestamp': timestamp,
                    'filepath': filepath
                })
        
        df = pd.DataFrame(metadata)
        if not df.empty:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    @staticmethod
    def _parse_filename_timestamp(filename: str) -> Optional[datetime]:
        """
        Parse timestamp from filename format: 2025-10-11T07-00-30-005.wav
        Handles both full paths and just filenames.
        
        Args:
            filename: Audio filename (with or without path)
            
        Returns:
            datetime object or None if parsing fails
        """
        try:
            # Extract just the filename from the path
            filename_only = Path(filename).name
            
            # Remove .wav extension and split
            name_part = filename_only.replace('.wav', '')
            
            # Split on 'T' to separate date and time
            date_part, time_part = name_part.split('T')
            
            # Parse date: 2025-10-11
            year, month, day = map(int, date_part.split('-'))
            
            # Parse time: 07-00-30-005 (hour-minute-second-millisecond)
            time_components = time_part.split('-')
            hour = int(time_components[0])
            minute = int(time_components[1])
            second = int(time_components[2])
            millisecond = int(time_components[3]) if len(time_components) > 3 else 0
            
            return datetime(year, month, day, hour, minute, second, millisecond * 1000)
        
        except (ValueError, IndexError) as e:
            print(f"Warning: Could not parse timestamp from {filename}: {e}")
            return None
    
    @staticmethod
    def _parse_cpa_timestamp(timestamp_str: str) -> datetime:
        """
        Parse CPA timestamp from ISO format: 2025-10-11T07:30:20Z
        
        Args:
            timestamp_str: ISO format timestamp string
            
        Returns:
            datetime object
        """
        # Remove 'Z' suffix if present
        timestamp_str = timestamp_str.rstrip('Z')
        
        # Parse ISO format
        return pd.to_datetime(timestamp_str).to_pydatetime()
    
    def _find_audio_file_for_timestamp(self, target_timestamp: datetime) -> Optional[Tuple[int, datetime]]:
        """
        Find the audio file that contains the target timestamp.
        
        Args:
            target_timestamp: The CPA timestamp to find
            
        Returns:
            Tuple of (file_index, file_timestamp) or None if not found
        """
        if self.audio_metadata.empty:
            return None
        
        # Find the file whose timestamp is closest but not after the target
        # (assuming each file contains audio starting from its timestamp)
        valid_files = self.audio_metadata[
            self.audio_metadata['timestamp'] <= target_timestamp
        ]
        
        if valid_files.empty:
            return None
        
        # Get the most recent file before or at the target timestamp
        closest_idx = valid_files['timestamp'].idxmax()
        
        return closest_idx, self.audio_metadata.loc[closest_idx, 'timestamp']
    
    def _get_audio_duration(self, filepath: Path) -> float:
        """
        Get duration of a WAV file in seconds.
        
        Args:
            filepath: Path to WAV file
            
        Returns:
            Duration in seconds
        """
        try:
            with wave.open(str(filepath), 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                return frames / float(rate)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return 0.0
    
    def extract_clip_around_cpa(
        self, 
        cpa_timestamp: str, 
        clip_duration: float = 30.0,
        output_path: Optional[str] = None
    ) -> Optional[dict]:
        """
        Extract audio clip around a CPA timestamp.
        
        Args:
            cpa_timestamp: CPA timestamp string (ISO format)
            clip_duration: Total duration of clip in seconds (default: 30)
            output_path: Optional path to save the extracted clip
            
        Returns:
            Dictionary with extraction metadata or None if extraction failed
        """
        target_dt = self._parse_cpa_timestamp(cpa_timestamp)
        
        # Find the audio file containing this timestamp
        result = self._find_audio_file_for_timestamp(target_dt)
        
        if result is None:
            print(f"No audio file found for timestamp {cpa_timestamp}")
            return None
        
        file_idx, file_start_dt = result
        filepath = self.audio_metadata.loc[file_idx, 'filepath']
        
        # Calculate offset from start of file
        time_offset = (target_dt - file_start_dt).total_seconds()
        
        # Calculate start and end times for the clip (centered on CPA)
        half_duration = clip_duration / 2.0
        clip_start = max(0, time_offset - half_duration)
        clip_end = time_offset + half_duration
        
        # Get audio duration to validate
        audio_duration = self._get_audio_duration(filepath)
        
        if time_offset < 0 or time_offset > audio_duration:
            print(f"CPA timestamp {cpa_timestamp} outside audio file duration")
            return None
        
        # Adjust clip_end if it exceeds file duration
        clip_end = min(clip_end, audio_duration)
        
        metadata = {
            'cpa_timestamp': cpa_timestamp,
            'cpa_datetime': target_dt,
            'source_file': filepath.name,
            'source_filepath': str(filepath),
            'time_offset_seconds': time_offset,
            'clip_start_seconds': clip_start,
            'clip_end_seconds': clip_end,
            'clip_duration_seconds': clip_end - clip_start
        }
        
        # Extract the actual audio if output path is specified
        if output_path:
            success = self._extract_audio_segment(
                filepath, 
                clip_start, 
                clip_end, 
                output_path
            )
            metadata['output_path'] = output_path if success else None
            metadata['extraction_success'] = success
        
        return metadata
    
    def _extract_audio_segment(
        self, 
        input_path: Path, 
        start_sec: float, 
        end_sec: float, 
        output_path: str
    ) -> bool:
        """
        Extract a segment from a WAV file.
        
        Args:
            input_path: Path to input WAV file
            start_sec: Start time in seconds
            end_sec: End time in seconds
            output_path: Path for output file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with wave.open(str(input_path), 'rb') as wav_in:
                # Get audio parameters
                params = wav_in.getparams()
                framerate = wav_in.getframerate()
                n_channels = wav_in.getnchannels()
                sampwidth = wav_in.getsampwidth()
                
                # Calculate frame positions
                start_frame = int(start_sec * framerate)
                end_frame = int(end_sec * framerate)
                
                # Set position and read frames
                wav_in.setpos(start_frame)
                frames_to_read = end_frame - start_frame
                audio_data = wav_in.readframes(frames_to_read)
                
                # Write output file
                with wave.open(output_path, 'wb') as wav_out:
                    wav_out.setparams(params)
                    wav_out.writeframes(audio_data)
            
            return True
        
        except Exception as e:
            print(f"Error extracting audio segment: {e}")
            return False

    def _extract_audio_segment_multifile(
        self,
        files_to_process,
        start_timestamp: datetime,
        end_timestamp: datetime,
        output_path: str
    ) -> bool:
        """
        Extract an audio segment that may span multiple WAV files.
        Uses the audio_metadata DataFrame to find and extract from relevant files.
        
        Args:
            start_timestamp: Start time as datetime object
            end_timestamp: End time as datetime object
            output_path: Path for output file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Collect audio data from all files
            all_audio_data = []
            output_params = None
            print("***Start extracting audio files***")
            for filepath in files_to_process: 
                print(filepath)
                file_start_timestamp = self._parse_filename_timestamp(filepath)

                # Get the file duration to determine end time
                with wave.open(str(filepath), 'rb') as wav:
                    params = wav.getparams()
                    framerate = wav.getframerate()
                    n_frames = wav.getnframes()
                    file_duration = n_frames / float(framerate)
                    print(framerate, n_frames, file_duration)
                    file_end_timestamp=file_start_timestamp+ timedelta(seconds=file_duration)
                    print(start_timestamp, file_end_timestamp)

                    if output_params is None:
                        output_params = params
                    else:
                        if (params.nchannels != output_params.nchannels or
                            params.sampwidth != output_params.sampwidth or
                            params.framerate != output_params.framerate):
                            print(f"Warning: Audio parameters mismatch in {filepath}")
                
                    if (start_timestamp>=file_start_timestamp) & (start_timestamp<=file_end_timestamp):
                        print("flag1")
                        # this is the starting file, contains only part of the data to be extracted
                        start_frame = (start_timestamp - file_start_timestamp).total_seconds()*float(framerate)
                        end_frame = n_frames
                        print(start_frame, end_frame)
                    elif (end_timestamp>=file_start_timestamp) & (end_timestamp<=file_end_timestamp):
                        print("flag2")
                        # this is the ending file, contains only part of the data to be extracted 
                        start_frame = 0
                        end_frame = (end_timestamp - file_start_timestamp).total_seconds()*float(framerate)
                        print(start_frame, end_frame)
                    elif (end_timestamp < file_start_timestamp) | (start_timestamp > file_end_timestamp):
                        #this is not the file we need because either the target end_timestamp is earlier than the file start time or the target start_timestamp is later than the file end time
                        continue 
                    else:
                        print("flag3")
                        # this is the middle file, extract the entire file
                        start_frame = 0
                        end_frame = n_frames
                        print(start_frame, end_frame)
                    
                    frames_to_read = int(end_frame - start_frame)            
                    print(f"start_frame: {start_frame}; end_frame: {end_frame}; frames_to_read: {frames_to_read}; file_n_frames: {n_frames}")
                        
                    wav.setpos(int(start_frame))
                    audio_chunk = wav.readframes(frames_to_read)
                    all_audio_data.append(audio_chunk)
                        
            # Concatenate all audio data
            combined_audio = b''.join(all_audio_data)
                
            # Write output file
            with wave.open(output_path, 'wb') as wav_out:
                wav_out.setparams(output_params)
                wav_out.writeframes(combined_audio)
                
            duration = (end_timestamp - start_timestamp).total_seconds()
            print(f"✓ Created {duration:.2f}s clip from {len(files_to_process)} file(s)")

            return True
            
        except Exception as e:
            print(f"Error extracting multi-file audio segment: {e}")
            import traceback
            traceback.print_exc()
            return False


    def _extract_audio_segment_multifile_old(
        self,
        start_timestamp: datetime,
        end_timestamp: datetime,
        output_path: str
    ) -> bool:
        """
        Extract an audio segment that may span multiple WAV files.
        Uses the audio_metadata DataFrame to find and extract from relevant files.
        
        Args:
            start_timestamp: Start time as datetime object
            end_timestamp: End time as datetime object
            output_path: Path for output file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.audio_metadata.empty:
                print("Error: No audio files available")
                return False
            
            # Find which files contain the start and end timestamps
            files_to_process = []
            
            for idx, row in self.audio_metadata.iterrows():
                file_start = row['timestamp']
                filepath = row['filepath']
                
                # Get the file duration to determine end time
                with wave.open(str(filepath), 'rb') as wav:
                    framerate = wav.getframerate()
                    n_frames = wav.getnframes()
                    file_duration = n_frames / float(framerate)
                
                file_end = file_start.timestamp() + file_duration
                file_start_sec = file_start.timestamp()
                
                # Check if this file overlaps with our target range
                target_start = start_timestamp.timestamp()
                target_end = end_timestamp.timestamp()
                
                if file_start_sec <= target_end and file_end >= target_start:
                    # Calculate what portion of this file to extract
                    extract_start = max(0, target_start - file_start_sec)
                    extract_end = min(file_duration, target_end - file_start_sec)
                    
                    files_to_process.append({
                        'filepath': filepath,
                        'file_start_time': file_start,
                        'file_duration': file_duration,
                        'extract_start_sec': extract_start,
                        'extract_end_sec': extract_end
                    })
            
            if not files_to_process:
                print(f"Error: No files found containing timestamps {start_timestamp} to {end_timestamp}")
                return False
            
            # Collect audio data from all files
            all_audio_data = []
            output_params = None
            
            print(f"Extracting audio from {len(files_to_process)} file(s):")
            
            for file_info in files_to_process:
                filepath = file_info['filepath']
                start_sec = file_info['extract_start_sec']
                end_sec = file_info['extract_end_sec']
                
                with wave.open(str(filepath), 'rb') as wav_in:
                    params = wav_in.getparams()
                    framerate = wav_in.getframerate()
                    
                    if output_params is None:
                        output_params = params
                    else:
                        if (params.nchannels != output_params.nchannels or
                            params.sampwidth != output_params.sampwidth or
                            params.framerate != output_params.framerate):
                            print(f"Warning: Audio parameters mismatch in {filepath}")
                    
                    start_frame = int(start_sec * framerate)
                    end_frame = int(end_sec * framerate)
                    frames_to_read = end_frame - start_frame
                    
                    wav_in.setpos(start_frame)
                    audio_chunk = wav_in.readframes(frames_to_read)
                    all_audio_data.append(audio_chunk)
                    
                    print(f"  ✓ {end_sec - start_sec:.2f}s from {Path(filepath).name} "
                          f"(offset {start_sec:.2f}s to {end_sec:.2f}s)")
            
            # Concatenate all audio data
            combined_audio = b''.join(all_audio_data)
            
            # Write output file
            with wave.open(output_path, 'wb') as wav_out:
                wav_out.setparams(output_params)
                wav_out.writeframes(combined_audio)
            
            duration = (end_timestamp - start_timestamp).total_seconds()
            print(f"✓ Created {duration:.2f}s clip from {len(files_to_process)} file(s)")
            return True
        
        except Exception as e:
            print(f"Error extracting multi-file audio segment: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def extract_clip_around_cpa_multifile(
        self,
        cpa_timestamp: str,
        clip_duration: float = 30.0,
        output_path: Optional[str] = None
    ) -> Optional[dict]:
        """
        Extract audio clip around a CPA timestamp using multi-file extraction.
        This method can handle clips that span multiple audio files.
        
        Args:
            cpa_timestamp: CPA timestamp string (ISO format)
            clip_duration: Total duration of clip in seconds (default: 30)
            output_path: Optional path to save the extracted clip
            
        Returns:
            Dictionary with extraction metadata or None if extraction failed
        """
        target_dt = self._parse_cpa_timestamp(cpa_timestamp)
        
        # Calculate start and end times for the clip (centered on CPA)
        half_duration = clip_duration / 2.0
        from datetime import timedelta
        clip_start_dt = target_dt - timedelta(seconds=half_duration)
        clip_end_dt = target_dt + timedelta(seconds=half_duration)
        
        metadata = {
            'cpa_timestamp': cpa_timestamp,
            'cpa_datetime': target_dt,
            'clip_start_datetime': clip_start_dt,
            'clip_end_datetime': clip_end_dt,
            'clip_duration_seconds': clip_duration,
            'extraction_method': 'multifile'
        }
        
        # Extract the actual audio if output path is specified
        if output_path:
            success = self._extract_audio_segment_multifile(
                clip_start_dt,
                clip_end_dt,
                output_path
            )
            metadata['output_path'] = output_path if success else None
            metadata['extraction_success'] = success
        
        return metadata

    def extract_clip_around_cpa_mix(
            self,
            cpa_timestamp: str, 
            clip_duration: float = 30.0,
            output_path = None
        ):
            """
            Extract audio clip around a CPA timestamp.
            
            Args:
                cpa_timestamp: CPA timestamp string (ISO format)
                clip_duration: Total duration of clip in seconds (default: 30)
                output_path: Optional path to save the extracted clip
                
            Returns:
                Dictionary with extraction metadata or None if extraction failed
            """
            if type(cpa_timestamp)==str:
                target_dt = self._parse_cpa_timestamp(cpa_timestamp)
            else:
                target_dt= cpa_timestamp
            # Find the audio file containing this timestamp
            result = self._find_audio_file_for_timestamp(target_dt)
            
            if result is None:
                print(f"No audio file found for timestamp {cpa_timestamp}")
                return None
            
            file_idx, file_start_dt = result
            filepath = self.audio_metadata.loc[file_idx, 'filepath']
            
            audio_duration = 10 #<- for orcasound clips, most are around 10s

            if clip_duration < audio_duration:
                # Calculate offset from start of file
                time_offset = (target_dt - file_start_dt).total_seconds()
                
                # Calculate start and end times for the clip (centered on CPA)
                half_duration = clip_duration / 2.0
                clip_start = max(0, time_offset - half_duration)
                clip_end = time_offset + half_duration
                        
                if time_offset < 0 or time_offset > audio_duration:
                    print(f"CPA timestamp {cpa_timestamp} outside audio file duration")
                    return None
                
                # Adjust clip_end if it exceeds file duration
                clip_end = min(clip_end, audio_duration)
                metadata = {
                    'cpa_timestamp': cpa_timestamp,
                    'cpa_datetime': target_dt,
                    'source_file': filepath.name,
                    'source_filepath': str(filepath),
                    'time_offset_seconds': time_offset,
                    'clip_start_seconds': clip_start,
                    'clip_end_seconds': clip_end,
                    'clip_duration_seconds': clip_end - clip_start
                }
                # Extract the actual audio if output path is specified
                if output_path:
                    success = self._extract_audio_segment(
                        filepath, 
                        clip_start, 
                        clip_end, 
                        output_path
                    )
                    metadata['output_path'] = output_path if success else None
                    metadata['extraction_success'] = success

            else:
                tmdelta = timedelta(seconds=(clip_duration/2))
                clip_start = target_dt - tmdelta
                clip_end = target_dt + tmdelta 
                print(clip_start, clip_end)
                starting_clip = self._find_audio_file_for_timestamp(clip_start)
                ending_clip = self._find_audio_file_for_timestamp(clip_end)
                file_idx0, _ = starting_clip
                file_idx1, _ = ending_clip
                filepath = self.audio_metadata.loc[file_idx0:file_idx1+1, 'filepath']   
                for f in filepath:
                    print(f)
                metadata = {
                    'cpa_timestamp': cpa_timestamp,
                    'cpa_datetime': target_dt,
                    'source_file': filepath.tolist(),
                    'source_filepath': filepath,
                    'time_offset_seconds': None,
                    'clip_start_seconds': clip_start,
                    'clip_end_seconds': clip_end,
                    'clip_duration_seconds': clip_end - clip_start
                }    
            # Extract the actual audio if output path is specified
            if output_path:
                success = self._extract_audio_segment_multifile(
                    filepath,
                    clip_start,
                    clip_end,
                    output_path
                )
                metadata['output_path'] = output_path if success else None
                metadata['extraction_success'] = success
            return metadata

    def process_cpa_dataframe(
        self, 
        df: pd.DataFrame, 
        timestamp_column: str = 't_cpa',
        output_folder: Optional[str] = None,
        clip_duration: float = 30.0
    ) -> pd.DataFrame:
        """
        Process all CPA timestamps in a dataframe.
        
        Args:
            df: DataFrame containing CPA timestamps
            timestamp_column: Name of the timestamp column
            output_folder: Optional folder to save extracted clips
            clip_duration: Duration of clips in seconds
            
        Returns:
            DataFrame with extraction results
        """
        results = []
        
        if output_folder:
            output_path = Path(output_folder)
            output_path.mkdir(parents=True, exist_ok=True)
        
        for idx, row in df.iterrows():
            cpa_timestamp = row[timestamp_column]
            
            # Generate output filename if folder specified
            output_file = None
            if output_folder:
                # Create safe filename from timestamp
                safe_ts = cpa_timestamp.replace(':', '-').replace('Z', '')
                output_file = str(output_path / f"cpa_clip_{safe_ts}_{idx}.wav")
            
            # Extract the clip
            result = self.extract_clip_around_cpa_mix(
                cpa_timestamp, 
                clip_duration=clip_duration,
                output_path=output_file
            )
            
            if result:
                result['dataframe_index'] = idx
                results.append(result)
        
        return pd.DataFrame(results)


# Convenience functions for quick usage
def extract_single_cpa_clip(
    cpa_timestamp: str,
    audio_folder: str,
    audio_filenames: List[str],
    output_path: str,
    clip_duration: float = 30.0
) -> Optional[dict]:
    """
    Extract a single CPA audio clip.
    
    Args:
        cpa_timestamp: CPA timestamp (ISO format)
        audio_folder: Path to audio folder
        audio_filenames: List of audio filenames
        output_path: Path for output file
        clip_duration: Duration in seconds
        
    Returns:
        Extraction metadata dictionary
    """
    extractor = AudioCPAExtractor(audio_folder, audio_filenames)
    return extractor.extract_clip_around_cpa(cpa_timestamp, clip_duration, output_path)


def process_cpa_dataframe(
    df: pd.DataFrame,
    audio_folder: str,
    audio_filenames: List[str],
    output_folder: str,
    timestamp_column: str = 't_cpa',
    clip_duration: float = 30.0
) -> pd.DataFrame:
    """
    Process all CPAs in a dataframe and extract audio clips.
    
    Args:
        df: DataFrame with CPA timestamps
        audio_folder: Path to audio folder
        audio_filenames: List of audio filenames
        output_folder: Folder to save clips
        timestamp_column: Name of timestamp column
        clip_duration: Duration in seconds
        
    Returns:
        DataFrame with extraction results
    """
    extractor = AudioCPAExtractor(audio_folder, audio_filenames)
    return extractor.process_cpa_dataframe(
        df, 
        timestamp_column=timestamp_column,
        output_folder=output_folder,
        clip_duration=clip_duration
    )
