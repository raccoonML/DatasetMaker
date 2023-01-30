import argparse
import datetime
import librosa
import numpy as np
import os
import soundfile as sf
import struct
import time
import torch
import webrtcvad
from pathlib import Path
from scipy import signal
from scipy.ndimage.morphology import binary_dilation
from speaker_encoder.voice_encoder import SpeakerEncoder
from tqdm import tqdm

vad_window_length = 30                     # In milliseconds (allowed values = 10, 20, or 30)
vad_moving_average_width = 8               # Number of frames to use when smoothing
vad_max_silence_length = 6                 # Max number of consecutive silent frames

"""
Process an audio file with these steps:
    1. Identify speech and non-speech areas
    2. For speech areas, use the speaker encoder to ID the voice.
        a. If it matches an existing speaker, write the audio segment
           to that speaker's folder.
        b. If it doesn't match, then write to a new folder.
"""

def voice_activity_detection(wav, vad_sample_rate=16000):
    int16_max = (2 ** 15) - 1

    # Compute the voice detection window size
    samples_per_window = (vad_window_length * vad_sample_rate) // 1000

    # Zero-pad the end of the audio to have a multiple of the window size
    if len(wav) % samples_per_window > 0:
        pad_len = samples_per_window - (len(wav) % samples_per_window)
        wav = np.pad(wav, ((0, pad_len)))
    else:
        pad_len = 0

    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))

    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=vad_sample_rate))
    voice_flags = np.array(voice_flags)

    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width

    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)

    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)

    # Remove padding from audio mask
    audio_mask = audio_mask[:-pad_len]
    return audio_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Splits a long audio file into segments, sorted by speaker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_file", type=Path, help=\
        "Path to the audio file.")
    parser.add_argument("-d", "--distance", type=float, default=0.5, help=\
        "Max distance for a segment to be classified as an existing speaker. Must be positive. "
        "Higher values cause dissimilar segments to be recognized as the same speaker. ")
    parser.add_argument("--duration", type=float, default=2.0, help=\
        "Min duration for audio segments, anything shorter than this is discarded.")
    parser.add_argument("-m", "--min_count", type=int, default=5, help=\
        "Minimum number of wavs needed for a speaker to get an output folder. "
        "Speakers with fewer than this number are collected in an 'uncategorized' folder.")
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
        "Path to the output directory. Must not already exist. Defaults to 'output'")
    parser.add_argument("--sr", type=int, default=None, help=\
        "Output audio sample rate. If not specified, SR of input file is used.")
    parser.add_argument("--vctk_index", type=int, default=None, help=\
        "If specified, output files follow VCTK file format starting at given index. "
        "For example, setting this to 400 will start speaker at p400.")
    args = parser.parse_args()

    # Process the arguments
    if not hasattr(args, "out_dir"):
        args.out_dir = Path("output")

    # Create directories
    try:
        args.out_dir.mkdir(exist_ok=False, parents=True)
    except FileExistsError:
        # If the specified directory is not empty, print an error message and exit.
        if len(os.listdir(args.out_dir)) > 0:
            print(f"The specified output directory: '{args.out_dir}' already exists and is not empty.\n"
              "Please specify a different directory.")
            exit(1)
    except Exception as e:
        raise e

    # Load audio
    if args.sr is None:
        # Determine sample rate
        sr = librosa.get_samplerate(args.input_file)
        print(f"Using sample rate of input file: {sr} Hz")
    else:
        sr = args.sr
        print(f"Audio will be resampled to user-specified rate: {sr} Hz")

    print("Loading file...")
    start_time = time.time()
    wav, _ = librosa.load(args.input_file, sr=sr)
    input_audio_duration = len(wav)/sr
    print(f"Done. ({time.time()-start_time:.2f} sec)")

    # Segment to speech and non-speech areas
    if sr >= 28000:
        vad_sample_rate = 32000
    else:
        vad_sample_rate = 16000

    print("Performing voice activity detection...")
    speech_mask = voice_activity_detection(wav, vad_sample_rate=vad_sample_rate)

    # Extract speech areas
    padded_speech_mask = np.concatenate(([False], speech_mask, [False]))
    idx = np.flatnonzero(padded_speech_mask[1:] != padded_speech_mask[:-1])
    wavs_speech = [wav[idx[i]:idx[i+1]] for i in range(0, len(idx), 2)]

    print("Loading speaker encoder...")
    smodel = SpeakerEncoder('speaker_encoder/pretrained.pt')

    print("Speaker classification...")
    speakers = []
    n = 0
    n_skipped = 0
    duration_sorted = 0
    duration_unsorted = 0
    duration_skipped = 0
    for wav_speech in tqdm(wavs_speech, total=len(wavs_speech), unit="wavs"):
        n = n + 1

        # Don't process segments that are less than min duration
        if (len(wav_speech) / sr) < args.duration:
            n_skipped = n_skipped + 1
            duration_skipped = duration_skipped + (len(wav_speech) / sr)
            continue

        # Speaker ID
        # Resample for speaker encoder, since it is trained on 16khz audio
        if sr != 16000:
            wav_speech_16khz = librosa.resample(wav_speech, orig_sr=sr, target_sr=16000)
        else:
            wav_speech_16khz = wav_speech

        # Get the speaker embed
        emb = smodel.embed_utterance(wav_speech_16khz)

        # Look for a matching speaker
        idx = None
        # Iterate over existing speakers
        for i in range(len(speakers)):
            # Difference vector
            diff = emb - speakers[i][1]
            # L2 norm of distance
            if np.sum(np.power(diff, 2)) < args.distance:
                idx = i
                break

        if idx is None:
            # If no speaker matches, then make a new one
            #print(f"Added speaker {len(speakers)} at wav segment {n}.")
            speakers.append([[wav_speech] , emb, [emb]])
        else:
            # Append wav to speaker
            speakers[idx][0].append(wav_speech)
            speakers[idx][2].append(emb)

    # Sort speaker list by number of wavs, in descending order
    speakers.sort(key = lambda l: len(l[0]), reverse=True)

    # Write wavs to file
    n_speakers = 0
    n_unsorted_speakers = 0
    n_unsorted_wavs = 0
    for speaker in speakers:
        if len(speaker[0]) > args.min_count:
            n_speakers = n_speakers + 1
            if args.vctk_index is not None:
                # Use VCTK names if specified
                args.out_dir.joinpath(f"p{args.vctk_index+n_speakers-1}").mkdir(exist_ok=True, parents=True)
            else:
                # Otherwise, use generic speaker names
                args.out_dir.joinpath(f"speaker{n_speakers}").mkdir(exist_ok=True, parents=True)
            n_wavs = 0
            print(f"Writing speaker {n_speakers} to file...")
            for wav in tqdm(speaker[0], total=len(speaker[0]), unit="wavs"):
                n_wavs = n_wavs + 1
                duration_sorted = duration_sorted + (len(wav) / sr)
                if args.vctk_index is not None:
                    # Write to VCTK filenames
                    sf.write(args.out_dir.joinpath(f"p{args.vctk_index+n_speakers-1}/p{args.vctk_index+n_speakers-1}_{n_wavs:04d}.wav"), wav, sr)
                else:
                    # Write to generic filenames
                    sf.write(args.out_dir.joinpath(f"speaker{n_speakers}/speaker{n_speakers}_{n_wavs:04d}.wav"), wav, sr)
        else:
            n_unsorted_speakers = n_unsorted_speakers + 1
            # Write uncategorized wavs to a single folder
            args.out_dir.joinpath("uncategorized").mkdir(exist_ok=True, parents=True)
            for wav in speaker[0]:
                n_unsorted_wavs = n_unsorted_wavs + 1
                duration_unsorted = duration_unsorted + (len(wav) / sr)
                sf.write(args.out_dir.joinpath(f"uncategorized/unsorted{n_unsorted_speakers}_{n_unsorted_wavs:04d}.wav"), wav, sr)

    # Print stats
    silence_duration = input_audio_duration-(duration_sorted+duration_unsorted+duration_skipped)
    print(f"\nInput audio duration: {input_audio_duration/60:.1f} minutes")
    print(f"Silence length: {silence_duration/60:.1f} minutes ({silence_duration*100/input_audio_duration:.1f}%)")
    print(f"Segmented speech duration: {duration_sorted/60:.1f} minutes sorted ({duration_sorted*100/input_audio_duration:.1f}%), {duration_unsorted/60:.1f} minutes unsorted ({duration_unsorted*100/input_audio_duration:.1f}%)")
    print(f"Duration of audios too short to be processed (less than {args.duration} sec): {duration_skipped/60:.1f} minutes ({duration_skipped*100/input_audio_duration:.1f}%)")
