# DatasetMaker

Segment a long audio file into short audios suitable for model training.

## Installation

```
# First, set up PyTorch according to instructions at https://pytorch.org/get-started/locally
# If GPU support is not needed, simply run "pip install torch"
pip install -r requirements.txt
```

## Running the program

Split an audio file. It can accept any audio format supported by librosa.
```
python process_file.py myfile.wav
```

This makes a folder called "output" and separate "speaker" folders with sorted audios.

### Usage notes

1. If the output folder doesn't exist, it will be created.
2. If the output folder exists and is not empty, the program will exit without doing anything.

###  Options

Specify the embed tolerance for matching a wav to a previously identified speaker. (Default: 0.5) Larger values cluster wavs into fewer speaker folders, at the expense of increased classification error. 
```
python process_file.py myfile.wav --distance 0.25
```

Write output to a different directory. (Default: "output")
```
python process_file.py myfile.wav --out_dir mydir
```

Require at least 20 audios to define a separate speaker folder. (Default: 5 files) Files not matched to a speaker will appear in the "uncategorized" folder.
```
python process_file.py myfile.wav --min_count 20
```

Resample audio to a different sample rate, instead of following the input file.
```
python process_file.py myfile.wav --sr 16000
```

Discard all audio segments shorter than 3 seconds. (Default: 2.0 seconds)
```
python process_file.py myfile.wav --duration 3.0
```

Output VCTK folders and filenames, starting with speaker p400.
```
python process_file.py myfile.wav --vctk_index 400
```

## Acknowledgements

This repo uses the voice encoder and checkpoints from:
https://github.com/liusongxiang/ppg-vc

Which is based on the speaker encoder from:
https://github.com/CorentinJ/Real-Time-Voice-Cloning
