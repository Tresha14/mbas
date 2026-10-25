# Public datasets (raw data only, never model weights)

Each script downloads (or takes a manually downloaded archive), unpacks and converts one dataset into our folder
format. All models are still trained from random initialisation; these datasets are for pre-training, and testing is
always done on our own recordings.

## Folder format

```
data/raw/public/<name>/downloads/      archives as downloaded
data/raw/public/<name>/extracted/      unpacked
data/processed/public/<name>/
    manifest.parquet                   one row per item: item_id, subject_id, split_official, path, + labels
    info.json                          title, licence, source, counts, hours, status
    ...                                converted media (16 kHz WAV clips, resized images, .npz eye crops)
```

`subject_id` is the person (or the closest grouping available), so `python -m tools.make_splits --source public:<name>`
can split any of them by person. Paths are relative to the repo root.

## Datasets

| Script | Dataset | Access | Size | Converted to |
| --- | --- | --- | --- | --- |
| `wider_face.py` | WIDER FACE train + val | automatic | 1.8 GB | image paths + boxes (JSON) |
| `mpiigaze.py` | MPIIGaze (normalised) | automatic | 2.1 GB | per-person `.npz`: 36x60 eye crops, gaze + head pitch/yaw |
| `columbia_gaze.py` | Columbia Gaze | request form | 1.7 GB | 1280 px JPEGs, head pose, gaze V/H, looking-at-camera |
| `daisee.py` | DAiSEE | request form | 15 GB | clip manifest, 4 affect levels + our 3 engagement classes |
| `librispeech.py` | LibriSpeech train-clean-100, dev-clean, test-clean | automatic | 7 GB | utterance manifest (FLAC), `text_corpus.txt` |
| `common_voice.py` | Common Voice EN, Indian-accent subset | free account | ~80 GB archive | 16 kHz WAVs of the subset |
| `sep28k.py` | SEP-28k | automatic (podcast URLs) | ~3 GB of episodes, deleted after cutting | 3 s 16 kHz clips + disfluency labels |
| `mit_interview.py` | MIT Interview Dataset | request authors | ~10 GB | per-interview ratings, transcript, media paths |
| `ami.py` | AMI Meeting Corpus | automatic | 23 MB annotations + ~35 GB audio | turns, per-speaker headset WAVs, talk shares |

```powershell
.\tasks.ps1 datasets                                     # all automatic ones (resumable)
.\.venv\Scripts\python -m tools.datasets.librispeech --subsets dev-clean
.\.venv\Scripts\python -m tools.datasets.daisee --archive D:\Downloads\DAiSEE.zip
```

Downloads resume after an interruption, and finished steps are skipped on a re-run. `tools/dataset_stats.py` lists
each dataset's status in `docs/dataset_card.md`.

## Licences

All of these are research or non-commercial licences. Use them only for this academic project, cite them in the
report, and do not redistribute the raw data.
