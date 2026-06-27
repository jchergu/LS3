# Preprocessing

Dataset is loaded from data/raw/song_lyrics.csv, not included in repo for copyright and size.
Downloading steps:
1. [https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information](Download from Kaggle)
2. Create data/raw/ folder in root of project
3. Extract downloaded song_lyrics.csv in data/raw/

## How to run

Then you can run preprocessing (make sure you are in the virtual environment):
```bash
python main.py preprocessing
```

After preprocessing, the following structure is expected:
```
data/
    raw/  
        song_lyrics.csv        # original dataset         
    processed/
        lyrics_clean.csv       # cleaned dataset
        lyrics_en.csv          # english-only lyrics
        lyrics_normalized.csv  # normalized text
        lyrics_no_dup.csv      # no duplicates
        lyrics_final.csv       # final dataset used for encoding
```

Each step of the preprocessing writes in a new file, to avoid re-running the entire pipeline if something goes wrong. In addition, each step is idempotent.