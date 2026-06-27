---
title: LS3 Lyrics Similarity Search
emoji: 🎵
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# LS3: Lyrics Similarity Search System

Search songs by meaning, not keywords.    

LS3 can receive a string-query from the user and perform a semantic-search of lyrics, thanks to Large Language Model embeddings and vector databases.

Examples of queries and responses from the system:

```
"Happy songs about love"

1. "Can't Help Falling in Love" — Elvis Presley  
2. "Happy" — Pharrell Williams  
3. "I Wanna Dance with Somebody (Who Loves Me)" — Whitney Houston 
```

```
"upbeat workout tracks"

1. "TNT" - AC/DC
2. "Party Rock Anthem" - LMFAO
3. "Survival" - Eminem
```

## How to run

Clone repo, create environment and install dependencies:
```bash
git clone git@github.com:jchergu/LS3.git # or use HTTPS: git clone https://github.com/jchergu/LS3.git
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

Install the dataset:
```bash
curl -L -o ~/Downloads/genius-song-lyrics-with-language-information.zip\
  https://www.kaggle.com/api/v1/datasets/download/carlosgdcj/genius-song-lyrics-with-language-information
```

Run the preprocessing:
```bash
python -m preprocessing.run
``` 

Run the encoding:
```bash
python -m encoding.run
```

Run the main:
```bash
python -m main
``` 

You can also test the system directly:
```bash
python -m test
```

Each module is tested individually.

#### In alternative  
You can run the backend server and query it via HTTP:
Start the uvicorn backend server:
```bash
uvicorn backend.app:app --reload
```

Open the frontend/index.html in your browser:
```bash
frontend/index.html 
```


## Architecture

### Preprocessing
- Download dataset
- Filter to english-only lyrics (for now, to save memory)
- Select columns `{artist_name, song_name, lyrics}`
- Remove duplicates
- Clean lyric annotations
- Normalize text
- Handle long lyrics

### Encoding
The default Language Model is `sentence-transformers/all-MiniLM-L6-v2`, but you can change it in `encoding/config.py`

The encoding process is as follows:
- For each song, generate embeddings using a Language Model
- Store encodings

The encoding/ folder is divided into 4 responsibilities:
1. Configuration
2. Embedding logic
3. Orchestration
4. Storage

Embeddings are stored in the `data/embeddings` folder

### Backend

The default vector database is FAISS, but you can change it in `backend/index.py`.

Structure:
- Application layer: API
- Data layer: vector index + metadata
- Logic: similarity search

The backend is responsible for:
- Load embeddings into memory
- Embed user query
- Find nearest vectors
- Return results in JSON format

Backend is fully testable with no need of real (big) data.

#### Comparing NumPy vs FAISS
*TL;DR*: FAISS is 10x faster. See the comparison folder for details.
