"""
search_index.py
-----------------
This replaces what we were doing before with sentence-transformers + FAISS.
Here's why: that combo needed PyTorch, which alone uses way more than the
512MB of RAM Render gives you on the free tier - the app kept crashing
with "Out of memory" errors. So instead, we're using something much
lighter: TF-IDF (Term Frequency-Inverse Document Frequency).

In plain English, what TF-IDF does:
  - It looks at which words appear in each chunk of news text.
  - Words that are common across EVERY chunk (like "the", "news", "today")
    get treated as less important.
  - Words that are distinctive to a specific chunk (like "earthquake" or
    "election") get treated as more important.
  - When you ask a question, it does the same thing to your question, and
    finds whichever chunks share the most "important" words with it.

It's not as smart as a full neural embedding model at understanding
meaning (it matches on actual words more than deep meaning), but for a
news chatbot where people mostly ask about people/places/topics by name,
it works surprisingly well - and it uses a tiny fraction of the memory,
starts up almost instantly, and needs zero large downloads.

Everything gets saved into search_index/ as a couple of small files.
"""

import os
import re
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = "data"
INDEX_DIR = "search_index"
INDEX_FILE = os.path.join(INDEX_DIR, "index.pkl")

CHUNK_SIZE = 500       # roughly how many characters per chunk
CHUNK_OVERLAP = 80     # slight overlap so we don't cut a sentence in half


def load_text_files():
    """Reads every .txt file in data/ and returns their combined text."""
    combined_text = ""
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(DATA_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                combined_text += f.read() + "\n"
    return combined_text


def split_into_chunks(text):
    """Splits text into small overlapping chunks. We first try splitting
    on article boundaries (the '---' separator our news file uses), then
    make sure none of those pieces are too huge by breaking big ones down
    further by character count."""

    # First split on article separators, so each chunk is ideally one
    # whole news article rather than a random slice of text.
    rough_pieces = re.split(r"\n-{3,}\n", text)

    chunks = []
    for piece in rough_pieces:
        piece = piece.strip()
        if not piece:
            continue

        if len(piece) <= CHUNK_SIZE:
            chunks.append(piece)
        else:
            # This piece is too long (rare, but possible) - break it into
            # smaller overlapping windows.
            start = 0
            while start < len(piece):
                end = start + CHUNK_SIZE
                chunks.append(piece[start:end])
                start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def build_search_index():
    """Builds the TF-IDF search index from whatever's in data/ and saves
    it to disk so the app can load it quickly."""
    print("Building the search index from whatever's in data/ ...")

    text = load_text_files()
    if not text.strip():
        raise ValueError(
            "Couldn't find any text inside data/. Something's off - there "
            "should at least be the offline fallback file in there."
        )

    print("Splitting the news into small chunks...")
    chunks = split_into_chunks(text)
    print(f"   Got {len(chunks)} chunks total.")

    print("Building the TF-IDF index (this is quick, no big downloads needed)...")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    chunk_vectors = vectorizer.fit_transform(chunks)

    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(
            {
                "vectorizer": vectorizer,
                "chunk_vectors": chunk_vectors,
                "chunks": chunks,
            },
            f,
        )

    print(f"Done! Search index saved to '{INDEX_FILE}'.")


if __name__ == "__main__":
    build_search_index()
