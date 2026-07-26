# 🎬 Movie Recommendation System

A beginner-friendly movie recommendation web app built with **Flask** and
**Item-Based Collaborative Filtering**, using the **MovieLens 100K** dataset
format (`userId, movieId, rating, timestamp`).

Pick a user ID → see the movies they already watched → get new movie
recommendations based on similar movies other users also liked.

---

## How it works (simple explanation)

1. We build a big table: rows = users, columns = movies, values = ratings.
2. We compare every movie to every other movie, using **cosine similarity**,
   to see which movies get similar ratings from the same people.
3. For a chosen user, we look at movies they rated highly, find similar
   movies, and recommend the ones they haven't watched yet.

All of this logic lives in `recommender.py`, with comments explaining every
step in plain English.

---

## Project structure

```
movie-recommendation-system/
│
├── app.py                # Flask website (routes/pages)
├── recommender.py        # The recommendation "brain" (all the math)
├── download_data.py      # Downloads the REAL MovieLens 100K dataset
│
├── data/
│   ├── ratings.csv        # userId, movieId, rating, timestamp
│   └── movies.csv         # movieId, title, genres
│
├── templates/
│   └── index.html         # The webpage
│
├── static/
│   └── style.css           # Page styling
│
├── requirements.txt        # Python packages needed
├── Procfile                 # Tells Render how to start the app
├── runtime.txt               # Pins the Python version
└── README.md
```

---

## About the dataset

**The dataset is NOT included in this GitHub repository** (per course
policy on not uploading datasets to version control).

This project uses the official **MovieLens 100K dataset**
(100,000 ratings from 943 users on 1,682 movies):
🔗 https://grouplens.org/datasets/movielens/100k/

The `data/` folder is empty except for a `.gitkeep` placeholder. To get the
actual `ratings.csv` and `movies.csv` files, run:

```bash
python download_data.py
```

This script downloads the dataset directly from GroupLens's official
server and converts it into `data/ratings.csv` and `data/movies.csv`,
in the exact format the app expects. It needs to be run once — locally
before you test, and it also runs automatically during deployment (see
the Render section below).


---

## Running it on your computer

1. **Install Python 3.11** (recommended, matches `runtime.txt`)

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Get the dataset (required — it's not in the repo):**
   ```bash
   python download_data.py
   ```

5. **Run the app:**
   ```bash
   python app.py
   ```

6. Open your browser at: `http://127.0.0.1:5000`

---

## Testing the recommendation engine alone (no website)

You can test the "brain" directly without running the website:
```bash
python recommender.py
```
This will print out recommendations for a sample user (user ID 5) in the
terminal.

---

## Deploying to Render

Since the dataset isn't in the repo, Render needs to download it as part of
the build, before the app starts.

1. Push this whole folder to a **GitHub repository**.
2. Go to [render.com](https://render.com) → **New Web Service** → connect
   your GitHub repo.
3. Set these settings:
   - **Root Directory:** leave blank (or set to the correct subfolder if
     your repo has a nested structure — check your GitHub repo's file list
     to confirm where `app.py` actually sits)
   - **Build Command:**
     ```
     pip install -r requirements.txt && python download_data.py
     ```
     (this installs the packages, then downloads the real dataset so it's
     ready before the server starts)
   - **Start Command:** `gunicorn app:app`
   - **Python Version:** already pinned via `runtime.txt` (3.11.9) — add it
     as an environment variable too (`PYTHON_VERSION` = `3.11.9`) if the
     build doesn't pick it up automatically.
4. Click **Deploy**. Once live, Render will give you a public URL.

**Note:** the dataset download adds a little time to each build/deploy
(a few seconds, it's only ~5MB), but this only happens on deploy — not on
every visitor request.

---

## Notes for submission

- The recommendation approach used is **Item-Based Collaborative
  Filtering** with **cosine similarity** — a well known, simple, and
  explainable technique (not a black-box deep learning model), which makes
  it easy to explain.
- The dataset itself is intentionally **not** committed to GitHub. It's
  fetched fresh via `download_data.py` from the official source:
  https://grouplens.org/datasets/movielens/100k/ — the real, unmodified
  dataset (100,000 ratings, 943 users, 1,682 movies), just reformatted from
  the original tab/pipe-separated files into clean CSVs.
