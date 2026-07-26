"""
========================================================
 download_data.py
========================================================
What this file does (in very simple words):

1. It goes to the internet and downloads the REAL MovieLens 100K
   dataset (100,000 movie ratings made by real people).
2. That dataset comes as a ZIP file, so we unzip it.
3. The original files are named funny things like "u.data" and
   "u.item", so we convert them into two clean, easy files:
       - data/ratings.csv   (who rated what, and how much)
       - data/movies.csv    (movie id, title, genres)

You only need to run this file ONCE.
After that, ratings.csv and movies.csv will already be sitting
inside the "data" folder, ready to be used by the website.

How to run it:
    python download_data.py
========================================================
"""

import os
import zipfile
import urllib.request

# This is the official link where the MovieLens 100K dataset lives
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

# Where we will temporarily save the zip file
ZIP_FILE_NAME = "ml-100k.zip"

# The folder where the final clean CSV files should go
DATA_FOLDER = "data"


def download_dataset():
    """Step 1: Download the zip file from the internet."""
    print("Downloading MovieLens 100K dataset... please wait.")
    urllib.request.urlretrieve(DATASET_URL, ZIP_FILE_NAME)
    print("Download finished!")


def extract_dataset():
    """Step 2: Unzip the downloaded file."""
    print("Unzipping the dataset...")
    with zipfile.ZipFile(ZIP_FILE_NAME, "r") as zip_ref:
        zip_ref.extractall(".")
    print("Unzipping finished!")


def convert_to_csv():
    """Step 3: Turn the raw MovieLens files into simple CSV files."""

    # Make sure the "data" folder exists
    os.makedirs(DATA_FOLDER, exist_ok=True)

    # ---------- Convert ratings ----------
    # The original file "u.data" looks like this (columns separated by TAB):
    #   userId    movieId    rating    timestamp
    print("Creating ratings.csv ...")
    ratings_path_in = os.path.join("ml-100k", "u.data")
    ratings_path_out = os.path.join(DATA_FOLDER, "ratings.csv")

    with open(ratings_path_in, "r", encoding="latin-1") as infile, \
         open(ratings_path_out, "w", encoding="utf-8") as outfile:

        # Write the header row first
        outfile.write("userId,movieId,rating,timestamp\n")

        for line in infile:
            # Each line is: userId \t movieId \t rating \t timestamp
            user_id, movie_id, rating, timestamp = line.strip().split("\t")
            outfile.write(f"{user_id},{movie_id},{rating},{timestamp}\n")

    # ---------- Convert movies ----------
    # The original file "u.item" looks like this (columns separated by |):
    #   movieId | title | release_date | ... | genre flags (0 or 1) ...
    print("Creating movies.csv ...")
    movies_path_in = os.path.join("ml-100k", "u.item")
    movies_path_out = os.path.join(DATA_FOLDER, "movies.csv")

    # These are the 19 genre names, in the same order MovieLens uses them
    genre_names = [
        "unknown", "Action", "Adventure", "Animation", "Children",
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
        "Sci-Fi", "Thriller", "War", "Western"
    ]

    with open(movies_path_in, "r", encoding="latin-1") as infile, \
         open(movies_path_out, "w", encoding="utf-8") as outfile:

        outfile.write("movieId,title,genres\n")

        for line in infile:
            parts = line.strip().split("|")
            movie_id = parts[0]
            title = parts[1]

            # The last 19 values in the line are 0/1 flags for each genre
            genre_flags = parts[-19:]
            genres_for_this_movie = [
                genre_names[i] for i, flag in enumerate(genre_flags) if flag == "1"
            ]
            genres_text = "|".join(genres_for_this_movie) if genres_for_this_movie else "(no genres)"

            # Titles sometimes contain commas, so we wrap the title in quotes
            outfile.write(f'{movie_id},"{title}",{genres_text}\n')

    print("All done! Check the 'data' folder for ratings.csv and movies.csv")


def cleanup():
    """Step 4: Remove the temporary zip file and extracted folder (optional, keeps things tidy)."""
    if os.path.exists(ZIP_FILE_NAME):
        os.remove(ZIP_FILE_NAME)
    print("Cleaned up temporary files.")


if __name__ == "__main__":
    download_dataset()
    extract_dataset()
    convert_to_csv()
    cleanup()
    print("\nSuccess! Your real MovieLens 100K data is ready in the 'data' folder.")
