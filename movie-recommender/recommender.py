"""
========================================================
 recommender.py
========================================================
This file contains the "brain" of our movie recommendation
website. In very simple words, here is the idea:

  1. We look at all the ratings every user has given to movies.
  2. We build a big table (a "user-item matrix") where:
        - each ROW is a user
        - each COLUMN is a movie
        - each CELL is the rating that user gave that movie
        - if a user never watched a movie, the cell is 0
  3. We then figure out which MOVIES are similar to each other,
     based on how users rated them. If two movies are usually
     rated the same way by the same people, they are "similar".
     (This is called Item-Based Collaborative Filtering.)
  4. When we want recommendations for a user, we look at the
     movies they liked, find similar movies, and suggest the
     ones they have NOT already watched.

This is beginner friendly on purpose - no deep learning, just
simple math (cosine similarity) that is easy to understand.
========================================================
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    """A simple class that loads the data and gives movie recommendations."""

    def __init__(self, ratings_path="data/ratings.csv", movies_path="data/movies.csv"):
        # Step 1: Load the two CSV files into pandas tables (DataFrames)
        self.ratings = pd.read_csv(ratings_path)
        self.movies = pd.read_csv(movies_path)

        # Step 2: Build the user-item matrix
        # rows = userId, columns = movieId, values = rating
        # fillna(0) means "if a user never rated a movie, put 0"
        self.user_item_matrix = self.ratings.pivot(
            index="userId",
            columns="movieId",
            values="rating"
        ).fillna(0)

        # Step 3: Build the movie-similarity table
        # This tells us: "how similar is movie A to movie B?"
        self._build_movie_similarity()

    def _build_movie_similarity(self):
        """
        This function calculates how similar every movie is to every
        other movie, based on how users rated them.

        cosine_similarity gives a score between 0 and 1:
           1.0 = the two movies are rated in an almost identical way
           0.0 = the two movies have nothing in common
        """
        # We need movies as ROWS and users as COLUMNS for this step,
        # so we transpose (flip) the user-item matrix.
        movie_user_matrix = self.user_item_matrix.T

        similarity_scores = cosine_similarity(movie_user_matrix)

        # Put the result into a nice labeled table
        self.movie_similarity_df = pd.DataFrame(
            similarity_scores,
            index=movie_user_matrix.index,
            columns=movie_user_matrix.index
        )

    def get_all_users(self):
        """Returns a list of all user IDs available in the dataset."""
        return sorted(self.user_item_matrix.index.tolist())

    def get_movie_title(self, movie_id):
        """Given a movie ID, returns its title."""
        row = self.movies[self.movies["movieId"] == movie_id]
        if row.empty:
            return "Unknown Movie"
        return row.iloc[0]["title"]

    def get_watched_movies(self, user_id):
        """
        Returns a small table of movies this user has already
        rated, sorted from highest rating to lowest.
        """
        if user_id not in self.user_item_matrix.index:
            return pd.DataFrame(columns=["movieId", "title", "rating"])

        user_ratings = self.ratings[self.ratings["userId"] == user_id]
        merged = user_ratings.merge(self.movies, on="movieId")
        merged = merged.sort_values(by="rating", ascending=False)
        return merged[["movieId", "title", "rating"]]

    def recommend_movies(self, user_id, top_n=5):
        """
        This is the main function you will use.

        Steps it follows:
          1. Find all movies this user has already rated.
          2. For every movie they rated, find similar movies.
          3. Give each similar movie a "score" based on:
                (how much the user liked the original movie)
                     x
                (how similar the new movie is to it)
          4. Add up scores for movies that show up more than once.
          5. Remove movies the user has ALREADY watched.
          6. Sort by score and return the top results.
        """
        if user_id not in self.user_item_matrix.index:
            return []

        # Step 1: movies this user has rated, and how much they rated them
        user_ratings = self.user_item_matrix.loc[user_id]
        watched_movies = user_ratings[user_ratings > 0]

        # This dictionary will store: {movieId: total_score}
        scores = {}

        # Step 2 and 3: go through every movie the user watched
        for movie_id, rating_given in watched_movies.items():

            # Skip if this movie has no similarity data (edge case safety)
            if movie_id not in self.movie_similarity_df.columns:
                continue

            # Get similarity scores between this movie and every other movie
            similar_movies = self.movie_similarity_df[movie_id]

            for other_movie_id, similarity_score in similar_movies.items():

                # Don't recommend the same movie back to the user
                if other_movie_id == movie_id:
                    continue

                # Don't recommend movies the user already watched
                if other_movie_id in watched_movies.index:
                    continue

                # The more the user liked movie_id AND the more similar
                # other_movie_id is to it, the higher the score
                weighted_score = similarity_score * rating_given

                if other_movie_id not in scores:
                    scores[other_movie_id] = 0
                scores[other_movie_id] += weighted_score

        # Step 4 & 5 already done above. Step 6: sort by score, highest first
        sorted_movie_ids = sorted(scores, key=scores.get, reverse=True)
        top_movie_ids = sorted_movie_ids[:top_n]

        # Turn movie IDs into nice readable results
        recommendations = []
        for movie_id in top_movie_ids:
            title = self.get_movie_title(movie_id)
            recommendations.append({
                "movieId": int(movie_id),
                "title": title,
                "score": round(scores[movie_id], 2)
            })

        return recommendations


# This part only runs if you execute this file directly
# (python recommender.py) - useful for quick testing.
if __name__ == "__main__":
    engine = MovieRecommender()

    test_user_id = 5
    print(f"\nMovies user {test_user_id} already watched and liked:")
    print(engine.get_watched_movies(test_user_id))

    print(f"\nTop recommendations for user {test_user_id}:")
    for movie in engine.recommend_movies(test_user_id, top_n=5):
        print(f"  {movie['title']}  (score: {movie['score']})")
