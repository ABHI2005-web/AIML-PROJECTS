"""
========================================================
 app.py
========================================================
This is the main website file. It uses Flask (a simple
Python web framework) to:

  1. Show a home page where you pick a user ID from a dropdown.
  2. When you click "Get Recommendations", it shows:
        - the movies that user already watched and liked
        - new movies we think they will also like

This file does NOT do any complicated math itself - all the
"thinking" happens inside recommender.py. This file is only
in charge of the WEBSITE part (routes, pages, forms).
========================================================
"""

from flask import Flask, render_template, request
from recommender import MovieRecommender

# Create the Flask application
app = Flask(__name__)

# Load the recommendation engine ONE TIME when the server starts.
# (Loading it once is much faster than loading it on every click.)
engine = MovieRecommender()


@app.route("/", methods=["GET"])
def home():
    """
    This is the home page.
    It shows a dropdown list of all users so you can pick one.
    """
    all_users = engine.get_all_users()
    return render_template("index.html", users=all_users)


@app.route("/recommend", methods=["POST"])
def recommend():
    """
    This page runs when you click the "Get Recommendations" button.
    It reads the user ID you picked, and shows recommendations for them.
    """
    # Get the user ID that was submitted from the dropdown form
    user_id = int(request.form.get("user_id"))

    # How many recommendations to show (default is 5)
    top_n = 5

    # Ask the recommendation engine for results
    watched_movies = engine.get_watched_movies(user_id)
    recommended_movies = engine.recommend_movies(user_id, top_n=top_n)

    return render_template(
        "index.html",
        users=engine.get_all_users(),
        selected_user=user_id,
        watched_movies=watched_movies.to_dict("records"),
        recommendations=recommended_movies
    )


# This makes sure the server only runs when you execute this file directly
# (python app.py) - it will NOT run again if this file is imported elsewhere.
if __name__ == "__main__":
    # debug=True shows helpful error messages while you are building the app.
    # On Render, gunicorn will run the app instead of this line.
    app.run(debug=True, host="0.0.0.0", port=5000)
