"""
fetch_news.py
--------------
Plain English version of what this file does:

We go and grab today's top news headlines from GNews (a free news API)
across a bunch of categories - world news, business, tech, sports, health,
science, entertainment, general stuff. For each category we pull a handful
of articles (title + short description + a bit of the article body).

Then we dump ALL of that into one text file (data/news_articles.txt) so our
existing RAG pipeline (build_vector_store.py) can chunk it, embed it, and
let the chatbot search through it - exact same way it searches through any
other document.

We also save a "fetched_at" timestamp into data/news_cache.json so the app
knows how old the news is, and won't waste API calls re-fetching every
single time someone opens the site (the free GNews tier only gives us
100 requests/day, so we've gotta be a little careful with it).

IMPORTANT AND HONEST NOTE: this pulls TOP HEADLINES per category, NOT
literally every news story happening in the world. No free API (or paid
one, honestly) can give you 100% of world news in real time. This gives
you a solid, real, current snapshot across major categories - which is
genuinely useful, just don't expect it to know about every tiny local
story from every corner of the planet.
"""

import os
import json
import time
import threading
import requests

DATA_DIR = "data"
NEWS_TEXT_FILE = os.path.join(DATA_DIR, "news_articles.txt")
NEWS_CACHE_FILE = os.path.join(DATA_DIR, "news_cache.json")

GNEWS_API_KEY = os.environ.get("GNEWS_API_KEY", "")
GNEWS_URL = "https://gnews.io/api/v4/top-headlines"

# GNews supports these built-in categories out of the box.
CATEGORIES = [
    "general",
    "world",
    "business",
    "technology",
    "entertainment",
    "sports",
    "science",
    "health",
    "nation",
]

# How many articles to grab per category. Keep this smallish since we're
# on a free tier - 9 categories x 1 request each = 9 requests used per
# refresh, out of our 100/day budget.
ARTICLES_PER_CATEGORY = 6

# We won't bother re-fetching if our cached news is younger than this,
# so we don't burn through the free daily quota every time someone
# restarts the server or clicks refresh a bunch of times.
CACHE_FRESH_SECONDS = 3 * 60 * 60  # 3 hours

# HARD ceiling on how long we'll wait for ONE category's request, no
# matter what. This matters because requests' own "timeout" only limits
# time BETWEEN chunks of data arriving - a connection that trickles data
# back slowly can dodge that and hang way longer than expected. Running
# each call in its own daemon thread and using thread.join(timeout=...)
# gives us a real, unavoidable deadline - if the thread's still alive
# after that, we just walk away from it and move on, no matter what it's
# doing internally.
PER_CATEGORY_HARD_TIMEOUT = 12  # seconds

# HARD ceiling on the whole fetch operation across all categories, so a
# string of slow ones can't add up to several minutes of hanging.
TOTAL_FETCH_HARD_TIMEOUT = 70  # seconds


def is_cache_fresh():
    """Checks if we already have reasonably recent news saved locally."""
    if not os.path.exists(NEWS_CACHE_FILE):
        return False
    try:
        with open(NEWS_CACHE_FILE, "r", encoding="utf-8") as f:
            cache_info = json.load(f)
        fetched_at = cache_info.get("fetched_at", 0)
        age = time.time() - fetched_at
        return age < CACHE_FRESH_SECONDS
    except Exception:
        return False


def _do_request(category):
    """The actual network call."""
    params = {
        "category": category,
        "lang": "en",
        "max": ARTICLES_PER_CATEGORY,
        "apikey": GNEWS_API_KEY,
    }
    response = requests.get(GNEWS_URL, params=params, timeout=(5, 10))
    response.raise_for_status()
    data = response.json()
    return data.get("articles", [])


def _request_worker(category, result_holder):
    """Runs inside the daemon thread. Stores whatever happens (success or
    error) into result_holder so the main thread can check it after."""
    try:
        result_holder["data"] = _do_request(category)
    except Exception as e:
        result_holder["error"] = e


def fetch_category(category):
    """Grabs top headlines for one category, with a genuine hard deadline.
    Returns a list of articles, or an empty list if anything goes wrong
    (slow, errored, or timed out) - one bad category should never be able
    to hang or crash the whole refresh.

    We run the request in its own daemon thread. daemon=True means this
    thread can NEVER block the app from continuing or shutting down, even
    if it's still stuck mid-request when we give up on it - we just walk
    away and let it die quietly in the background whenever it eventually
    finishes (or the whole process exits)."""
    print(f"   Grabbing '{category}' headlines...", flush=True)

    result_holder = {}
    thread = threading.Thread(
        target=_request_worker, args=(category, result_holder), daemon=True
    )
    thread.start()
    thread.join(timeout=PER_CATEGORY_HARD_TIMEOUT)

    if thread.is_alive():
        print(f"   '{category}' took too long (over {PER_CATEGORY_HARD_TIMEOUT}s), skipping it.", flush=True)
        return []

    if "error" in result_holder:
        print(f"   Couldn't fetch '{category}' news, skipping it: {result_holder['error']}", flush=True)
        return []

    return result_holder.get("data", [])


def fetch_and_save_news(force=False):
    """The main function - goes and fetches fresh news for every category
    and saves it all to a text file our RAG pipeline can read."""

    if not GNEWS_API_KEY:
        print(
            "No GNEWS_API_KEY set, so we can't fetch live news. "
            "The app will fall back to the offline sample instead. "
            "Get a free key at https://gnews.io",
            flush=True,
        )
        return False

    if not force and is_cache_fresh():
        print("News cache is still fresh (less than 3 hours old), skipping fetch.", flush=True)
        return False

    print("Fetching today's top headlines across categories...", flush=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    all_articles = []
    start_time = time.time()

    for category in CATEGORIES:
        # If we're already past our total time budget, stop trying more
        # categories and just go with whatever we've got so far.
        elapsed = time.time() - start_time
        if elapsed > TOTAL_FETCH_HARD_TIMEOUT:
            print(
                f"   Hit the overall {TOTAL_FETCH_HARD_TIMEOUT}s fetch budget, "
                f"stopping here with {len(all_articles)} articles so far.",
                flush=True,
            )
            break

        articles = fetch_category(category)
        for article in articles:
            article["category"] = category
        all_articles.extend(articles)
        # Being polite to the API and not hammering it back to back.
        time.sleep(0.3)

    if not all_articles:
        print("Didn't manage to fetch any articles. Keeping old data (if any).", flush=True)
        return False

    # Now let's write everything out into one readable text file.
    with open(NEWS_TEXT_FILE, "w", encoding="utf-8") as f:
        for article in all_articles:
            title = article.get("title", "").strip()
            description = article.get("description", "").strip()
            content = article.get("content", "").strip()
            source = article.get("source", {}).get("name", "Unknown source")
            published = article.get("publishedAt", "")
            category = article.get("category", "general")
            url = article.get("url", "")

            f.write(f"CATEGORY: {category.upper()}\n")
            f.write(f"HEADLINE: {title}\n")
            f.write(f"SOURCE: {source}\n")
            f.write(f"PUBLISHED: {published}\n")
            f.write(f"SUMMARY: {description}\n")
            if content:
                f.write(f"DETAILS: {content}\n")
            f.write(f"LINK: {url}\n")
            f.write("\n---\n\n")

    # Save the timestamp so we know how fresh this news is.
    with open(NEWS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "fetched_at": time.time(),
                "article_count": len(all_articles),
                "categories": CATEGORIES,
            },
            f,
        )

    # Now that we've got real news, get rid of the offline fallback notice
    # so it doesn't get mixed in and confuse the chatbot's answers.
    fallback_file = os.path.join(DATA_DIR, "offline_fallback.txt")
    if os.path.exists(fallback_file):
        os.remove(fallback_file)

    print(f"Done! Saved {len(all_articles)} articles across {len(CATEGORIES)} categories.", flush=True)
    return True


if __name__ == "__main__":
    fetch_and_save_news(force=True)
