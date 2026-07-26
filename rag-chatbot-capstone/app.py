"""
app.py
------
This runs the actual website. Here's the flow in plain English:

STARTUP:
  1. We check if we have reasonably fresh news saved locally (less than
     3 hours old). If not (or it's our very first run), we go fetch fresh
     headlines from GNews across a bunch of categories.
  2. We build a lightweight TF-IDF search index from that news text (see
     search_index.py for why we use TF-IDF instead of a heavy AI model -
     short version: memory limits on the free hosting tier).

WHEN SOMEONE ASKS A QUESTION:
  1. We search the index for the news chunks that share the most
     important words with the question.
  2. We hand those chunks + the question to an LLM (Groq, free & fast)
     and say "answer using ONLY this info."
  3. We send the answer back.

WHEN SOMEONE HITS THE REFRESH BUTTON:
  We go grab fresh headlines again (unless we just refreshed super
  recently - free tier API quota is limited, gotta be careful).

IMPORTANT DESIGN NOTE: all of this loading happens in a BACKGROUND
THREAD, not before the server starts. That's on purpose - hosting
platforms like Render expect your app to open its port quickly. If we
did all this fetching/building BEFORE opening the port, the platform
would think the app crashed and give up on it. So instead: open the
port immediately, then quietly load everything in the background.
"""

import os
import time
import pickle
import threading
import requests
from flask import Flask, render_template, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity

from search_index import build_search_index, INDEX_FILE
from fetch_news import fetch_and_save_news, is_cache_fresh

app = Flask(__name__)

# Groq gives a generous free API for fast LLM calls - it runs on Groq's
# servers, not ours, so it doesn't eat into our limited RAM at all.
# Get a free key at https://console.groq.com
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# Loaded once in the background after the server starts. Routes check
# is_ready and tell people to hang on a moment if it's not ready yet.
search_data = None  # will hold {"vectorizer", "chunk_vectors", "chunks"}
is_ready = False
startup_error = None

# Basic cooldown so someone spamming the refresh button doesn't burn
# through our whole daily GNews quota in two minutes.
last_manual_refresh = 0
MANUAL_REFRESH_COOLDOWN_SECONDS = 5 * 60  # 5 minutes


def load_index_from_disk():
    """Reads the saved TF-IDF index back into memory."""
    with open(INDEX_FILE, "rb") as f:
        return pickle.load(f)


def load_everything():
    """Fetches news if needed, then builds (or rebuilds) the search index."""
    global search_data, is_ready, startup_error

    try:
        print("Booting up the news chatbot in the background...", flush=True)

        # Step 1: grab fresh news if our cache is stale (or missing).
        fetch_and_save_news(force=False)

        # Step 2: build the lightweight search index from data/.
        print("Building the search index...", flush=True)
        build_search_index()

        print("Loading the search index into memory...", flush=True)
        search_data = load_index_from_disk()

        is_ready = True
        print("All set. Chatbot is ready!", flush=True)

    except Exception as e:
        startup_error = str(e)
        print(f"Startup failed: {e}", flush=True)


def get_relevant_chunks(question, k=4):
    """Finds the k most relevant news chunks for the question using
    TF-IDF + cosine similarity (fancy way of saying: how much do the
    important words in the question overlap with each chunk)."""
    vectorizer = search_data["vectorizer"]
    chunk_vectors = search_data["chunk_vectors"]
    chunks = search_data["chunks"]

    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, chunk_vectors)[0]

    # Grab the indexes of the top k most similar chunks.
    top_indexes = similarities.argsort()[::-1][:k]

    return [chunks[i] for i in top_indexes if similarities[i] > 0]


def ask_llm(question, context_chunks):
    """Sends the question + relevant news context to the LLM."""

    if not context_chunks:
        return (
            "I couldn't find anything in today's headlines that matches "
            "your question. Try asking about a major topic in world news, "
            "business, tech, sports, health, or science."
        )

    if not GROQ_API_KEY:
        joined = "\n\n---\n\n".join(context_chunks)
        return (
            "(No GROQ_API_KEY set, so I can't write you a proper answer yet "
            "- but here's the most relevant news I found:)\n\n" + joined
        )

    context_text = "\n\n".join(context_chunks)
    system_prompt = (
        "You are a helpful, friendly news assistant. Answer the user's "
        "question using ONLY the news snippets provided below. Summarize "
        "in simple, everyday language like you're catching a friend up on "
        "the news. Mention the source name if it's relevant. If the "
        "snippets don't actually cover what's being asked, say so honestly "
        "instead of guessing or making something up - and mention that "
        "this covers top headlines from a limited set of categories, not "
        "every news story out there."
    )
    user_prompt = f"News snippets:\n{context_text}\n\nQuestion: {question}"

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 600,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(GROQ_URL, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Oops, something went wrong talking to the AI model: {e}"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    if startup_error:
        return jsonify({
            "answer": f"The chatbot failed to start up properly: {startup_error}"
        })
    if not is_ready:
        return jsonify({
            "answer": "Still warming up (fetching news and building the search index) - this can take up to about 60-90 seconds on first load, give it a moment and try again!"
        })

    data = request.get_json()
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"answer": "You didn't actually type a question!"})

    chunks = get_relevant_chunks(question)
    answer = ask_llm(question, chunks)

    return jsonify({"answer": answer})


@app.route("/refresh", methods=["POST"])
def refresh():
    """Lets someone manually pull fresh headlines instead of waiting for
    the 3-hour auto-refresh window. Has a cooldown so it can't be spammed."""
    global last_manual_refresh, search_data

    if not is_ready:
        return jsonify({
            "status": "not_ready",
            "message": "Still starting up, give it a moment before refreshing."
        })

    now = time.time()
    if now - last_manual_refresh < MANUAL_REFRESH_COOLDOWN_SECONDS:
        wait_left = int(MANUAL_REFRESH_COOLDOWN_SECONDS - (now - last_manual_refresh))
        return jsonify({
            "status": "cooldown",
            "message": f"Already refreshed recently, hang on {wait_left} more seconds."
        })

    fetched = fetch_and_save_news(force=True)
    if fetched:
        build_search_index()
        search_data = load_index_from_disk()
        last_manual_refresh = now
        return jsonify({"status": "ok", "message": "News refreshed successfully!"})
    else:
        return jsonify({
            "status": "failed",
            "message": "Couldn't fetch fresh news (check GNEWS_API_KEY is set)."
        })


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "ready": is_ready,
        "startup_error": startup_error,
        "news_cache_fresh": is_cache_fresh(),
    })


# Start loading news + building the search index in the background, so
# the port opens immediately and the host doesn't think the app crashed.
threading.Thread(target=load_everything, daemon=True).start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
