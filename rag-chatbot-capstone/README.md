# 🌍 World News RAG Chatbot (Beginner Friendly)

A chatbot that answers questions using **real, current top headlines**
pulled from a live news API, using the RAG (Retrieval-Augmented
Generation) pattern - same core idea as the first version of this project,
just pointed at live news instead of static text files.

## ⚠️ Important - what this actually covers (read this first)

No API, free or paid, can give you "literally every news story happening
in the world" - that doesn't exist. What this app actually does:

- Pulls **top headlines** from [GNews](https://gnews.io) (a free news API)
  across 9 categories: general, world, business, technology,
  entertainment, sports, science, health, and nation.
- Grabs ~6 articles per category (~54 articles) each refresh.
- Refreshes automatically if the cached news is older than 3 hours, or
  whenever you click the "Refresh news" button (max once every 5 minutes,
  to protect the free API quota of 100 requests/day).

So it's genuinely aware of **today's major headlines across those
categories** - not small local stories, not everything ever published,
and not stuff older than what GNews indexes. Be upfront about this scope
if you're presenting it - "aware of today's top headlines across major
categories" is accurate, "aware of every single news story" is not.

## Project structure

```
rag-chatbot-capstone/
├── data/
│   ├── news_articles.txt       <- auto-generated from live news on startup
│   └── offline_fallback.txt    <- only used if no GNEWS_API_KEY is set
├── vector_db/                  <- gets auto-built on every startup
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── app.py                      <- the Flask app
├── fetch_news.py                <- pulls headlines from GNews
├── build_vector_store.py       <- builds the FAISS index from news text
├── requirements.txt
├── Procfile
├── .env.example
└── README.md
```

## The 9 news categories it covers

Every question you ask gets matched against headlines from these
categories specifically. Here's what's actually in each one, with example
questions you can ask:

| Category | What it covers | Example question |
|---|---|---|
| `general` | Broad/top mixed headlines | "What's the biggest news today?" |
| `world` | International affairs, global events | "What's happening internationally?" |
| `business` | Markets, companies, economy | "Any major business news today?" |
| `technology` | Tech industry, gadgets, AI, software | "What's new in tech today?" |
| `entertainment` | Movies, music, celebrities, TV | "Any entertainment news?" |
| `sports` | Sports results, matches, athletes | "What's going on in sports?" |
| `science` | Scientific discoveries, research | "Any science news today?" |
| `health` | Medical news, health research | "What's the latest health news?" |
| `nation` | Domestic/national headlines (defaults to US since no country is set — see note below) | "What's the top national news?" |

**Note on `nation`:** GNews's `nation` category pulls domestic headlines
for a specific country, and we haven't set one in `fetch_news.py`, so it
defaults to US national news. If you want it to reflect your own country
instead, open `fetch_news.py` and add `"country": "in"` (or your country's
2-letter code) to the `params` dict inside `fetch_category()`.

Anything outside these 9 categories (e.g. hyper-local news, a very
specific niche topic) likely won't be covered - the bot will say so
honestly rather than guess.

## Getting your free API keys (takes about 3 minutes total)

1. **GNews** (for real news): go to [gnews.io](https://gnews.io), sign up
   free, copy your API key from the dashboard. Free tier = 100
   requests/day, non-commercial/personal/academic use - perfect for a
   student project.
2. **Groq** (for the AI to write proper answers): go to
   [console.groq.com](https://console.groq.com), sign up free, create an
   API key. Without this key the app still runs, it just shows you the
   raw matched headlines instead of a written summary.

## Running it locally

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt

set GNEWS_API_KEY=your_key_here      # Windows
set GROQ_API_KEY=your_key_here
export GNEWS_API_KEY=your_key_here   # Mac/Linux
export GROQ_API_KEY=your_key_here

python app.py
```

Open `http://localhost:5000`. First run takes longer since it's fetching
news, downloading the embedding model, and building the vector database -
that's normal.

## Deploying to Render (free tier)

1. Push this whole folder to a new GitHub repository.
2. On [render.com](https://render.com): New + → Web Service → connect repo.
3. Settings:
   - **Root Directory:** blank if `app.py` is at repo root, otherwise the
     folder name it's nested inside.
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --timeout 120`
4. Environment variables:
   - `GNEWS_API_KEY` = your free GNews key
   - `GROQ_API_KEY` = your free Groq key
   - `PYTHON_VERSION` = `3.11.9`
5. Deploy and test at the `.onrender.com` URL.

**Free tier heads-up:** Render's free web services spin down when idle
and lose their disk on restart. That means every time it wakes back up,
it re-fetches news and rebuilds everything from scratch - which uses a
few GNews API calls and takes some time (30-90 seconds). The 3-hour
freshness check and 5-minute refresh cooldown exist specifically to keep
this from accidentally burning through the 100 requests/day free limit
if the app gets restarted a lot.

## Example questions you can ask

- "What's the top tech news today?"
- "Any major business headlines?"
- "What's going on in sports right now?"
- "Summarize today's health news"
- "What's the biggest story in general news today?"

Questions about very specific/local topics not covered in the fetched
categories, or events older than the current news cycle, likely won't
have a good answer - that's expected given the scope above.
