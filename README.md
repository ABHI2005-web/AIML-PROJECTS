# AIML PROJECTS

Name- Abhigyan Gaurav

Registration no- 23BCE10378

A collection of machine learning, reinforcement learning, and full-stack web development projects covering classification, computer vision, reinforcement learning, recommendation systems, retrieval-augmented generation, and end-to-end deployment.

## Table of Contents

1. [Adult Census Income Classification](#1-adult-census-income-classification)
2. [CIFAR-10 Image Classification using CNN](#2-cifar-10-image-classification-using-cnn)
3. [Face Recognition using CNN in the Wild (LFW Dataset)](#3-face-recognition-using-cnn-in-the-wild-lfw-dataset)
4. [Cancer Detection using MRI Images](#4-cancer-detection-using-mri-images)
5. [Cart-Pole RL Agent Training](#5-cart-pole-rl-agent-training)
6. [Lunar Lander RL Agent Training](#6-lunar-lander-rl-agent-training)
7. [Movie Recommendation System](#7-movie-recommendation-system)
8. [End-to-End Render Deployment Project (Car Price Predictor)](#8-end-to-end-render-deployment-project-car-price-predictor)
9. [RAG Chatbot (Capstone Project)](#9-rag-chatbot-capstone-project)

---

## 1. Adult Census Income Classification

A binary classification project predicting whether an individual's annual income exceeds $50,000/year using demographic and employment-related attributes from the Adult Census Income dataset.

- **Tech stack:** Python, pandas, numpy, scikit-learn
- **Problem type:** Binary classification — `0` = income ≤ 50K, `1` = income > 50K
- **Folder:** `adult-census-income-classification/`
- **Code file:** `adult_census_income_classification.ipynb`
- **Dataset:** [Adult Census Income (Kaggle, uciml)](https://www.kaggle.com/datasets/uciml/adult-census-income)

### Dataset

| # | Feature | Description |
|---|---|---|
| 1 | age | Age of individual |
| 2 | workclass | Type of employment |
| 3 | fnlwgt | Final weight |
| 4 | education | Education level |
| 5 | education.num | Numerical representation of education |
| 6 | marital.status | Marital status |
| 7 | occupation | Type of occupation |
| 8 | relationship | Family relationship |
| 9 | race | Race |
| 10 | sex | Gender |
| 11 | capital.gain | Capital gains |
| 12 | capital.loss | Capital losses |
| 13 | hours.per.week | Weekly working hours |
| 14 | native.country | Country of origin |
| target | income | Target variable (>50K / ≤50K) |

> Note: this Kaggle CSV uses dot-separated column names (e.g. `education.num`) rather than the hyphenated names (`education-num`) used in some other versions of this dataset.

- No. of instances: **48,842**
- No. of features: **14**

### Pipeline

1. **Data cleaning** — missing values (represented as `?` in the raw data) replaced with `NaN` and dropped; cleaning verified via `df.shape` and `df.isnull().sum()`.
2. **Feature engineering** — categorical columns label-encoded, features/target separated, features scaled with `StandardScaler`.
3. **Train/test split** — 80/20 split, `random_state=42`.
4. **Model training** — five classifiers trained: Logistic Regression, Decision Tree, Random Forest, KNN (k=5), and SVM (with probability estimates enabled).
5. **Evaluation** — accuracy, precision, recall, F1-score, and ROC-AUC computed for each model.

### Results

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.82 | 0.72 | 0.44 | 0.55 | 0.84 |
| Decision Tree | 0.80 | 0.60 | 0.62 | 0.61 | 0.74 |
| Random Forest | **0.85** | 0.73 | **0.62** | **0.67** | **0.90** |
| KNN | 0.82 | 0.66 | 0.58 | 0.62 | 0.84 |
| SVM | 0.84 | 0.73 | 0.54 | 0.62 | 0.89 |

Random Forest gave the best overall performance across accuracy, F1-score, and ROC-AUC.

### Run it

```bash
pip install pandas numpy scikit-learn
```
Open `adult_census_income_classification.ipynb` in Jupyter/Colab and run all cells.

---

## 2. CIFAR-10 Image Classification using CNN

An image classification project on the CIFAR-10 dataset (60,000 32×32 color images across 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) using a Convolutional Neural Network.

- **Tech stack:** Python, TensorFlow/Keras, matplotlib, numpy
- **Folder:** `cifar10-cnn-classification/`
- **Code file:** `cifar10_cnn_classification.ipynb`
- **Dataset:** [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (loaded via `tensorflow.keras.datasets.cifar10`, no manual download needed)

### What the code does

1. Loads CIFAR-10 directly via `tensorflow.keras.datasets.cifar10` (no manual download needed) and previews sample images.
2. Normalizes pixel values to `[0, 1]` and one-hot encodes the 10 class labels.
3. Builds a CNN with 3 convolutional blocks (Conv2D → BatchNorm → Conv2D → MaxPool → Dropout, increasing filters 32 → 64 → 128), followed by a dense classification head.
4. Trains for 20 epochs with a 10% validation split.
5. Evaluates on the 10,000-image test set and plots accuracy/loss curves.
6. Saves the trained model to `cifar10_cnn_model.keras`.

### Run it

```bash
pip install tensorflow matplotlib numpy
```
Open `cifar10_cnn_classification.ipynb` in Jupyter/Colab (GPU runtime recommended) and run all cells.

---

## 3. Face Recognition using CNN in the Wild (LFW Dataset)

A face recognition project that classifies a person's identity from a face image using a CNN, trained on the Labeled Faces in the Wild (LFW) dataset.

- **Tech stack:** Python, scikit-learn (for dataset loading), TensorFlow/Keras, matplotlib, numpy
- **Folder:** `face-recognition-lfw/`
- **Code file:** `face_recognition_lfw_cnn.ipynb` 
- **Dataset:** [Labeled Faces in the Wild (LFW)](https://vis-www.cs.umass.edu/lfw/) (loaded via `sklearn.datasets.fetch_lfw_people`, auto-downloads and caches)

### What the code does

1. Loads the dataset via `sklearn.datasets.fetch_lfw_people` (auto-downloads and caches), keeping only people with at least 70 images so every class has enough samples.
2. Previews sample faces and prints dataset stats (image count, dimensions, number of people/classes).
3. Normalizes images to `[0, 1]` and reshapes them for CNN input.
4. Builds a CNN with 3 convolutional blocks + dropout, followed by a dense classification head sized to the number of people in the dataset.
5. Trains for 30 epochs with a stratified 80/20 train/test split.
6. Evaluates test accuracy/loss and plots training curves.
7. Saves the trained model to `lfw_face_recognition_cnn.keras`.

### Run it

```bash
pip install scikit-learn tensorflow matplotlib numpy
```
Open `face_recognition_lfw_cnn.ipynb` in Jupyter/Colab (GPU runtime recommended) and run all cells.

---

## 4. Cancer Detection using MRI Images

A binary classifier that detects the presence of a brain tumor from MRI scans using a CNN.

- **Tech stack:** Python, TensorFlow/Keras, scikit-learn, matplotlib, numpy, Pillow
- **Problem type:** Binary classification — `0` = No tumor, `1` = Tumor
- **Folder:** `cancer-detection-mri/`
- **Code file:** `cancer_detection_mri_cnn.ipynb`
- **Dataset:** [Brain MRI Images for Brain Tumor Detection (Kaggle, Navoneel Chakrabarty)](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) — 253 grayscale brain MRI images

### Expected data layout

```
data/
├── yes/    # MRI images that show a tumor (~155 images)
└── no/     # MRI images with no tumor (~98 images)
```

### What the code does

1. Loads and augments images from the `data/yes` and `data/no` folders using `ImageDataGenerator` (rescaling, rotation, zoom, shift, horizontal flip — helpful here since the dataset is small), with an 80/20 train/validation split.
2. Previews sample MRI images with their labels.
3. Builds a CNN with 3 convolutional + max-pooling blocks, followed by a dense head with a sigmoid output for binary classification.
4. Trains for 25 epochs with a small batch size (16), suited to the small dataset.
5. Evaluates validation accuracy/loss, prints a confusion matrix and classification report, and plots training curves.
6. Saves the trained model to `cancer_detection_mri_cnn.keras`.

### Run it

```bash
pip install tensorflow scikit-learn matplotlib numpy pillow
```
Open `cancer_detection_mri_cnn.ipynb` in Jupyter/Colab (GPU runtime recommended) and run all cells — including the upload cell if using Colab.

---

## 5. Cart-Pole RL Agent Training

A Flask web app that serves a trained Proximal Policy Optimization (PPO) agent playing OpenAI Gym's `CartPole-v1` — click "Run Episode" and it plays one episode live, rendered server-side as a GIF (no local install needed to view it). It **does not retrain anything**, it only loads already-trained weights.

- **Folder:** `cartpole-ppo-webapp/`
- **Tech stack:** Flask, Gymnasium, Pillow, NumPy, gunicorn
- **Live Demo:** **https://cartpole-ppo.onrender.com**

**Why NumPy instead of PyTorch for the demo:** the full PyTorch install (even CPU-only) uses more RAM than fits on Render's free tier alongside gymnasium and pygame. The trained weights are exported from the original PyTorch model into `ppo_cartpole_weights.npz`, and `app.py` runs inference with a pure NumPy forward pass that mirrors the trained actor network exactly.

### Files

| File | Purpose |
|---|---|
| `app.py` | Flask server — loads the NumPy weights, runs episodes, renders GIFs |
| `templates/index.html` | The demo page (button + GIF viewer + stats) |
| `ppo_cartpole_weights.npz` | Trained policy weights, exported from PyTorch for NumPy inference |
| `requirements.txt` | Python dependencies |
| `.python-version` | Pins Python to 3.11.9 (avoids build failures on newer Python) |

**Memory optimizations for the free tier:** a full 500-step episode at full resolution can use 300+ MB of raw frame data alone, so the app only keeps every 3rd rendered frame (`FRAME_SKIP = 3`) and shrinks each frame to half size (`GIF_SCALE = 0.5`) before encoding, to fit within a 512 MB instance alongside Flask/gymnasium/pygame's own baseline memory use.

### Run it locally

```bash
pip install -r requirements.txt
python app.py
```
Open `http://localhost:5000`.

### Deploy on Render

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn app:app`
- **Environment Variable:** `SDL_VIDEODRIVER=dummy` (required — lets CartPole render frames without a real display on the server)

### Notes

- First load after inactivity may take ~30–60 seconds on Render's free tier (server "spins up" from sleep) — normal, just let it load once before showing it live.
- Each click runs a fresh episode with the policy acting greedily (no randomness in action selection), so results vary episode to episode, typically landing somewhere in the 200–500 step range.

---

## 6. Lunar Lander RL Agent Training

Trains an AI to safely land a spacecraft using Gymnasium's `LunarLander-v3` environment, PPO via Stable-Baselines3, with a lightweight Flask site to showcase results.

- **Tech stack:** Python, Gymnasium, Stable-Baselines3, Flask
- **Folder:** `lunarlander-rl/`
- **Live Demo:** **https://my-demo-3-jbwe.onrender.com**

### Two-part architecture

1. **Training** (`train.py`) — run once locally to train the agent (needs heavy packages: PyTorch, Box2D).
2. **Website** (`app.py`) — a small Flask site deployed to Render that only needs Flask, so it deploys fast and reliably, serving pre-saved results instead of retraining on the server.

A fully trained model is included — trained for **1,050,000 steps**, reaching an average score of **236.64** over 30 test games (above the 200 "solved" threshold). A landing video and training progress graph are included, so the site works immediately without retraining.

### Folder overview

```
lunarlander-rl/
├── train.py                      # Trains the AI (run locally)
├── app.py                        # Flask site shown on Render
├── requirements.txt              # Small package list, used by Render
├── requirements-training.txt     # Big package list, used only for training
├── Procfile                      # Tells Render how to start the site
├── runtime.txt                   # Python version for Render
├── models/
│   └── ppo_lunar_lander.zip      # Trained AI (236.64 avg score)
├── static/
│   ├── results.json
│   ├── training_reward_graph.png
│   └── landing_video.mp4
├── templates/
│   └── index.html
└── logs/
    └── monitor.csv
```

### Run it locally

```bash
pip install -r requirements.txt
python app.py
```
Open `http://localhost:5000`.

### Retrain (optional)

```bash
pip install -r requirements-training.txt
python train.py
```
~15–25 minutes for the default 1,000,000 steps. A resumable variant, `train_chunk.py`, trains in smaller chunks:
```bash
python train_chunk.py 200000
```

### Deploy on Render

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn app:app`
- **Python Version:** pinned via `runtime.txt` (3.11.9)

### Common issues

- **"Application failed to respond"** — check the Start Command is exactly `gunicorn app:app` and `gunicorn` is in `requirements.txt`.
- **Build fails** — check the Render Root Directory matches where the files live in the repo.
- **Box2D install error** — install `swig` first, then re-run `pip install gymnasium[box2d]`.

---

## 7. Movie Recommendation System

A movie recommendation web app built with Flask and **item-based collaborative filtering**, using the real **MovieLens 100K** dataset (100,000 ratings from 943 users on 1,682 movies).

- **Tech stack:** Python, Flask, pandas, cosine similarity
- **Folder:** `movie-recommender/`
- **Live Demo:** **https://movie-recommendation-system-vz7v.onrender.com**
- **Dataset:** [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) — **not committed to the repo** (per course policy on not uploading datasets to version control); fetched fresh via `download_data.py`, which converts the original files into `data/ratings.csv` and `data/movies.csv`

### How it works

1. Builds a users × movies ratings table.
2. Compares every movie to every other movie using **cosine similarity**.
3. For a chosen user, finds movies similar to ones they rated highly and recommends unwatched ones.

All logic lives in `recommender.py`.

### Project structure

```
movie-recommender/
├── app.py                # Flask website (routes/pages)
├── recommender.py        # Recommendation engine
├── download_data.py      # Downloads the real MovieLens 100K dataset
├── data/                  # Empty in the repo (.gitkeep only) — populated by download_data.py
├── templates/index.html
├── static/style.css
├── requirements.txt
├── Procfile
└── runtime.txt
```

### Run it locally

```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
python download_data.py      # required — fetches ratings.csv and movies.csv, not included in the repo
python app.py
```
Open `http://127.0.0.1:5000`.

Test the recommendation engine alone (no website): `python recommender.py`

### Deploy on Render

Since the dataset isn't in the repo, Render needs to download it as part of the build, before the app starts.

- **Build Command:** `pip install -r requirements.txt && python download_data.py`
- **Start Command:** `gunicorn app:app`
- **Python Version:** pinned via `runtime.txt` (3.11.9) — also add as an environment variable (`PYTHON_VERSION` = `3.11.9`) if the build doesn't pick it up automatically

---

## 8. End-to-End Render Deployment Project (Car Price Predictor)

A FastAPI web app that predicts a used car's resale price using a RandomForestRegressor trained on the real CarDekho used-car dataset, deployed end-to-end on Render.

- **Tech stack:** Python, FastAPI, scikit-learn (RandomForestRegressor)
- **Folder:** `car-price-prediction/`
- **Live Demo:** **https://car-price-predictor-lesn.onrender.com**
- **Dataset:** [Vehicle Dataset from CarDekho (Kaggle, nehalbirla)](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) — 301 real used-car listings: `Car_Name, Year, Selling_Price, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner`

### Project structure  

```
car-price-prediction/
├── main.py                # FastAPI app (routes + prediction logic)
├── train_model.py         # Downloads real dataset, trains, saves model
├── requirements.txt
├── Procfile
├── render.yaml
├── runtime.txt             # Pins Python to 3.11.9
├── templates/
│   └── index.html
└── static/
    └── style.css
```

### Run it locally

```bash
pip install -r requirements.txt
python train_model.py      # trains the model on the CarDekho dataset, creates model/car_price_model.pkl
uvicorn main:app --reload   # starts the site at http://localhost:8000
```

### Deploy on Render

Render auto-detects `render.yaml`. If setting manually:
- **Build Command:** `pip install -r requirements.txt && python train_model.py`
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Environment:** Python 3

### Notes

- Trained on the real CarDekho used-car dataset (301 listings) — captures genuine price depreciation patterns by car age, mileage, fuel type, transmission, and ownership history, rather than synthetic/simulated data.
- Model files are generated at build time on Render (not committed), keeping the repo lightweight.

---

## 9. RAG Chatbot (Capstone Project)

A chatbot that answers questions using **real, current top headlines** pulled from a live news API, following the Retrieval-Augmented Generation (RAG) pattern.

- **Tech stack:** Python, Flask, FAISS (vector search), GNews API, Groq API
- **Folder:** `rag-chatbot-capstone/`
- **Live Demo:** **https://news-chatbot-1-kz4n.onrender.com**

### Scope

Pulls **top headlines** from [GNews](https://gnews.io) across 9 categories — general, world, business, technology, entertainment, sports, science, health, and nation — roughly 54 articles per refresh. Refreshes automatically if cached news is older than 3 hours, or on manual request (max once every 5 minutes, to protect the free 100 requests/day API quota).

| Category | Covers | Example question |
|---|---|---|
| general | Broad/top mixed headlines | "What's the biggest news today?" |
| world | International affairs, global events | "What's happening internationally?" |
| business | Markets, companies, economy | "Any major business news today?" |
| technology | Tech industry, gadgets, AI, software | "What's new in tech today?" |
| entertainment | Movies, music, celebrities, TV | "Any entertainment news?" |
| sports | Sports results, matches, athletes | "What's going on in sports?" |
| science | Scientific discoveries, research | "Any science news today?" |
| health | Medical news, health research | "What's the latest health news?" |
| nation | Domestic/national headlines (defaults to US) | "What's the top national news?" |

### Project structure

```
rag-chatbot-capstone/
├── data/
│   ├── news_articles.txt       # auto-generated from live news on startup
│   └── offline_fallback.txt    # used only if no GNEWS_API_KEY is set
├── vector_db/                  # auto-built on every startup
├── templates/index.html
├── static/style.css
├── app.py                      # the Flask app
├── fetch_news.py                # pulls headlines from GNews
├── build_vector_store.py       # builds the FAISS index from news text
├── requirements.txt
├── Procfile
└── .env.example
```

### Setup

1. **GNews API key** — sign up free at [gnews.io](https://gnews.io) (100 requests/day free tier, non-commercial/academic use).
2. **Groq API key** — sign up free at [console.groq.com](https://console.groq.com) (used to generate written answers; without it, the app shows raw matched headlines instead).

### Run it locally

```bash
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

export GNEWS_API_KEY=your_key_here
export GROQ_API_KEY=your_key_here

python app.py
```
Open `http://localhost:5000`. First run takes longer — fetching news, downloading the embedding model, and building the vector database.

### Deploy on Render

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `gunicorn app:app --timeout 120`
- **Environment variables:** `GNEWS_API_KEY`, `GROQ_API_KEY`, `PYTHON_VERSION=3.11.9`

**Free tier note:** Render's free services spin down when idle and lose disk on restart, so the app re-fetches news and rebuilds the vector index on every wake-up (30–90 seconds, uses a few GNews calls). The 3-hour freshness check and 5-minute refresh cooldown protect the daily API quota.

### Example questions

- "What's the top tech news today?"
- "Any major business headlines?"
- "What's going on in sports right now?"
- "Summarize today's health news"

---

## Repository Structure

```
.
├── adult-census-income-classification/
├── cifar10-cnn-classification/
├── face-recognition-lfw/
├── cancer-detection-mri/
├── cartpole-ppo-webapp/
├── lunarlander-rl/
├── movie-recommender/
├── car-price-prediction/
├── rag-chatbot-capstone/
└── README.md
```


