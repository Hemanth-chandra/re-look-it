# Re-look-it

Review intelligence for Indian quick-commerce apps (Blinkit, Zepto, BigBasket, Swiggy Instamart).
195,875 Google Play reviews (1 Jul – 30 Sep 2026) scored by one multi-task Twitter-RoBERTa model:
sentiment with a confidence band, urgency (Low / Medium / High) with a reason, 8 issue types, and the
words that drove each prediction.

**Team:** Rasee Singh (2025ms_016), S Hemanth Chandra (2025ms_020) · **Mentor:** Dr. Yashwant Singh Patel, IIITM Gwalior

## Pages

| Page | What it shows |
|---|---|
| Overview | Negative share, star rating and High-urgency share per app; weekly trend; sentiment mix; issues in negative reviews; confidence bands; spike days |
| Competitor comparison | Zepto vs the rest with bootstrap CIs (model or star ratings); issue heatmap; biggest weakness per app; daily spikes with evidence; app versions; how each app replies |
| Priority queue | High and Medium reviews sorted by urgency level, then score; filters, search, full review with why-words, the app's reply, CSV download |
| Analyse a review | Paste any review and run the full deployed pipeline live |
| About the model | Architecture, results on 400 human-checked reviews, ablation, calibration, training curves, limitations |

## Repository layout

```
streamlit_app.py          entry point: navigation + sidebar filters
app_pages/                one file per page
core/model.py             model class, download of model.pt, live pipeline (analyse)
core/text_rules.py        text cleaning, keyword urgency rules, clean-up helpers (same as the notebook)
core/keyword_aspects.py   spaCy keyword matcher for issue types (same as aspects/extract_aspects.py)
core/lexicon.py           issue-type lexicon
core/data.py, theme.py    cached loaders, colours and styling
data/                     app_data.parquet (final predictions) + result tables
model/                    config, tokenizer, settings.json   (model.pt comes from the GitHub Release)
assets/                   loss curves
.streamlit/config.toml    warm theme
```

## Deploy in 4 steps

### 1. Create the GitHub repository and upload the files
1. On github.com click **New repository**, name it `relookit-app`, set it to **Public**, and create it.
2. Click **uploading an existing file**, then drag in everything inside this folder (all files and the folders
   `app_pages`, `core`, `data`, `model`, `assets`, `.streamlit`). Click **Commit changes**.
   - If the `.streamlit` folder does not upload (hidden folders are sometimes skipped), click **Add file → Create new file**,
     type `.streamlit/config.toml` as the name and paste the contents of that file.

### 2. Attach the model weights to a Release
`model.pt` is about 496 MB. A normal commit allows at most 100 MB per file, so it goes in a Release.
1. Download `model.pt` from the Drive folder `relookit/outputs/model/`.
2. In the repository click **Releases → Create a new release**.
3. **Choose a tag:** type `v1.0` and click *Create new tag*. Title: `Model v3`.
4. Drag `model.pt` into **Attach binaries** and wait for the upload to finish. Click **Publish release**.
5. Right-click `model.pt` under *Assets* and copy the link. It looks like
   `https://github.com/<your-username>/relookit-app/releases/download/v1.0/model.pt`.

### 3. Point the app at the model
Open `core/model.py` on GitHub, click the pencil icon, and replace the `MODEL_URL` line with your link.
Commit the change.

### 4. Deploy on Streamlit Community Cloud
1. Go to **share.streamlit.io** and sign in with GitHub.
2. Click **Create app → Deploy a public app from GitHub**.
3. Repository: `<your-username>/relookit-app`, branch `main`, main file `streamlit_app.py`.
4. Open **Advanced settings** and choose **Python 3.12** (the PyTorch line in `requirements.txt` is built for 3.12).
5. Click **Deploy**. The first build takes about 5 minutes. The first visit to *Analyse a review* downloads the
   model once (about a minute); after that it stays in memory.

The app sleeps after a few days without visitors. Opening it wakes it up (and downloads the model again on first use).

## Run locally
```bash
pip install -r requirements.txt        # on Python 3.12, Linux. Elsewhere: pip install torch, then the rest
# put model.pt in model/  (or let the app download it from MODEL_URL)
streamlit run streamlit_app.py
```

## How the live pipeline matches the project
For a new review, *Analyse a review* applies exactly the deployed rules from `settings.json` and the notebook:
- **Sentiment:** logits / T (0.61), neutral logit −0.9, softmax. Band High ≥ 0.893, Medium ≥ 0.821, else Low.
  Reviews in non-Latin scripts are always Low.
- **Urgency:** the higher of the model and the keyword rules. Score = min(1, max(p_High + 0.5·p_Medium, rule score) + 0.05·log(1 + likes)).
  The rules need to know whether the review is negative: the star rating if given, otherwise the model's sentiment.
- **Issues:** model probability above the per-issue threshold **or** a keyword match; reviews under 3 words get none.
- **Why-words:** gradient × input on the predicted sentiment class plus the predicted urgency, top 3 cleaned words.

## Data notes
- `data/app_data.parquet`: final predictions for all 195,875 reviews (22 columns).
- `data/model_tables/`, `data/star_tables/`: competitor and spike tables from `competitor_comparison.py` and
  `spike_analysis.py`, re-run on the final predictions (model) and on star ratings.
- Results in *About the model* come from `before_after.csv`, built in the final notebook on the 400 human-checked reviews.
