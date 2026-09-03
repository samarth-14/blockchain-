# Face → Web Search → Blockchain Verification

A **non-commercial research / demo** pipeline. It takes a single face image,
discovers publicly available candidate images via reverse image search, finds
the strongest visual match, fingerprints the discovered metadata, and anchors
that fingerprint on the Ethereum **Sepolia** testnet so it can later be verified
as `VERIFIED` / `TAMPERED`.

> ⚠️ **Scope & ethics.** This project is for demonstration with a **consenting
> participant or a controlled test image only**. It must **not** be used for mass
> identification, surveillance, scraping login-protected/private accounts, or
> building dossiers on people. Only public web content and a controlled test
> image are in scope.

## Status

| Phase | Description | State |
|------:|-------------|-------|
| **1** | Local face detection + embedding (InsightFace, CPU) | ✅ **Implemented** |
| **2** | Reverse image search via SerpApi Google Lens | ✅ **Implemented** |
| 3 | Download + match candidate faces | ⏳ stub |
| 4 | SHA-256 fingerprint + Sepolia record/verify | ⏳ stub |

Phases 3–4 are intentionally unimplemented placeholders that raise
`NotImplementedError`. No fake API responses, no hardcoded match URLs — the
Phase 2 search makes a genuine external SerpApi request every run.

## Requirements

- Python **3.11+**
- macOS/Linux, CPU is fine (ONNX Runtime CPU by default)

## Setup

```bash
cd face-web-blockchain

# 1. Create and activate a virtual environment (Python 3.11+)
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Configure environment (Phase 2/4 secrets — optional for Phase 1)
cp .env.example .env      # then edit .env; never commit it
```

On first run InsightFace downloads its model pack (`buffalo_l`, ~300 MB) into
`~/.insightface`. This is a one-time download.

## Run — Phase 1

Place a test image (one clear, consenting face) at `data/input/test.jpg`, then:

```bash
python app.py --image data/input/test.jpg
```

Expected output:

```
[1/2] Loading image  (data/input/test.jpg)
  Image loaded: WxH px
[2/2] Detecting face
✓ Face detected
  Bounding box (x1,y1,x2,y2): (...)
  Detector confidence: 0.9xxx
✓ Embedding generated
  Embedding: [+0.0123, ... dim=512]
  Face crop saved: data/results/test_face.jpg

✓ Phase 1 complete.
```

Useful flags: `--crop-out PATH`, `--model buffalo_s`, `--det-size 640`,
`--det-threshold 0.5`.

### MVP face rules

- **0 faces →** rejected (`No face detected`, exit code 3)
- **>1 face →** rejected (`... detected N`, exit code 4) — keeps the demo to a
  single consenting subject
- exactly 1 face → returns bounding box + 512-d L2-normalized embedding, saves a
  face crop

## Phase 2 — Reverse image search (SerpApi Google Lens)

Phase 2 takes the validated input image and performs a **genuine** reverse image
search to discover where that image (or visually similar ones) appears on the
public web. It uses the official [SerpApi Google Lens API](https://serpapi.com/google-lens-api)
— no manual Google scraping, no TinEye, no hardcoded or mocked results.

### Architecture / data flow

```
input image
  └─ (Phase 1) detect single face  ─ gate: exactly one consenting face
  └─ upload image to SerpApi Image API  ──►  image_id      (data stays with SerpApi)
        POST https://serpapi.com/image   (multipart, ≤500 KB, field "image")
  └─ Google Lens search by image_id     ──►  raw JSON
        GET https://serpapi.com/search?engine=google_lens&image_id=...
  └─ parse `visual_matches` (+ `exact_matches`)  ──►  list[Candidate]
        {position, title, url, source, domain, thumbnail_url, image_url, result_type}
```

Why the two-step upload? Google Lens needs an image it can fetch — either a
public URL or an `image_id` from SerpApi's own upload endpoint. We use the
**upload endpoint** so a *local* file never has to be hosted on an arbitrary
third-party service; the bytes go only to SerpApi. The image-submission strategy
is factored out in `pipeline/reverse_search.py` (`search_image(image_path=...)`
uploads; `search_image(image_url=...)` passes a public URL straight through).

Phase 3 will consume the returned `Candidate` objects (downloading `image_url` /
`url` where publicly accessible and matching faces against the input embedding).

### Get and configure a SerpApi key

1. Create a free account at <https://serpapi.com/users/sign_up>.
2. Copy your key from <https://serpapi.com/manage-api-key>.
3. Put it in `.env` (git-ignored — never commit it):

```env
SERPAPI_API_KEY=your_real_key_here
SERPAPI_ENGINE=google_lens
```

The key is read via `config.py` / `python-dotenv`. It is **never printed** and
**never written** to `data/results/lens_response.json` (that file is scrubbed of
any `api_key` field before writing).

### Run a real search

```bash
python app.py --image data/input/test.jpg
# optionally dump the raw (credential-scrubbed) SerpApi JSON:
python app.py --image data/input/test.jpg --debug-search
```

Stages become:

```
[1/3] Loading image
[2/3] Detecting face
[3/3] Searching web with Google Lens

==================================================
GOOGLE LENS SEARCH
==================================================

✓ Search completed
  Candidates found: N

[1] <title>
    Source: <domain>
    URL:    <url>
...
```

`--debug-search` writes the full raw response to
`data/results/lens_response.json`. It can be large, so it is **not** shown by
default. Use `--max-show N` to change how many candidates print.

### Free-plan / quota note

The SerpApi free plan allows **~100 searches/month**. Each `app.py` run performs
**one** upload + **one** Lens search (which counts as one search). The client
makes **no automatic retries** on rate-limit/5xx errors, so a transient failure
never silently burns extra quota — you see the error and decide.

### Privacy & ethics note

This is a **non-commercial hackathon / research demo**. Only run it against a
**consenting participant or a controlled public/test image**. Uploading an image
to SerpApi sends it to a third-party API (SerpApi → Google Lens); do not submit
images of non-consenting individuals. The tool must not be used for mass
identification, surveillance, or building dossiers on people, and it does not
scrape login-protected/private accounts.

### Troubleshooting

| Symptom | Exit code | Fix |
|---------|:--------:|-----|
| `SERPAPI_API_KEY is not set` | 6 | Add the key to `.env`. |
| `SerpApi rejected the API key` | 7 | Key is wrong/expired — recopy from the dashboard. |
| `rate limit / quota exceeded` | 8 | Free monthly quota used up; wait or upgrade. |
| `Network/timeout during search` | 9 | Check connectivity; retry manually. |
| `Could not parse SerpApi response` | 10 | Transient API error / no results field — try `--debug-search`. |
| `Image problem for search` | 11 | Image missing/corrupt/too large to compress under 500 KB. |
| `Candidates found: 0` | 0 (success) | Genuinely no matches for that image — try a more public image. |

## Tests

```bash
pytest -q
```

Phase 2 tests **mock the HTTP layer only** and never call the live API or use
quota. For a genuine end-to-end check, run the opt-in live script explicitly
(needs a real key in `.env`):

```bash
./.venv/bin/python tests/live_search.py --image data/input/test.jpg
```

The fast unit tests need no model download. An integration test runs the real
model **only if** `data/input/test.jpg` exists and `insightface` is installed;
otherwise it is skipped.

## Project layout

```
face-web-blockchain/
├── app.py                  # CLI entry point (Phase 1 wired up)
├── config.py               # env-driven config (dotenv); no hardcoded secrets
├── requirements.txt
├── .env.example            # copy to .env (git-ignored)
├── pipeline/
│   ├── face_detector.py    # ✅ InsightFace detection + embedding
│   ├── reverse_search.py   # ⏳ Phase 2
│   ├── candidate_matcher.py# ⏳ Phase 3
│   ├── fingerprint.py      # ⏳ Phase 4
│   └── blockchain.py       # ⏳ Phase 4
├── utils/
│   ├── logger.py           # stage/success/failure CLI helpers
│   └── http.py             # shared requests session (for Phase 2/3)
├── data/{input,candidates,results}/
└── tests/
```

## Security notes

- Secrets live only in `.env` (git-ignored). `.env.example` documents the keys.
- Use a **throwaway** Sepolia wallet funded with test ETH only.
- Generated artifacts under `data/candidates` and `data/results` are git-ignored.
