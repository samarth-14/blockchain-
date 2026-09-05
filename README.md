# Face → Web Search → Blockchain Verification

**Phase 1 ✅ Complete | Phase 2 ✅ Complete | Phase 3 ✅ Complete | Phase 4 ✅ Complete | Phase 5 ⏳**

A **non-commercial hackathon / research demonstration** pipeline. It takes a
single face image, detects and embeds the face locally (InsightFace), then runs a
**genuine** reverse image search (SerpApi Google Lens) to discover where that
image — or visually similar images — appear on the public web.

> ⚠️ **Scope & responsible use.** This project is for demonstration with a
> **consenting participant or a public-domain / controlled test image only**. It
> must **not** be used for mass identification, surveillance, scraping
> login-protected/private accounts, or building dossiers on people. Only public
> web content and a controlled test image are in scope.

> **Pipeline scope.** Phase 2 *discovers* candidate web/search results for an
> image. Phase 3 then verifies whether a discovered page shows the **same
> person** (candidate face detection + similarity comparison). Phase 4
> fingerprints the verified evidence (SHA-256) and anchors it on the Ethereum
> **Sepolia test network**, then re-reads and re-verifies it on-chain.

---

## 1. Project overview

Given one input image the pipeline currently:

1. **Detects a single face** locally with InsightFace and produces a
   512-dimensional, L2-normalized face embedding (plus bounding box and detector
   confidence).
2. **Reverse-searches** the image with the official SerpApi Google Lens API and
   returns a list of real candidate web/search results discovered dynamically at
   runtime.

The current pipeline in one line:

```
Input image
  → InsightFace face detection + 512-d embedding
  → SerpApi Google Lens reverse image search
  → real web / search candidates (URLs discovered dynamically, not hardcoded)
```

Downstream verification — confirming a discovered candidate is the **same
person**, then fingerprinting and anchoring that result on-chain — is planned for
later phases and is **not implemented yet**.

---

## 2. Current implementation status

### Phase 1 — Local face detection ✅ **Complete**

- Local face detection using InsightFace (CPU by default)
- Single-face validation (exactly one face required)
- 512-dimensional, L2-normalized face embedding
- Bounding box + detector confidence
- Face crop generation
- Invalid / missing image handling
- Zero-face rejection
- Multiple-face rejection
- **11 / 11 Phase 1 tests passing**

### Phase 2 — Reverse image search ✅ **Complete**

- Genuine SerpApi integration (no mocked/hardcoded results in the live path)
- Google Lens reverse image search
- Local image uploaded through SerpApi's supported image-upload flow
- Real Google Lens search executed successfully against the test image
- Candidate URLs discovered **dynamically at runtime** — **not hardcoded**
- Raw response can be saved (credential-scrubbed) via `--debug-search`
- Proper API / network / rate-limit / error handling with distinct exit codes
- **19 Phase 2 mocked / unit tests** (HTTP layer mocked — no quota used)

**Total test suite: 94 tests passing** (Phase 1 + 2 + Phase 3 candidate
matching + Phase 4 fingerprint/blockchain). Phase 4 blockchain tests are fully
mocked — `pytest` never sends a transaction or needs a funded wallet.

### Phase 3 — Candidate face verification ✅ **Complete**

- Candidate image retrieval
- Candidate face detection + embedding
- Face similarity comparison (cosine) with match / uncertain / no-match status
- Best-match verification, structured output to `data/results/verification.json`

### Phase 4 — Blockchain anchoring ✅ **Complete**

- Stable **canonical evidence record** built from the Phase 3 result
  (title, source, source URL, candidate image URL, best similarity,
  verification status, input-image SHA-256) — mutable/raw API data excluded
- Deterministic serialization → **SHA-256 fingerprint of the canonical record**
  (not of the raw `verification.json`)
- Ethereum **Sepolia** transaction storing the fingerprint in calldata (web3.py)
- Transaction receipt wait + tx hash captured
- **True re-verification**: independently rebuilds the canonical evidence,
  recomputes the SHA-256, reads the on-chain fingerprint, and compares →
  `VERIFIED` / `TAMPERED`
- Re-verifiable proof persisted to `data/results/blockchain_proof.json`
  (no secrets)

### Phase 5 — Polish ⏳ **Not started**

- Final CLI / demo polish
- Final documentation / demo recording preparation

---

## 3. Architecture

Current (Phase 1 + Phase 2) data flow:

```
input image
  └─ (Phase 1) detect single face  ─ gate: exactly one consenting face
  │     └─ 512-d L2-normalized embedding + bbox + confidence + face crop
  └─ (Phase 2) upload image to SerpApi Image API  ──►  image_id
  │        POST https://serpapi.com/image   (multipart, ≤500 KB, field "image")
  └─ (Phase 2) Google Lens search by image_id     ──►  raw JSON
  │        GET https://serpapi.com/search?engine=google_lens&image_id=...
  └─ parse `visual_matches` (+ `exact_matches`)   ──►  list[Candidate]
           {position, title, url, source, domain, thumbnail_url, image_url, result_type}
```

**Why the two-step upload?** Google Lens needs an image it can fetch — either a
public URL or an `image_id` from SerpApi's own upload endpoint. We use the
**upload endpoint** so a *local* file never has to be hosted on an arbitrary
third-party service; the bytes go only to SerpApi. The image-submission strategy
lives in `pipeline/reverse_search.py` (`search_image(image_path=...)` uploads;
`search_image(image_url=...)` passes a public URL straight through).

**Phase 3 + 4 (implemented).** Phase 3 consumes the returned `Candidate`
objects — downloading publicly accessible candidate images, detecting and
embedding their faces, and comparing them against the input embedding to decide
whether the same person appears. Phase 4 builds a canonical evidence record from
that verified result, fingerprints it (SHA-256), anchors the fingerprint on the
Ethereum **Sepolia** test network, then re-reads and re-verifies it on-chain.

### Phase 4 proof flow

```
Phase 3 verification.json
  → build canonical evidence record (stable fields only)
  → deterministic JSON (sorted keys, compact) → SHA-256 fingerprint
  → Sepolia transaction (fingerprint in calldata) → wait for receipt → tx hash
  → read fingerprint back from the transaction
  → independently rebuild canonical evidence + recompute SHA-256
  → compare recomputed hash vs on-chain fingerprint
  → VERIFIED / TAMPERED   (proof saved to data/results/blockchain_proof.json)
```

Re-verify a saved proof later, without sending a new transaction:

```bash
./.venv/bin/python app.py --verify-proof data/results/blockchain_proof.json
```

---

## 4. Project structure

```
face-web-blockchain/
├── app.py                  # CLI entry point; Phase 1 + Phase 2 wired
├── config.py               # env-driven config (dotenv); no hardcoded secrets
├── requirements.txt
├── .env.example            # copy to .env (git-ignored)
├── pipeline/
│   ├── face_detector.py    # ✅ Phase 1: InsightFace detection + embedding
│   ├── reverse_search.py   # ✅ Phase 2: SerpApi Google Lens search
│   ├── candidate_matcher.py# ✅ Phase 3: candidate face verification
│   ├── fingerprint.py      # ✅ Phase 4: canonical evidence + SHA-256
│   └── blockchain.py       # ✅ Phase 4: Ethereum Sepolia record/retrieve
├── utils/
│   ├── logger.py           # stage/success/failure CLI helpers
│   └── http.py             # shared HTTP/session utilities
├── data/
│   ├── input/
│   ├── candidates/
│   └── results/
└── tests/
    ├── test_face_detector.py
    ├── test_reverse_search.py
    └── live_search.py
```

---

## 5. Requirements

- Python **3.11+**
- macOS / Linux, CPU is fine (ONNX Runtime CPU by default)
- A SerpApi API key for Phase 2 (free plan is enough — see below)

On first run InsightFace downloads its model pack (`buffalo_l`, ~300 MB) into
`~/.insightface`. This is a one-time download.

---

## 6. Setup

```bash
cd /Users/kriti_xr/face-web-blockchain

# 1. Create and activate a virtual environment (Python 3.11+)
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Configure environment (copy the template, then edit .env)
cp .env.example .env      # .env is git-ignored — never commit it
```

---

## 7. Environment variables

Copy `.env.example` to `.env` and fill in real values locally. `.env` is
git-ignored and must never be committed.

**Currently required (Phase 2):**

```env
SERPAPI_API_KEY=
SERPAPI_ENGINE=google_lens
```

The key is read via `config.py` / `python-dotenv`. It is **never printed** and
**never written** to `data/results/lens_response.json` (that file is scrubbed of
any `api_key` field before writing).

**Phase 4 (blockchain) — required for the on-chain step:**

```env
ETH_RPC_URL=            # e.g. https://sepolia.infura.io/v3/<project-id>
ETH_PRIVATE_KEY=        # throwaway test key ONLY — never a real-funds key
ETH_CHAIN_ID=11155111   # Sepolia test network
```

- **Sepolia is a test network.** Use a **throwaway** wallet funded only with
  Sepolia test ETH (from a public faucet) — never a real-funds key.
- The sender address is **derived from `ETH_PRIVATE_KEY`** via web3.py; there is
  no `WALLET_ADDRESS` variable.
- `ETH_CHAIN_ID` must be `11155111`; Phase 4 validates the connected chain and
  refuses to proceed on any other network.
- The private key is read via `config.py` / `python-dotenv`. It is **never
  printed**, never written to the proof file, and never committed.

Do **not** put any real API key or private key in this README, in commits, or in
any tracked file.

---

## 8. Running Phase 1 + Phase 2

Place a test image (one clear, consenting or public-domain face) at
`data/input/test.jpg`, then run:

```bash
cd /Users/kriti_xr/face-web-blockchain

./.venv/bin/python app.py --image data/input/test.jpg
```

To also save the sanitized raw SerpApi response for inspection:

```bash
./.venv/bin/python app.py --image data/input/test.jpg --debug-search
```

`--debug-search` writes the **sanitized (credential-scrubbed) raw SerpApi
response** to `data/results/lens_response.json`. That file can be large, so it is
not printed by default. Use `--max-show N` to change how many candidates print.

Other useful flags: `--crop-out PATH`, `--model buffalo_s`, `--det-size 640`,
`--det-threshold 0.5`.

Typical stages:

```
[1/3] Loading image
[2/3] Detecting face
✓ Face detected
  Bounding box (x1,y1,x2,y2): (...)
  Detector confidence: 0.9xxx
✓ Embedding generated
  Embedding: [+0.0123, ... dim=512]
  Face crop saved: data/results/test_face.jpg
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

### Face-validation rules (Phase 1)

- **0 faces →** rejected (`No face detected`, exit code 3)
- **>1 face →** rejected (`... detected N`, exit code 4) — keeps the demo to a
  single consenting subject
- exactly 1 face → returns bounding box + 512-d L2-normalized embedding, saves a
  face crop

### Phase 2 exit codes

| Symptom | Exit code | Fix |
|---------|:--------:|-----|
| `SERPAPI_API_KEY is not set` | 6 | Add the key to `.env`. |
| `SerpApi rejected the API key` | 7 | Key is wrong/expired — recopy from the dashboard. |
| `rate limit / quota exceeded` | 8 | Free monthly quota used up; wait or upgrade. |
| `Network/timeout during search` | 9 | Check connectivity; retry manually. |
| `Could not parse SerpApi response` | 10 | Transient API error / no results field — try `--debug-search`. |
| `Image problem for search` | 11 | Image missing/corrupt/too large to compress under 500 KB. |
| `Candidates found: 0` | 0 (success) | Genuinely no matches for that image — try a more public image. |

---

## 9. Testing

Run the full suite (94 tests):

```bash
./.venv/bin/python -m pytest -q
```

Expected: **94 passed**. Phase 4 blockchain tests are fully mocked, so
`pytest -q` **never sends a Sepolia transaction and needs no funded wallet**.

Phase 2 tests **mock the HTTP layer only** and never call the live API or use
quota. The fast unit tests need no model download. An integration test runs the
real model **only if** `data/input/test.jpg` exists and `insightface` is
installed; otherwise it is skipped.

For a genuine end-to-end check against the live API (needs a real key in `.env`
and consumes one search from your quota), run the opt-in live script explicitly:

```bash
./.venv/bin/python tests/live_search.py --image data/input/test.jpg
```

---

## 10. How the Google Lens search works

Phase 2 takes the validated input image and performs a genuine reverse image
search via the official [SerpApi Google Lens API](https://serpapi.com/google-lens-api)
— no manual Google scraping, no hardcoded or mocked results in the live path.

1. **Upload** the local image to SerpApi's Image API (`POST /image`, multipart,
   ≤500 KB) to get an `image_id`. This keeps a local file from having to be
   hosted on an arbitrary third-party service; the bytes go only to SerpApi.
2. **Search** Google Lens by `image_id`
   (`GET /search?engine=google_lens&image_id=...`) and receive raw JSON.
3. **Parse** `visual_matches` (and `exact_matches`) into `Candidate` objects
   (`position, title, url, source, domain, thumbnail_url, image_url,
   result_type`). Every candidate URL comes from this live response — **nothing
   is hardcoded**.

### Get and configure a SerpApi key

1. Create a free account at <https://serpapi.com/users/sign_up>.
2. Copy your key from <https://serpapi.com/manage-api-key>.
3. Put it in `.env` (git-ignored — never commit it):

```env
SERPAPI_API_KEY=
SERPAPI_ENGINE=google_lens
```

### Verified live run

In a verified live run against the current test image, the Google Lens search
returned **59 real candidates**, with sources such as **Wikipedia**, **NASA**,
and **Britannica** among the results.

These are **examples observed in that specific test run** — they are produced
dynamically by the live SerpApi Google Lens query and are **not hardcoded**. A
different image (or a later run) will return different candidates.

---

## 11. Current limitations

- **Discovery only.** Phase 2 discovers candidate web/search results. It does
  **not** verify that a discovered page contains the same person — that is
  Phase 3 (not started).
- **Phase 4 needs a funded Sepolia key for the live step.** The full on-chain
  write requires `ETH_RPC_URL` + a throwaway `ETH_PRIVATE_KEY` with Sepolia test
  ETH. Without it, Phases 1–3 still run; the blockchain step fails gracefully
  with a clear error and a non-zero exit code.
- **SerpApi free-plan quota.** The free plan allows roughly **100 searches per
  month**. Each `app.py` run performs **one** upload + **one** Lens search (which
  counts as one search against your quota). The client makes **no automatic
  retries** on rate-limit / 5xx errors, so a transient failure never silently
  burns extra quota — you see the error and decide.
- **Third-party dependency.** Uploading an image sends it to SerpApi → Google
  Lens. Results depend on Google Lens coverage for that image and can vary run to
  run.

---

## 12. Privacy / responsible-use note

- This is a **non-commercial hackathon / research demonstration**.
- Run it only against a **consenting participant or a public-domain / controlled
  test image**. Uploading an image sends it to a third-party API (SerpApi →
  Google Lens); do not submit images of non-consenting individuals.
- The system is **not** intended for mass identification, surveillance, or
  accessing private / login-protected accounts, and it does not scrape such
  accounts.
- `.env` is **git-ignored**; API keys and private keys must **never** be
  committed.
- Test / generated images and results are **git-ignored** (the input test image,
  generated face crops under `data/results`, downloaded candidate images under
  `data/candidates`, and generated search JSON such as
  `data/results/lens_response.json`).
- For the future Phase 4 blockchain work, use a **throwaway** Sepolia wallet
  funded with test ETH only — never a real-funds key.

---

## 13. Roadmap

| Phase | Description | State |
|------:|-------------|-------|
| **1** | Local face detection + 512-d embedding (InsightFace, CPU) | ✅ Complete |
| **2** | Reverse image search via SerpApi Google Lens (discovery) | ✅ Complete |
| **3** | Candidate retrieval, face detection/embedding, similarity match + ranking, best-match verification | ✅ Complete |
| **4** | Canonical evidence → SHA-256 fingerprint → Ethereum Sepolia record → on-chain re-verification | ✅ Complete |
| **5** | Final CLI / demo polish and documentation / demo recording | ⏳ Not started |
