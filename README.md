# Face → Web Search → Blockchain Verification

**Phase 1 ✅ | Phase 2 ✅ | Phase 3 ✅ | Phase 4 ✅ | Phase 5 ✅**

A **non-commercial hackathon / research demonstration** that connects local face analysis, reverse image search, candidate verification, cryptographic fingerprinting, and blockchain anchoring into one reproducible CLI pipeline.

> ⚠️ **Responsible use**
>
> Use this project only with a **consenting participant or a public-domain / controlled test image**. It is not intended for mass identification, surveillance, scraping private or login-protected accounts, or building dossiers about people.
>
> The system processes publicly accessible web results only. Candidate images must be publicly accessible.

---

## 1. What this project demonstrates

Given one input image containing exactly one face, the pipeline:

1. Detects the face locally with **InsightFace**.
2. Generates a **512-dimensional L2-normalized embedding**.
3. Sends the image to **SerpApi → Google Lens** for genuine reverse image search.
4. Retrieves publicly accessible candidate images discovered dynamically.
5. Detects and embeds faces in those candidates.
6. Compares candidate embeddings using cosine similarity.
7. Selects and records the verification evidence.
8. Builds a stable canonical evidence record.
9. Computes a deterministic **SHA-256 fingerprint**.
10. Stores that fingerprint in an **Ethereum Sepolia** transaction.
11. Reads the fingerprint back from the blockchain.
12. Independently recomputes the fingerprint and reports **VERIFIED** or **TAMPERED**.

### Pipeline

```text
Input image
    │
    ▼
[1] InsightFace
    │  single-face validation
    │  512-d embedding
    ▼
[2] Google Lens / SerpApi
    │  dynamic web candidates
    ▼
[3] Candidate verification
    │  download public images
    │  face detection + embeddings
    │  cosine similarity
    ▼
verification.json
    │
    ▼
[4] Canonical evidence
    │
    ▼
SHA-256 fingerprint
    │
    ▼
Ethereum Sepolia
    │
    ▼
[5] Read back + independently recompute
    │
    ▼
VERIFIED / TAMPERED
    │
    ▼
blockchain_proof.json
```

The blockchain does **not** store the face image or private data. It stores the cryptographic fingerprint of the canonical evidence.

---

## 2. Phase status

| Phase | Description                                                       | Status |
| ----: | ----------------------------------------------------------------- | :----: |
|     1 | Local face detection + 512-d embedding                            |    ✅   |
|     2 | Genuine SerpApi Google Lens reverse search                        |    ✅   |
|     3 | Candidate retrieval + face similarity verification                |    ✅   |
|     4 | Canonical evidence → SHA-256 → Ethereum Sepolia → re-verification |    ✅   |
|     5 | CLI polish + documentation + demo preparation                     |    ✅   |

### Test status

```text
105 passed, 1 warning
```

The warning comes from a deprecated InsightFace/face-alignment API used by the installed dependency. It does not cause a test failure.

Blockchain tests are mocked; the normal test suite does **not** send a transaction or require a funded wallet.

---

## 3. Project structure

```text
face-web-blockchain/
├── app.py                    # CLI entry point
├── config.py                 # environment-driven configuration
├── requirements.txt
├── .env.example              # environment template
│
├── pipeline/
│   ├── face_detector.py      # Phase 1: InsightFace
│   ├── reverse_search.py     # Phase 2: SerpApi / Google Lens
│   ├── candidate_matcher.py  # Phase 3: candidate verification
│   ├── fingerprint.py        # Phase 4: canonical evidence + SHA-256
│   └── blockchain.py         # Phase 4: Ethereum Sepolia
│
├── utils/
│   ├── logger.py
│   └── http.py
│
├── data/
│   ├── input/
│   ├── candidates/
│   └── results/
│
└── tests/
    ├── test_face_detector.py
    ├── test_reverse_search.py
    └── live_search.py
```

Generated images and result files are intended to remain local and are git-ignored.

---

## 4. Requirements

* Python **3.11+**
* macOS or Linux
* CPU is sufficient
* SerpApi API key
* Sepolia RPC endpoint for the blockchain phase
* Throwaway Sepolia wallet containing test ETH

InsightFace downloads its model pack (`buffalo_l`) on first initialization. The model is cached locally afterward.

---

## 5. Setup

```bash
git clone <repository-url>
cd blockchain-

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
```

Edit `.env` with your local credentials.

**Never commit `.env`, API keys, or private keys.**

---

## 6. Environment configuration

### SerpApi

```env
SERPAPI_API_KEY=
SERPAPI_ENGINE=google_lens
```

The key is loaded through `python-dotenv`.

It is not printed by the application and sanitized debug output removes credential fields.

### Ethereum Sepolia

```env
ETH_RPC_URL=
ETH_PRIVATE_KEY=
ETH_CHAIN_ID=11155111
```

`ETH_PRIVATE_KEY` should belong to a **throwaway test wallet only**.

The application derives the sender address from the private key. There is no separate wallet-address configuration.

The application validates the configured chain and expects:

```text
11155111 = Ethereum Sepolia
```

Never use a real-funds private key for this project.

---

## 7. Run the complete pipeline

Place a suitable test image in:

```text
data/input/test.jpg
```

The image should contain **exactly one clear face**.

Run:

```bash
./.venv/bin/python app.py --image data/input/test.jpg
```

The CLI progresses through five stages:

```text
[1/5] Loading image
[2/5] Detecting face
[3/5] Searching web with Google Lens
[4/5] Verifying candidate faces
[5/5] Fingerprint + blockchain
```

The final blockchain stage performs:

```text
canonical evidence
    ↓
SHA-256 fingerprint
    ↓
Sepolia transaction
    ↓
transaction confirmation
    ↓
on-chain fingerprint readback
    ↓
independent recomputation
    ↓
VERIFIED
```

A successful run writes:

```text
data/results/verification.json
data/results/blockchain_proof.json
```

and normally also creates:

```text
data/results/<input>_face.jpg
data/candidates/*
```

---

## 8. Useful CLI options

### Show more Google Lens candidates

```bash
./.venv/bin/python app.py \
  --image data/input/test.jpg \
  --max-show 20
```

### Verify more candidates

```bash
./.venv/bin/python app.py \
  --image data/input/test.jpg \
  --max-candidates 20
```

### Save sanitized SerpApi debug output

```bash
./.venv/bin/python app.py \
  --image data/input/test.jpg \
  --debug-search
```

This writes:

```text
data/results/lens_response.json
```

The saved response is credential-scrubbed.

### Configure matching thresholds

```bash
./.venv/bin/python app.py \
  --image data/input/test.jpg \
  --match-threshold 0.40 \
  --uncertain-threshold 0.30
```

The thresholds used for the run are recorded in `verification.json`.

These thresholds are **demo configuration values, not scientifically validated biometric identity thresholds**.

### Change the face detector settings

```bash
./.venv/bin/python app.py \
  --image data/input/test.jpg \
  --det-size 640 \
  --det-threshold 0.5
```

---

## 9. Phase 4 recovery and re-verification

The blockchain workflow is deliberately designed to avoid duplicate transactions.

### Recover an already-submitted transaction

If the transaction was submitted but the program timed out while waiting for confirmation, **do not submit another transaction**.

Use:

```bash
./.venv/bin/python app.py \
  --recover-tx <TX_HASH>
```

This is read-only.

It:

1. Fetches the transaction state.
2. Checks whether it was mined successfully.
3. Reads the fingerprint from the transaction.
4. Rebuilds canonical evidence from the current `verification.json`.
5. Recomputes SHA-256.
6. Compares local and on-chain fingerprints.
7. Writes `blockchain_proof.json`.

You can provide a different verification file with:

```bash
./.venv/bin/python app.py \
  --recover-tx <TX_HASH> \
  --verification path/to/verification.json
```

### Re-verify an existing proof

Once `blockchain_proof.json` exists:

```bash
./.venv/bin/python app.py \
  --verify-proof data/results/blockchain_proof.json
```

This sends **no new transaction**.

The expected successful result is:

```text
✓ BLOCKCHAIN VERIFICATION: VERIFIED
```

If the recomputed fingerprint differs from the on-chain fingerprint:

```text
✗ BLOCKCHAIN VERIFICATION: TAMPERED
```

---

## 10. How the blockchain proof works

Phase 4 does **not** hash the raw `verification.json` directly.

Instead, it builds a small canonical evidence record containing stable fields such as:

* candidate title
* source
* source URL
* candidate image URL
* best similarity
* verification status
* original input-image SHA-256

Mutable or environment-specific information is intentionally excluded, including:

* local filesystem paths
* raw API responses
* SerpApi image/search IDs
* debug data
* timestamps

The canonical record is serialized deterministically using sorted JSON keys and compact separators.

```text
canonical evidence
        │
        ▼
deterministic JSON
        │
        ▼
SHA-256
        │
        ▼
fingerprint
        │
        ▼
Ethereum Sepolia transaction calldata
```

During verification, the application independently rebuilds the evidence and compares the resulting hash with the fingerprint retrieved from the blockchain.

That is what produces the final:

```text
VERIFIED
```

rather than merely trusting a locally stored status field.

---

## 11. Output artifacts

### `verification.json`

Phase 3 structured verification output.

Contains:

* input image metadata
* input image SHA-256
* search metadata
* candidate verification results
* similarity values
* match thresholds

### `blockchain_proof.json`

Phase 4 re-verifiable proof.

Contains:

* network
* chain ID
* transaction hash
* block number
* fingerprint
* canonical evidence
* verification status

No private key or API key is written to the proof.

### Face crop

```text
data/results/<input>_face.jpg
```

A locally generated crop of the single detected face.

### Candidate images

```text
data/candidates/
```

Downloaded public candidate images used during Phase 3 verification.

---

## 12. Google Lens integration

Phase 2 uses the official SerpApi Google Lens flow.

For a local image:

```text
local image
    ↓
SerpApi image upload
    ↓
image_id
    ↓
Google Lens search
    ↓
visual/exact matches
    ↓
Candidate objects
```

Candidate URLs are discovered dynamically from the live response. They are not hardcoded.

The application intentionally does not implement manual Google scraping.

SerpApi requests are also not automatically retried after rate-limit or server errors because unnecessary retries could consume metered search quota.

---

## 13. Error handling

The CLI uses distinct exit codes for major failure modes.

Important examples:

| Code | Meaning                                          |
| ---: | ------------------------------------------------ |
|    3 | No face detected                                 |
|    4 | Multiple faces detected                          |
|    6 | Missing SerpApi key                              |
|    7 | Invalid SerpApi key                              |
|    8 | SerpApi rate limit/quota                         |
|   12 | Other reverse-search failure                     |
|   16 | Candidate verification failure                   |
|   18 | Blockchain recording failure                     |
|   20 | Blockchain read/verification failure             |
|   21 | Fingerprint mismatch / `TAMPERED`                |
|   22 | Missing/malformed proof or verification evidence |
|   23 | Recovery transaction not found                   |
|   24 | Recovery transaction still pending               |
|   25 | Recovery transaction mined but failed            |
|   26 | Transaction submitted but receipt wait timed out |

The timeout recovery path preserves the submitted transaction hash and explicitly avoids resubmitting it.

---

## 14. Testing

Run the complete test suite:

```bash
python -m pytest -q
```

Current result:

```text
105 passed, 1 warning
```

The tests cover the face-detection, reverse-search, candidate-matching, fingerprint, and blockchain logic.

Blockchain unit tests are mocked and therefore do not send live Ethereum transactions.

---

## 15. Demo walkthrough

For a hackathon presentation, use this sequence.

### 1. Show the architecture

```text
Face
  ↓
Google Lens
  ↓
Candidate verification
  ↓
SHA-256
  ↓
Sepolia
  ↓
Independent verification
```

### 2. Run the pipeline

```bash
./.venv/bin/python app.py --image data/input/test.jpg
```

Point out:

* exactly one face is required
* embedding is generated locally
* Google Lens returns live candidates
* candidate faces are compared
* the best evidence is fingerprinted

### 3. Highlight the fingerprint

Show:

```text
SHA-256: <fingerprint>
```

Explain that this is the hash of the **canonical evidence**, not the raw API response.

### 4. Show the blockchain transaction

Show the transaction hash and confirmation block.

Explain:

> The blockchain stores the fingerprint, not the image or private information.

### 5. Demonstrate independent verification

Run:

```bash
./.venv/bin/python app.py \
  --verify-proof data/results/blockchain_proof.json
```

Show:

```text
Recomputed SHA-256:  ...
On-chain fingerprint: ...
✓ BLOCKCHAIN VERIFICATION: VERIFIED
```

### 6. Explain the security property

The important demo moment is:

```text
local evidence
      ↓
recompute hash
      ↓
compare with immutable on-chain fingerprint
      ↓
VERIFIED / TAMPERED
```

The verification step does not simply trust the stored proof's fingerprint field.

---

## 16. Privacy and responsible use

This project is intentionally scoped as a **non-commercial hackathon / research demonstration**.

Use only:

* a consenting participant, or
* a public-domain / controlled test image.

Do not use it for:

* mass identification
* surveillance
* tracking people
* scraping private or login-protected accounts
* building dossiers about individuals

The image submitted for reverse search is sent to a third-party API service (SerpApi → Google Lens). Do not upload images of people who have not consented to this processing.

The blockchain proof contains cryptographic evidence metadata, not the original face image.

---

## 17. Security notes

`.env` must remain untracked.

Never commit:

```text
SERPAPI_API_KEY
ETH_PRIVATE_KEY
```

For blockchain demonstrations, use a wallet containing **Sepolia test ETH only**.

If a private key has ever been exposed outside the intended local environment, treat that wallet as compromised and replace it with a fresh throwaway test wallet before public demonstration.

---

## 18. Known limitations

* Face similarity thresholds are configurable demo values and are **not scientifically validated identity thresholds**.
* Google Lens results depend on Google's coverage and may change between runs.
* SerpApi usage consumes API quota.
* Candidate verification only processes publicly accessible candidate images.
* The MVP intentionally requires exactly one face in the input image.
* The blockchain phase requires an available Sepolia RPC endpoint and test ETH for the sending wallet.
* A successful blockchain transaction proves that the fingerprint was anchored on-chain; it does not independently establish that the underlying face-match conclusion is objectively true.

---

## 19. Roadmap

The original five-phase implementation is complete:

```text
Phase 1  Local face detection + embedding       ✅
Phase 2  Reverse image search                   ✅
Phase 3  Candidate face verification            ✅
Phase 4  Blockchain fingerprint + verification  ✅
Phase 5  CLI + documentation + demo polish      ✅
```

Future work could focus on stronger evaluation datasets, threshold calibration, richer evidence provenance, and additional non-biometric verification signals while preserving the project's responsible-use constraints.
