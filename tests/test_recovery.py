"""Phase 4 transaction-recovery tests — fully mocked. No network, no tx sent."""

from __future__ import annotations

import json

import pytest
from web3.exceptions import TransactionNotFound, TimeExhausted

import app
from config import config as cfg
import pipeline.blockchain as bc
from pipeline.blockchain import fetch_tx_state, BlockchainError, SEPOLIA_CHAIN_ID
from pipeline.fingerprint import build_canonical_evidence, sha256_fingerprint


# ---------------------------------------------------------------------------
# Fakes for fetch_tx_state (read-only tx state)
# ---------------------------------------------------------------------------
class _EthState:
    def __init__(self, tx=None, tx_missing=False, receipt=None, receipt_missing=False):
        self._tx = tx
        self._tx_missing = tx_missing
        self._receipt = receipt
        self._receipt_missing = receipt_missing

    def get_transaction(self, tx_hash):
        if self._tx_missing:
            raise TransactionNotFound("missing")
        return self._tx

    def get_transaction_receipt(self, tx_hash):
        if self._receipt_missing:
            raise TransactionNotFound("missing")
        return self._receipt


class _W3State:
    def __init__(self, eth):
        self.eth = eth


def test_fetch_tx_state_mined_success():
    w3 = _W3State(_EthState(tx={"blockNumber": 11638627}, receipt={"status": 1}))
    st = fetch_tx_state("0xabc", w3)
    assert st == {"exists": True, "state": "MINED",
                  "block_number": 11638627, "receipt_status": 1}


def test_fetch_tx_state_pending():
    w3 = _W3State(_EthState(tx={"blockNumber": None}))
    st = fetch_tx_state("0xabc", w3)
    assert st["state"] == "PENDING"
    assert st["block_number"] is None
    assert st["receipt_status"] is None


def test_fetch_tx_state_missing():
    w3 = _W3State(_EthState(tx_missing=True))
    st = fetch_tx_state("0xabc", w3)
    assert st == {"exists": False, "state": "MISSING",
                  "block_number": None, "receipt_status": None}


def test_fetch_tx_state_failed_receipt():
    w3 = _W3State(_EthState(tx={"blockNumber": 99}, receipt={"status": 0}))
    st = fetch_tx_state("0xabc", w3)
    assert st["state"] == "MINED"
    assert st["receipt_status"] == 0


# ---------------------------------------------------------------------------
# Verification.json fixture + helpers
# ---------------------------------------------------------------------------
def _verification():
    return {
        "phase": 3,
        "status": "complete",
        "input_image": "/abs/data/input/photo.png",
        "input_image_hash": "d" * 64,
        "search": {"engine": "google_lens", "image_id": "img_1",
                   "image_url": "u", "search_id": "s", "candidate_count": 1},
        "verification": {
            "max_candidates": 10, "match_threshold": 0.40,
            "uncertain_threshold": 0.30,
            "results": [{
                "position": 1, "title": "Recovered Person",
                "source": "Wikipedia", "domain": "en.wikipedia.org",
                "page_url": "https://en.wikipedia.org/wiki/R",
                "image_url": "https://upload/r.jpg",
                "image_path": "/tmp/c/1.jpg", "status": "match",
                "face_count": 1, "best_similarity": 0.6543210,
                "best_face_index": 0, "error": None,
            }],
        },
        "next_phase": {"fingerprint": True, "blockchain": True},
    }


def _write_verification(tmp_path):
    p = tmp_path / "verification.json"
    p.write_text(json.dumps(_verification()), encoding="utf-8")
    return p


def _expected_fingerprint():
    return sha256_fingerprint(build_canonical_evidence(_verification()))


@pytest.fixture
def isolate_results(monkeypatch, tmp_path):
    """Keep proof writes inside tmp_path so real results are untouched."""
    monkeypatch.setattr(cfg, "results_dir", tmp_path)
    return tmp_path


@pytest.fixture
def stub_web3(monkeypatch):
    """_recover_tx / _run_phase4 need get_web3 to not touch the network."""
    monkeypatch.setattr(app, "get_web3", lambda: object())


# ---------------------------------------------------------------------------
# _recover_tx — end-to-end recovery states
# ---------------------------------------------------------------------------
def test_recover_verified(monkeypatch, tmp_path, isolate_results, stub_web3):
    vpath = _write_verification(tmp_path)
    fp = _expected_fingerprint()

    monkeypatch.setattr(app, "fetch_tx_state", lambda h, w3=None: {
        "exists": True, "state": "MINED", "block_number": 12345,
        "receipt_status": 1})
    monkeypatch.setattr(app, "get_onchain_fingerprint", lambda h, w3=None: fp)

    rc = app._recover_tx("0xdeadbeef", vpath)
    assert rc == 0

    proof = json.loads((tmp_path / "blockchain_proof.json").read_text())
    assert proof["verification_status"] == "VERIFIED"
    assert proof["tx_hash"] == "0xdeadbeef"
    assert proof["block_number"] == 12345
    # fingerprint stored is the INDEPENDENTLY recomputed one.
    assert proof["fingerprint"] == fp


def test_recover_tampered(monkeypatch, tmp_path, isolate_results, stub_web3):
    vpath = _write_verification(tmp_path)
    monkeypatch.setattr(app, "fetch_tx_state", lambda h, w3=None: {
        "exists": True, "state": "MINED", "block_number": 1,
        "receipt_status": 1})
    # On-chain fingerprint differs from the recomputed one.
    monkeypatch.setattr(app, "get_onchain_fingerprint", lambda h, w3=None: "f" * 64)

    rc = app._recover_tx("0xabc", vpath)
    assert rc == 21
    proof = json.loads((tmp_path / "blockchain_proof.json").read_text())
    assert proof["verification_status"] == "TAMPERED"


def test_recover_missing(monkeypatch, tmp_path, isolate_results, stub_web3):
    vpath = _write_verification(tmp_path)
    monkeypatch.setattr(app, "fetch_tx_state", lambda h, w3=None: {
        "exists": False, "state": "MISSING", "block_number": None,
        "receipt_status": None})
    assert app._recover_tx("0xabc", vpath) == 23
    assert not (tmp_path / "blockchain_proof.json").exists()


def test_recover_pending(monkeypatch, tmp_path, isolate_results, stub_web3):
    vpath = _write_verification(tmp_path)
    monkeypatch.setattr(app, "fetch_tx_state", lambda h, w3=None: {
        "exists": True, "state": "PENDING", "block_number": None,
        "receipt_status": None})
    assert app._recover_tx("0xabc", vpath) == 24
    assert not (tmp_path / "blockchain_proof.json").exists()


def test_recover_failed_receipt(monkeypatch, tmp_path, isolate_results, stub_web3):
    vpath = _write_verification(tmp_path)
    monkeypatch.setattr(app, "fetch_tx_state", lambda h, w3=None: {
        "exists": True, "state": "MINED", "block_number": 7,
        "receipt_status": 0})
    assert app._recover_tx("0xabc", vpath) == 25
    assert not (tmp_path / "blockchain_proof.json").exists()


# ---------------------------------------------------------------------------
# Timeout after submission MUST NOT trigger a second transaction
# ---------------------------------------------------------------------------
class _EthTimeout:
    def wait_for_transaction_receipt(self, tx_hash):
        raise TimeExhausted("timed out waiting for receipt")


class _W3Timeout:
    eth = _EthTimeout()


def test_timeout_after_submission_does_not_resubmit(monkeypatch, tmp_path):
    vpath = _write_verification(tmp_path)

    calls = {"n": 0}

    def fake_record(fingerprint):
        calls["n"] += 1
        return "0xSUBMITTEDONCE"

    monkeypatch.setattr(app, "record_fingerprint", fake_record)
    monkeypatch.setattr(app, "get_web3", lambda: _W3Timeout())

    rc = app._run_phase4(vpath)

    assert rc == 26                      # recoverable timeout exit code
    assert calls["n"] == 1               # submitted exactly once — no resubmit
    # No proof written on a timeout (nothing verified yet).
    assert not (tmp_path / "blockchain_proof.json").exists()


# ---------------------------------------------------------------------------
# Secret safety: private key never appears in recovery output/errors
# ---------------------------------------------------------------------------
def test_no_secret_in_recovery_output(monkeypatch, tmp_path, isolate_results, capsys):
    secret = "0x" + "ab" * 32
    monkeypatch.setattr(cfg, "eth_private_key", secret)
    vpath = _write_verification(tmp_path)

    # Force a blockchain read error to exercise the error path.
    monkeypatch.setattr(app, "get_web3", lambda: object())

    def boom(h, w3=None):
        raise BlockchainError("Could not fetch transaction 0xabc: HTTPError")

    monkeypatch.setattr(app, "fetch_tx_state", boom)

    rc = app._recover_tx("0xabc", vpath)
    assert rc == 20
    out = capsys.readouterr()
    assert secret not in (out.out + out.err)
