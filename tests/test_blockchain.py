"""Phase 4 blockchain tests — fully mocked. These NEVER touch the network,
never send a transaction, and never require a funded wallet.
"""

from __future__ import annotations

import pytest

from config import config as cfg
import pipeline.blockchain as bc
from pipeline.blockchain import (
    BlockchainError,
    SEPOLIA_CHAIN_ID,
    get_web3,
    get_account,
    record_fingerprint,
    get_onchain_fingerprint,
    retrieve_and_verify,
)


# ---------------------------------------------------------------------------
# Fakes standing in for web3.py — no real RPC/network.
# ---------------------------------------------------------------------------
class _FakeSigned:
    raw_transaction = b"\xde\xad\xbe\xef"


class _FakeTxHash:
    def __init__(self, value: str):
        self._value = value

    def hex(self) -> str:
        return self._value


class _FakeAccount:
    def __init__(self, address="0x000000000000000000000000000000000000dEaD", raises=False):
        self.address = address
        self.key = b"\x01" * 32
        self._raises = raises

    def from_key(self, key):
        if self._raises:
            raise ValueError("bad key")
        return self

    def sign_transaction(self, tx, key):
        return _FakeSigned()


class _FakeEth:
    def __init__(self, chain_id=SEPOLIA_CHAIN_ID, tx_input=None, account=None):
        self.chain_id = chain_id
        self.gas_price = 1_000_000_000
        self.account = account or _FakeAccount()
        self._tx_input = tx_input

    def get_transaction_count(self, address):
        return 0

    def send_raw_transaction(self, raw):
        return _FakeTxHash("0xabc123")

    def get_transaction(self, tx_hash):
        if self._tx_input is None:
            return None
        return {"input": self._tx_input}

    def wait_for_transaction_receipt(self, tx_hash):
        return {"blockNumber": 42}


class FakeWeb3:
    """Minimal stand-in for web3.Web3 as used by pipeline.blockchain."""

    _connected = True
    _chain_id = SEPOLIA_CHAIN_ID
    _tx_input = None
    _account = None

    def __init__(self, provider):
        self.provider = provider
        self.eth = _FakeEth(
            chain_id=type(self)._chain_id,
            tx_input=type(self)._tx_input,
            account=type(self)._account,
        )

    def is_connected(self):
        return type(self)._connected

    @staticmethod
    def HTTPProvider(url):
        return ("http", url)

    @staticmethod
    def to_bytes(text=None):
        return text.encode("utf-8")


def _make_fake_web3(*, connected=True, chain_id=SEPOLIA_CHAIN_ID, tx_input=None, account=None):
    return type(
        "FakeWeb3Variant",
        (FakeWeb3,),
        {
            "_connected": connected,
            "_chain_id": chain_id,
            "_tx_input": tx_input,
            "_account": account,
        },
    )


@pytest.fixture
def good_env(monkeypatch):
    monkeypatch.setattr(cfg, "eth_rpc_url", "https://sepolia.example/rpc")
    monkeypatch.setattr(cfg, "eth_private_key", "0x" + "11" * 32)
    monkeypatch.setattr(cfg, "eth_chain_id", SEPOLIA_CHAIN_ID)


# ---------------------------------------------------------------------------
# get_web3 — config + network validation
# ---------------------------------------------------------------------------
def test_missing_rpc_url_raises(monkeypatch, good_env):
    monkeypatch.setattr(cfg, "eth_rpc_url", "")
    with pytest.raises(BlockchainError, match="ETH_RPC_URL"):
        get_web3()


def test_not_connected_raises(monkeypatch, good_env):
    monkeypatch.setattr(bc, "Web3", _make_fake_web3(connected=False))
    with pytest.raises(BlockchainError, match="Could not connect"):
        get_web3()


def test_config_chain_id_must_be_sepolia(monkeypatch, good_env):
    monkeypatch.setattr(cfg, "eth_chain_id", 1)
    monkeypatch.setattr(bc, "Web3", _make_fake_web3())
    with pytest.raises(BlockchainError, match="ETH_CHAIN_ID must be"):
        get_web3()


def test_wrong_network_chain_id_raises(monkeypatch, good_env):
    monkeypatch.setattr(bc, "Web3", _make_fake_web3(chain_id=1))
    with pytest.raises(BlockchainError, match="Wrong network"):
        get_web3()


def test_get_web3_ok(monkeypatch, good_env):
    monkeypatch.setattr(bc, "Web3", _make_fake_web3())
    w3 = get_web3()
    assert w3.eth.chain_id == SEPOLIA_CHAIN_ID


# ---------------------------------------------------------------------------
# get_account — derive address, never require WALLET_ADDRESS
# ---------------------------------------------------------------------------
def test_missing_private_key_raises(monkeypatch, good_env):
    monkeypatch.setattr(cfg, "eth_private_key", "")
    w3 = _make_fake_web3()(("http", "x"))
    with pytest.raises(BlockchainError, match="ETH_PRIVATE_KEY"):
        get_account(w3)


def test_invalid_private_key_raises(monkeypatch, good_env):
    account = _FakeAccount(raises=True)
    w3 = _make_fake_web3(account=account)(("http", "x"))
    with pytest.raises(BlockchainError, match="Invalid ETH_PRIVATE_KEY"):
        get_account(w3)


def test_invalid_key_error_does_not_leak_secret(monkeypatch, good_env):
    secret = "0x" + "ab" * 32
    monkeypatch.setattr(cfg, "eth_private_key", secret)
    account = _FakeAccount(raises=True)
    w3 = _make_fake_web3(account=account)(("http", "x"))
    with pytest.raises(BlockchainError) as ei:
        get_account(w3)
    assert secret not in str(ei.value)


# ---------------------------------------------------------------------------
# record_fingerprint — signs + sends via the fake, returns tx hash
# ---------------------------------------------------------------------------
def test_record_fingerprint_rejects_empty(good_env):
    with pytest.raises(BlockchainError, match="empty/invalid"):
        record_fingerprint("")


def test_record_fingerprint_returns_tx_hash(monkeypatch, good_env):
    monkeypatch.setattr(bc, "Web3", _make_fake_web3())
    tx = record_fingerprint("f" * 64)
    assert tx == "0xabc123"


# ---------------------------------------------------------------------------
# get_onchain_fingerprint / retrieve_and_verify — decode + compare
# ---------------------------------------------------------------------------
def test_get_onchain_fingerprint_decodes_bytes(good_env):
    fp = "a" * 64
    w3 = _make_fake_web3(tx_input=fp.encode("utf-8"))(("http", "x"))
    assert get_onchain_fingerprint("0xabc", w3=w3) == fp


def test_get_onchain_fingerprint_decodes_hex_string(good_env):
    fp = "b" * 64
    hexstr = "0x" + fp.encode("utf-8").hex()
    w3 = _make_fake_web3(tx_input=hexstr)(("http", "x"))
    assert get_onchain_fingerprint("0xabc", w3=w3) == fp


def test_get_onchain_fingerprint_missing_tx_raises(good_env):
    w3 = _make_fake_web3(tx_input=None)(("http", "x"))
    with pytest.raises(BlockchainError, match="not found"):
        get_onchain_fingerprint("0xabc", w3=w3)


def test_retrieve_and_verify_matching_is_verified(monkeypatch, good_env):
    fp = "c" * 64
    monkeypatch.setattr(bc, "Web3", _make_fake_web3(tx_input=fp.encode("utf-8")))
    assert retrieve_and_verify("0xabc", fp) == "VERIFIED"


def test_retrieve_and_verify_mismatch_is_tampered(monkeypatch, good_env):
    stored = "c" * 64
    monkeypatch.setattr(bc, "Web3", _make_fake_web3(tx_input=stored.encode("utf-8")))
    assert retrieve_and_verify("0xabc", "d" * 64) == "TAMPERED"
