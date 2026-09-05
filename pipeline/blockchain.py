"""Phase 4 — record and retrieve fingerprints on Ethereum Sepolia.

Configuration comes from the shared ``config`` object (config.py), which reads
``ETH_RPC_URL`` / ``ETH_PRIVATE_KEY`` / ``ETH_CHAIN_ID`` from the environment /
``.env``. We never duplicate env parsing here and never print the private key.

The fingerprint is stored in transaction *calldata* of a zero-value
self-transaction (no smart contract required). Retrieval reads that calldata
back and decodes it.
"""
from __future__ import annotations
from typing import Optional
from web3 import Web3
from web3.exceptions import TransactionNotFound
from config import config
# Ethereum Sepolia test network.
SEPOLIA_CHAIN_ID = 11155111
class BlockchainError(RuntimeError):
    """Any Phase 4 blockchain/config/network failure (secret-safe message)."""
def _require_rpc_url() -> str:
    rpc_url = config.eth_rpc_url
    if not rpc_url:
        raise BlockchainError("ETH_RPC_URL is not set (add it to .env)")
    return rpc_url
def _require_private_key() -> str:
    key = config.eth_private_key
    if not key:
        raise BlockchainError("ETH_PRIVATE_KEY is not set (add it to .env)")
    return key
def get_web3() -> Web3:
    """Connect to the configured RPC and validate it is Sepolia.
    Raises :class:`BlockchainError` on missing config, failed connection, or a
    chain id that is not Sepolia (11155111).
    """
    rpc_url = _require_rpc_url()
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    try:
        connected = w3.is_connected()
    except Exception as e:  # network/DNS/etc. — never leak the URL contents.
        raise BlockchainError(f"Could not reach ETH_RPC_URL: {type(e).__name__}") from e
    if not connected:
        raise BlockchainError("Could not connect to the Ethereum RPC endpoint")
    expected = config.eth_chain_id
    if expected != SEPOLIA_CHAIN_ID:
        raise BlockchainError(
            f"ETH_CHAIN_ID must be {SEPOLIA_CHAIN_ID} (Sepolia); got {expected}"
        )
    try:
        actual = w3.eth.chain_id
    except Exception as e:
        raise BlockchainError(f"Could not read chain id: {type(e).__name__}") from e
    if actual != SEPOLIA_CHAIN_ID:
        raise BlockchainError(
            f"Wrong network. Expected Sepolia ({SEPOLIA_CHAIN_ID}), got {actual}"
        )
    return w3


def get_account(w3: Web3):
    """Derive the sending account from ETH_PRIVATE_KEY.

    The public address is derived via web3.py; ``WALLET_ADDRESS`` is not needed.
    An invalid key raises a secret-safe :class:`BlockchainError`.
    """
    key = _require_private_key()
    try:
        return w3.eth.account.from_key(key)
    except Exception as e:
        # Do NOT include the key or its contents in the message.
        raise BlockchainError(f"Invalid ETH_PRIVATE_KEY ({type(e).__name__})") from e


def record_fingerprint(fingerprint: str) -> str:
    """Store ``fingerprint`` in Sepolia transaction calldata; return the tx hash.

    Sends a zero-value transaction from the derived account to itself with the
    fingerprint bytes in the ``data`` field.
    """
    if not fingerprint or not isinstance(fingerprint, str):
        raise BlockchainError("Refusing to record an empty/invalid fingerprint")

    w3 = get_web3()
    account = get_account(w3)

    try:
        nonce = w3.eth.get_transaction_count(account.address)
        data = w3.to_bytes(text=fingerprint)
        transaction = {
            "from": account.address,
            "to": account.address,
            "value": 0,
            "nonce": nonce,
            "chainId": SEPOLIA_CHAIN_ID,
            "gas": 100000,
            "gasPrice": w3.eth.gas_price,
            "data": data,
        }
        signed = w3.eth.account.sign_transaction(transaction, account.key)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    except BlockchainError:
        raise
    except Exception as e:
        raise BlockchainError(f"Failed to send transaction: {type(e).__name__}") from e
    return tx_hash.hex()
def get_onchain_fingerprint(tx_hash: str, w3: Optional[Web3] = None) -> str:
    """Retrieve and decode the fingerprint stored in a transaction's calldata."""
    if not tx_hash or not isinstance(tx_hash, str):
        raise BlockchainError("A transaction hash is required")
    if w3 is None:
        w3 = get_web3()
    try:
        transaction = w3.eth.get_transaction(tx_hash)
    except Exception as e:
        raise BlockchainError(
            f"Could not fetch transaction {tx_hash}: {type(e).__name__}"
        ) from e
    if transaction is None:
        raise BlockchainError(f"Transaction not found: {tx_hash}")
    data = transaction["input"]
    try:
        if isinstance(data, (bytes, bytearray)):
            raw = bytes(data)
        else:
            text = str(data)
            raw = bytes.fromhex(text[2:] if text.startswith("0x") else text)
        return raw.decode("utf-8")
    except Exception as e:
        raise BlockchainError(
            f"Malformed on-chain calldata: {type(e).__name__}"
        ) from e
def fetch_tx_state(tx_hash: str, w3: Optional[Web3] = None) -> dict:
    """READ-ONLY: report the state of an already-submitted transaction.

    Never submits anything. Returns a dict:
        {
          "exists": bool,
          "state": "MISSING" | "PENDING" | "MINED",
          "block_number": int | None,
          "receipt_status": 1 | 0 | None,   # None while pending / missing
        }

    Used by the recovery path so a timed-out (but already-submitted) transaction
    can be checked later without sending a replacement.
    """
    if not tx_hash or not isinstance(tx_hash, str):
        raise BlockchainError("A transaction hash is required")

    if w3 is None:
        w3 = get_web3()

    try:
        tx = w3.eth.get_transaction(tx_hash)
    except TransactionNotFound:
        return {"exists": False, "state": "MISSING", "block_number": None,
                "receipt_status": None}
    except Exception as e:
        raise BlockchainError(
            f"Could not fetch transaction {tx_hash}: {type(e).__name__}"
        ) from e

    if tx is None:
        return {"exists": False, "state": "MISSING", "block_number": None,
                "receipt_status": None}

    block_number = tx["blockNumber"]
    if block_number is None:
        return {"exists": True, "state": "PENDING", "block_number": None,
                "receipt_status": None}

    receipt_status: Optional[int] = None
    try:
        receipt = w3.eth.get_transaction_receipt(tx_hash)
        if receipt is not None:
            receipt_status = receipt["status"]
    except TransactionNotFound:
        receipt_status = None
    except Exception as e:
        raise BlockchainError(
            f"Could not fetch receipt for {tx_hash}: {type(e).__name__}"
        ) from e

    return {"exists": True, "state": "MINED", "block_number": block_number,
            "receipt_status": receipt_status}


def retrieve_and_verify(tx_hash: str, expected_fingerprint: str) -> str:
    """Compare the on-chain fingerprint against ``expected_fingerprint``.
    Returns ``"VERIFIED"`` on match, ``"TAMPERED"`` otherwise. This is the
    low-level compare; true re-verification (rebuild + recompute) lives in the
    caller so the expected hash is recomputed from evidence, not trusted blindly.
    """
    stored = get_onchain_fingerprint(tx_hash)
    return "VERIFIED" if stored == expected_fingerprint else "TAMPERED"