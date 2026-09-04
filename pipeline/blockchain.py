"""Record and retrieve fingerprints on Ethereum Sepolia."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from web3 import Web3


load_dotenv()


def get_web3() -> Web3:
    """Connect to the configured Sepolia RPC."""

    rpc_url = os.getenv("SEPOLIA_RPC_URL")

    if not rpc_url:
        raise RuntimeError("SEPOLIA_RPC_URL is missing from .env")

    w3 = Web3(Web3.HTTPProvider(rpc_url))

    if not w3.is_connected():
        raise RuntimeError("Could not connect to Sepolia")

    if w3.eth.chain_id != 11155111:
        raise RuntimeError(
            f"Wrong network. Expected Sepolia (11155111), "
            f"got {w3.eth.chain_id}"
        )

    return w3


def record_fingerprint(fingerprint: str) -> str:
    """Store the fingerprint in Sepolia transaction calldata."""

    private_key = os.getenv("PRIVATE_KEY")
    wallet_address = os.getenv("WALLET_ADDRESS")

    if not private_key:
        raise RuntimeError("PRIVATE_KEY is missing from .env")

    if not wallet_address:
        raise RuntimeError("WALLET_ADDRESS is missing from .env")

    w3 = get_web3()

    account = w3.eth.account.from_key(private_key)

    # Make sure the private key belongs to the configured wallet.
    if account.address.lower() != wallet_address.lower():
        raise RuntimeError(
            "PRIVATE_KEY does not belong to WALLET_ADDRESS"
        )

    nonce = w3.eth.get_transaction_count(account.address)

    # Put the fingerprint into transaction calldata.
    data = w3.to_bytes(text=fingerprint)

    transaction = {
        "from": account.address,
        "to": account.address,
        "value": 0,
        "nonce": nonce,
        "chainId": 11155111,
        "gas": 100000,
        "gasPrice": w3.eth.gas_price,
        "data": data,
    }

    signed = w3.eth.account.sign_transaction(
        transaction,
        private_key,
    )

    tx_hash = w3.eth.send_raw_transaction(
        signed.raw_transaction
    )

    return tx_hash.hex()


def retrieve_and_verify(
    tx_hash: str,
    expected_fingerprint: str,
) -> str:
    """Retrieve the fingerprint and report VERIFIED or TAMPERED."""

    w3 = get_web3()

    transaction = w3.eth.get_transaction(tx_hash)

    data = transaction["input"]

    if isinstance(data, bytes):
        raw_data = data
    else:
        raw_data = bytes.fromhex(data[2:])

    stored_fingerprint = raw_data.decode("utf-8")

    if stored_fingerprint == expected_fingerprint:
        return "VERIFIED"

    return "TAMPERED"