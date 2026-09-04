import os
from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

rpc_url = os.getenv("SEPOLIA_RPC_URL")
wallet = os.getenv("WALLET_ADDRESS")

if not rpc_url:
    raise RuntimeError("SEPOLIA_RPC_URL is missing from .env")

if not wallet:
    raise RuntimeError("WALLET_ADDRESS is missing from .env")

w3 = Web3(Web3.HTTPProvider(rpc_url))

print("Connected:", w3.is_connected())
print("Chain ID:", w3.eth.chain_id)
print("Wallet:", wallet)

balance = w3.eth.get_balance(wallet)
print("Balance:", w3.from_wei(balance, "ether"), "ETH")