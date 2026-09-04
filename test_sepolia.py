import os
from web3 import Web3
from dotenv import load_dotenv

load_dotenv()

rpc_url = os.getenv("SEPOLIA_RPC_URL")
w3 = Web3(Web3.HTTPProvider(rpc_url))

wallet_address = "0xAEDCd1d4D7e82f1e9d701c4A096438067ae6d86A"

print("Connected:", w3.is_connected())
print("Chain ID:", w3.eth.chain_id)

balance_wei = w3.eth.get_balance(wallet_address)
balance_eth = w3.from_wei(balance_wei, "ether")

print("Wallet:", wallet_address)
print("Sepolia ETH balance:", balance_eth)

latest_block = w3.eth.block_number

print("Latest Sepolia block:", latest_block)