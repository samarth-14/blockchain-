from blockchain import record_fingerprint, retrieve_and_verify


fingerprint = "test-fingerprint-123456789"

print("Sending fingerprint to Sepolia...")

tx_hash = record_fingerprint(fingerprint)

print("Transaction sent!")
print("Transaction hash:")
print(tx_hash)

print("\nWaiting for transaction to be mined...")

from blockchain import get_web3

w3 = get_web3()

receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

print("Transaction mined!")
print("Block number:", receipt["blockNumber"])

print("\nVerifying fingerprint...")

result = retrieve_and_verify(
    tx_hash,
    fingerprint,
)

print("Result:", result)