from pipeline.blockchain import record_fingerprint, retrieve_and_verify


fingerprint = "phase4-test-fingerprint-123456"


print("1. Recording fingerprint on Sepolia...")
tx_hash = record_fingerprint(fingerprint)

print("2. Transaction sent!")
print("Transaction hash:", tx_hash)

print("3. Retrieving and verifying...")

result = retrieve_and_verify(tx_hash, fingerprint)

print("4. Verification result:", result)