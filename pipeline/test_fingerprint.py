import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.fingerprint import canonicalize, sha256_fingerprint


metadata = {
    "title": "Test Post",
    "author": "Alice",
    "timestamp": "2026-09-04",
}

print("Canonical:")
print(canonicalize(metadata))

print("\nFingerprint:")
print(sha256_fingerprint(metadata))