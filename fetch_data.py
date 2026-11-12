"""Download and verify the three USPTO-50K files used by the benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

COMMIT = "96fb40cd622ae4a7645e51b50479c930341afee8"
BASE_URL = f"https://raw.githubusercontent.com/coleygroup/openretro/{COMMIT}/data/USPTO_50k/raw"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    manifest = Path(__file__).resolve().parent / "reference" / "data_audit.json"
    files = json.loads(manifest.read_text(encoding="utf-8"))
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    for name, metadata in files.items():
        destination = args.raw_dir / name
        expected = metadata["sha256"]
        if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected:
            print(f"Verified {destination}")
            continue
        with urlopen(f"{BASE_URL}/{name}", timeout=120) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch for {name}")
        temporary = destination.with_suffix(".partial")
        temporary.write_bytes(payload)
        temporary.replace(destination)
        print(f"Downloaded and verified {destination}")


if __name__ == "__main__":
    main()
