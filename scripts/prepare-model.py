#!/usr/bin/env python3
"""Download the immutable original TabPFN-v2 regressor for offline compute."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

MODEL_REPO = "Prior-Labs/TabPFN-v2-reg"
REVISION = "02a44705ff7790b50cded404bc59adb94da456a1"
FILENAME = "tabpfn-v2-regressor.ckpt"
SHA256 = "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"
SIZE = 44390977


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    cache = args.cache_dir.resolve()
    repo = Path(__file__).resolve().parents[1]
    if cache == repo or repo in cache.parents:
        parser.error("model cache must be outside the Git checkout")
    cache.mkdir(parents=True, exist_ok=True)
    model = cache / FILENAME
    base = f"https://huggingface.co/{MODEL_REPO}/resolve/{REVISION}"
    if not model.exists() or checksum(model) != SHA256:
        temporary = model.with_suffix(".download")
        with urllib.request.urlopen(f"{base}/{FILENAME}", timeout=120) as source:
            with temporary.open("wb") as target:
                for chunk in iter(lambda: source.read(1048576), b""):
                    target.write(chunk)
        if temporary.stat().st_size != SIZE or checksum(temporary) != SHA256:
            temporary.unlink(missing_ok=True)
            raise RuntimeError("checkpoint checksum/size mismatch; no model accepted")
        os.replace(temporary, model)
    for name in ["LICENSE.txt", "README.md"]:
        with urllib.request.urlopen(f"{base}/{name}", timeout=120) as source:
            (cache / name).write_bytes(source.read())
    manifest = {
        "project": "PosteriorForge", "repo_id": MODEL_REPO,
        "revision": REVISION, "filename": FILENAME, "sha256": SHA256,
        "size_bytes": model.stat().st_size, "weights_in_git": False,
        "source_url": f"{base}/{FILENAME}",
        "license": "Prior Labs License 1.1",
        "attribution": "Built with PriorLabs-TabPFN",
        "local_path": str(model),
    }
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
