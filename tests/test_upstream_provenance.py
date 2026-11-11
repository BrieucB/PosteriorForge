"""Release provenance is checked without importing Torch or downloading weights."""
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "src" / "posteriorforge" / "upstream"


def test_six_baseline_modules_are_byte_identical_to_pinned_upstream():
    manifest = json.loads((ROOT / "provenance" / "upstream-npe-pfn.json").read_text())
    assert manifest["revision"] == "29c1e900a89781ee00677df6d2415a6664e59c37"
    paths = {p.relative_to(VENDOR).as_posix() for p in (VENDOR / "npe_pfn").glob("*.py")}
    assert len(paths) == manifest["file_count"] == 6
    assert paths == set(manifest["python_files"])
    for relative, metadata in manifest["python_files"].items():
        assert metadata["unchanged"] is True
        assert sha256((VENDOR / relative).read_bytes()).hexdigest() == metadata["sha256"]
    assert sha256((VENDOR / "LICENSE.npe-pfn").read_bytes()).hexdigest() == manifest["license_sha256"]


def test_importing_wrapper_does_not_load_inference_dependencies():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import posteriorforge.upstream; "
         "assert 'torch' not in sys.modules; assert 'tabpfn' not in sys.modules; "
         "assert 'sbi' not in sys.modules; assert 'npe_pfn' not in sys.modules"],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_model_manifest_pins_original_weights_and_excludes_git_weights():
    model = json.loads((ROOT / "provenance" / "model-tabpfn-v2.json").read_text())
    assert model["revision"] == "02a44705ff7790b50cded404bc59adb94da456a1"
    assert model["sha256"] == "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"
    assert model["size_bytes"] == 44390977
    assert model["weights_in_git"] is False
    assert (VENDOR / "LICENSE.TabPFN-model.txt").is_file()
    assert not list(ROOT.rglob("*.ckpt"))
