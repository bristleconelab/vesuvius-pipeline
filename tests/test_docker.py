"""Static checks that the CPU image stays in step with the main image (neither is built in the unit tests)."""
import re
from pathlib import Path

DOCKER = Path(__file__).resolve().parents[1] / "docker"


def _parse(name):
    text = (DOCKER / name).read_text()
    joined = re.sub(r"\\\n", " ", text)
    apt = re.search(r"apt-get install -y --no-install-recommends(.*?)&&", joined, re.S).group(1).split()
    args = dict(re.findall(r"^ARG (\w+)=(\S+)$", text, re.M))
    base = re.search(r"^FROM (\S+)$", text, re.M).group(1)
    return text, apt, args, base


def test_cpu_image_matches_main_image():
    _, apt, args, base = _parse("Dockerfile")
    cpu_text, cpu_apt, cpu_args, cpu_base = _parse("Dockerfile.cpu")
    assert cpu_base == base == "docker.io/library/ubuntu:24.04"
    assert cpu_apt == apt
    assert "libssl-dev" in apt
    for k in ("VILLA_REF", "VC3D_TAG", "VC3D_TAG_FALLBACK", "VC3D_ASSET", "VC3D_SHA256", "UV_VERSION"):
        assert cpu_args[k] == args[k], k
    assert "uv sync --extra models --frozen" in cpu_text
    assert "--no-install-package" in cpu_text and "COPY docker/torch-cpu.txt" in cpu_text


def test_vc3d_download_is_hash_checked_everywhere():
    """Every place that downloads the VC3D AppImage pins the same asset, tag and sha256 and verifies the hash."""
    root = DOCKER.parent
    _, _, args, _ = _parse("Dockerfile")
    sha, tag, asset = args["VC3D_SHA256"], args["VC3D_TAG"], args["VC3D_ASSET"]
    assert re.fullmatch(r"[0-9a-f]{64}", sha)
    assert tag != "stable" and args["VC3D_TAG_FALLBACK"] == "stable"
    for name in ("Dockerfile", "Dockerfile.cpu"):
        assert 'echo "${VC3D_SHA256}  vc3d.AppImage" | sha256sum -c -' in (DOCKER / name).read_text(), name
    get = (root / "scripts" / "get_vc3d.sh").read_text()
    assert f'VC3D_SHA256="${{VC3D_SHA256:-{sha}}}"' in get and f'VC3D_TAG="${{VC3D_TAG:-{tag}}}"' in get
    assert asset in get and 'echo "$VC3D_SHA256  $VC3D_ASSET" | sha256sum -c -' in get
    lead = (root / "kaggle" / "lead_screen" / "script.py").read_text()
    assert f"VC3D_SHA = '{sha}'" in lead and f"'{tag}'" in lead and asset in lead


def test_torch_cpu_requirements_are_hash_pinned():
    text = re.sub(r"\\\n", " ", (DOCKER / "torch-cpu.txt").read_text())
    reqs = [ln.split() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    names = {r[0].split("==")[0]: r for r in reqs}
    assert set(names) == {"torch", "torchvision"}
    for r in reqs:
        assert re.fullmatch(r"[a-z]+==[0-9.]+\+cpu", r[0]), r[0]
        hashes = [h for h in r[1:] if h.startswith("--hash=sha256:")]
        assert len(hashes) == len(r) - 1 >= 1
        assert all(re.fullmatch(r"--hash=sha256:[0-9a-f]{64}", h) for h in hashes)
