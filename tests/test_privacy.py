"""Guard: the public repo must not carry personal details, internal hosts or local paths.

The word list is base64 so the guard itself does not publish it. Add words by re-encoding the list.
"""
import base64
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORDS = base64.b64decode(
    "dXpiZWsKdGFzaGtlbnQKdG9zaGtlbnQKYnVraGFyYQpyb2JlcnRhCmVkY29tCnZzcGhlcmUKa2VyaW8Kb3RhYmVrCmlvdGFiZWsKMDAwOS0wMDA4LTI3ODU="
).decode().split("\n")
PATTERNS = [
    re.compile(r"\b(10\.\d{1,3}|172\.(1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b"),  # private IPv4
    re.compile(r"/Users/[A-Za-z]"),                                                    # local macOS paths
    re.compile(r"/private/tmp/claude"),                                                # local scratch paths
    re.compile(r"[A-Za-z0-9._%+-]+@(gmail|yahoo|mail|outlook|icloud)\.[a-z]+", re.I),   # personal mailboxes
]
SKIP_SUFFIX = {".png", ".jpg", ".webp", ".tif", ".zarr", ".lock"}


def tracked_text_files():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    for name in out.splitlines():
        p = ROOT / name
        if p.suffix.lower() in SKIP_SUFFIX or not p.is_file() or p == Path(__file__):
            continue
        try:
            yield name, p.read_text()
        except UnicodeDecodeError:
            continue


def test_no_private_details_in_tracked_files():
    hits = []
    for name, text in tracked_text_files():
        low = text.lower()
        hits += [f"{name}: word #{i}" for i, w in enumerate(WORDS) if w and w in low]
        hits += [f"{name}: {m.group(0)}" for pat in PATTERNS for m in pat.finditer(text)]
    assert not hits, "private details in tracked files:\n" + "\n".join(hits[:50])
