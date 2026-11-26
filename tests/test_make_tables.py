"""The README table generators rewrite README.md in place; they must not depend on the locale encoding."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script", ["make_results.py", "make_bakeoff.py"])
def test_tables_regenerate_unchanged_under_ascii_locale(tmp_path, script):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "figures").mkdir()
    shutil.copy(ROOT / "scripts" / script, tmp_path / "scripts" / script)
    shutil.copy(ROOT / "README.md", tmp_path / "README.md")
    shutil.copytree(ROOT / "results", tmp_path / "results")
    # an ASCII locale stands in for a non-UTF-8 default such as cp1252 on Windows; README has non-ASCII (um sign, arrows)
    env = dict(os.environ, LC_ALL="C", LANG="C", PYTHONCOERCECLOCALE="0", PYTHONUTF8="0", PYTHONIOENCODING="utf-8",
               MPLBACKEND="Agg")
    subprocess.run([sys.executable, "-X", "utf8=0", str(tmp_path / "scripts" / script)], env=env, check=True,
                   capture_output=True)
    before = (ROOT / "README.md").read_text(encoding="utf-8")
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == before
