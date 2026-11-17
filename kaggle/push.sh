#!/usr/bin/env bash
# Push a kernel folder to Kaggle with the account name filled in locally (the repo keeps the KAGGLE_USER placeholder).
# Usage: KAGGLE_USER=<your kaggle username> kaggle/push.sh kaggle/<kernel-dir>
set -euo pipefail
: "${KAGGLE_USER:?set KAGGLE_USER to your Kaggle username}"
src=$1; tmp=$(mktemp -d)
cp -R "$src"/. "$tmp"/
sed -i.bak "s#KAGGLE_USER/#${KAGGLE_USER}/#" "$tmp/kernel-metadata.json" && rm -f "$tmp/kernel-metadata.json.bak"
uvx -q --with six kaggle kernels push -p "$tmp"
rm -rf "$tmp"
