#!/usr/bin/env bash
# Install a pinned VC3D Linux AppImage for headless CLI use (vc_render_tifxyz etc.).
# The AppImage is extracted (no FUSE needed) and run through AppRun so its bundled loader/libs are used;
# host GL/X11 libs it does not bundle are installed from the distro (see ScrollPrize/villa#2153).
set -euo pipefail
# VC3D_TAG: per-build archive release holding this exact asset; `stable` is rolling and only the fallback.
# Whatever tag serves it, the file must match VC3D_SHA256 (see docs/data-inventory.md).
VC3D_TAG="${VC3D_TAG:-vc3d-archive-latest-433320258}"
VC3D_TAG_FALLBACK="${VC3D_TAG_FALLBACK:-stable}"
VC3D_ASSET="${VC3D_ASSET:-VC3D-f7756c7-2026-11-03-linux-x86_64.AppImage}"
VC3D_SHA256="${VC3D_SHA256:-17d9f2649c5f42b76c39e4e09ec87412002f505577e48400693c477b581e9c05}"
VC3D_DIR="${VC3D_DIR:-/data/vesuvius/vc3d/${VC3D_ASSET%.AppImage}}"

if command -v dnf >/dev/null; then
  dnf -y -q install libglvnd-egl libglvnd-glx libglvnd-opengl libICE libSM libX11 libX11-xcb libxcb \
    fontconfig freetype harfbuzz expat zlib gmp libgpg-error libcom_err libuuid ca-certificates
elif command -v apt-get >/dev/null; then
  apt-get update -qq && apt-get install -y -qq libegl1 libglx0 libopengl0 libice6 libsm6 libx11-xcb1 libx11-6 \
    libcom-err2 libexpat1 libfontconfig1 libfreetype6 libgmp10 libgpg-error0 libharfbuzz0b libuuid1 libxcb1 zlib1g ca-certificates curl
fi

mkdir -p "$VC3D_DIR" && cd "$VC3D_DIR"
if [ ! -f "$VC3D_ASSET" ]; then
  for tag in "$VC3D_TAG" "$VC3D_TAG_FALLBACK"; do
    curl -fL --retry 5 -o "$VC3D_ASSET.part" \
      "https://github.com/ScrollPrize/villa/releases/download/${tag}/${VC3D_ASSET}" && break
    echo "VC3D asset not found under release ${tag}" >&2; rm -f "$VC3D_ASSET.part"
  done
  echo "$VC3D_SHA256  $VC3D_ASSET.part" | sha256sum -c -
  mv "$VC3D_ASSET.part" "$VC3D_ASSET"
fi
echo "$VC3D_SHA256  $VC3D_ASSET" | sha256sum -c -
sha256sum "$VC3D_ASSET" | tee "$VC3D_ASSET.sha256"
if [ ! -x squashfs-root/AppRun ]; then
  chmod +x "$VC3D_ASSET" && "./$VC3D_ASSET" --appimage-extract >/dev/null
fi
# wrapper: vc3d <tool> [args...]
cat > /usr/local/bin/vc3d <<EOF
#!/usr/bin/env bash
exec "$VC3D_DIR/squashfs-root/AppRun" "\$@"
EOF
chmod +x /usr/local/bin/vc3d
(cd squashfs-root/usr/bin && printf '%s\n' vc_*) | head -40
vc3d vc_render_tifxyz --help 2>&1 | head -30
