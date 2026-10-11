#!/usr/bin/env bash
# Build an offline research client: original authentic v36.218 APK + Frida gadget,
# with System.loadLibrary("gadget") injected into com.supercell.titan.GameApp right
# after libg.so loads. Gadget has no config -> defaults to listen 127.0.0.1:27042,
# on_load=wait (app pauses after libg loads, before native game init). Debug-signed.
#
# Inputs (from scratchpad, passed as $1):
#   $1/dex/classes.dex      patched dex (gadget loader injected)
#   $1/gadget/libgadget.so  frida-gadget android-arm
# Usage: tools/build_gadget_client.sh <scratchpad_dir>
set -euo pipefail
SCRATCH="$1"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BT=~/Android/Sdk/build-tools/36.0.0
SRC_APK="$ROOT/apk/brawl-stars-36-218.apk"
OUT=$ROOT/apk/patched
mkdir -p "$OUT"

WORK="$SCRATCH/build"
rm -rf "$WORK"; mkdir -p "$WORK/lib/armeabi-v7a"
cp -f "$SRC_APK" "$WORK/bs36218-gadget-unsigned.apk"
cp -f "$SCRATCH/dex/classes.dex" "$WORK/classes.dex"
cp -f "$SCRATCH/gadget/libgadget.so" "$WORK/lib/armeabi-v7a/libgadget.so"
# gadget config + bypass script (named .so so extractNativeLibs=true copies them to the lib dir)
[ -f "$SCRATCH/gadget/libgadget.config.so" ] && cp -f "$SCRATCH/gadget/libgadget.config.so" "$WORK/lib/armeabi-v7a/libgadget.config.so"
[ -f "$SCRATCH/gadget/libscript.so" ] && cp -f "$SCRATCH/gadget/libscript.so" "$WORK/lib/armeabi-v7a/libscript.so"

cd "$WORK"
# replace the patched dex and add the gadget lib (deflate is fine: extractNativeLibs=true)
zip -q -X "bs36218-gadget-unsigned.apk" classes.dex
zip -q -X "bs36218-gadget-unsigned.apk" lib/armeabi-v7a/libgadget.so
[ -f "lib/armeabi-v7a/libgadget.config.so" ] && zip -q -X "bs36218-gadget-unsigned.apk" lib/armeabi-v7a/libgadget.config.so
[ -f "lib/armeabi-v7a/libscript.so" ] && zip -q -X "bs36218-gadget-unsigned.apk" lib/armeabi-v7a/libscript.so
# drop the original Supercell signature
zip -q -d "bs36218-gadget-unsigned.apk" 'META-INF/*.RSA' 'META-INF/*.SF' 'META-INF/*.MF' >/dev/null 2>&1 || true

"$BT/zipalign" -p -f 4 "bs36218-gadget-unsigned.apk" "bs36218-gadget-aligned.apk"
"$BT/apksigner" sign \
  --ks ~/.android/debug.keystore --ks-pass pass:android --key-pass pass:android \
  --ks-key-alias androiddebugkey \
  --v2-signing-enabled true --v3-signing-enabled true \
  --out "$OUT/bs36218-gadget.apk" "bs36218-gadget-aligned.apk"
"$BT/apksigner" verify "$OUT/bs36218-gadget.apk" >/dev/null && \
  echo "signed OK: $OUT/bs36218-gadget.apk"
echo "sha256: $(sha256sum "$OUT/bs36218-gadget.apk" | cut -d' ' -f1)"
echo "size:   $(du -h "$OUT/bs36218-gadget.apk" | cut -f1)"
