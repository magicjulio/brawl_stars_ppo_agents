#!/usr/bin/env bash
# Repack + align + debug-sign the patched Brawl Stars client.
# Usage: tools/repack.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BT=~/Android/Sdk/build-tools/36.0.0
cd "$ROOT/apk"

cp -f brawl-stars-36-218.apk patched/bs36218-unsigned.apk
rm -rf patched/stage && mkdir -p patched/stage/lib/armeabi-v7a
cp -f patched/libg.so patched/stage/lib/armeabi-v7a/libg.so
( cd patched/stage && zip -0 -X -q ../bs36218-unsigned.apk lib/armeabi-v7a/libg.so )
cd patched
zip -q -d bs36218-unsigned.apk 'META-INF/*.RSA' 'META-INF/*.SF' 'META-INF/*.MF' >/dev/null 2>&1 || true
"$BT/zipalign" -p -f 4 bs36218-unsigned.apk bs36218-aligned.apk
"$BT/apksigner" sign \
  --ks ~/.android/debug.keystore --ks-pass pass:android --key-pass pass:android \
  --ks-key-alias androiddebugkey \
  --v2-signing-enabled true --v3-signing-enabled true \
  --out bs36218-patched.apk bs36218-aligned.apk
"$BT/apksigner" verify bs36218-patched.apk >/dev/null && echo "signed OK: $ROOT/apk/patched/bs36218-patched.apk"
