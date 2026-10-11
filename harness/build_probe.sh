#!/bin/sh
# Build hostprobe + driver as 32-bit armv7a PIEs with NDK 25.1.
set -e
cd "$(dirname "$0")"
CC=$(ls -d ~/Android/Sdk/ndk/25.1*/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi21-clang | head -1)
"$CC" -O1 -Wall -fPIE -pie -o hostprobe jni/hostprobe.c -ldl
"$CC" -O1 -Wall -fPIE -pie -o driver jni/driver.c -ldl -lm
echo "built $(pwd)/hostprobe $(pwd)/driver"
