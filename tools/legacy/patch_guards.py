#!/usr/bin/env python3
"""Neutralize anti-tamper guards in Brawl Stars 36.218 armeabi-v7a libg.so.

Research use only: produces an offline-runnable client for local RL experiments.
Each target function's entry is overwritten with a 2-instruction ARM stub that
returns immediately. The rest of .text, the symbol table and DWARF are untouched,
so the binary stays fully analyzable.

Usage: python3 patch_guards.py <in/libg.so> <out/libg.so>
"""
import hashlib
import sys

EXPECTED_SHA256 = "f70b321c5597c959674e2aabebafc95939422a022fc812ecda51860ad49c1df9"

# ARM (A32) little-endian stubs. All targets sit in the first LOAD segment
# (vaddr 0 -> file offset 0), so file_offset == symbol st_value, and all are
# ARM mode (even st_value).
RET0 = bytes.fromhex("0000a0e3" "1eff2fe1")   # mov r0, #0 ; bx lr
RET1 = bytes.fromhex("0100a0e3" "1eff2fe1")   # mov r0, #1 ; bx lr


def thunk(target_va):
    """Absolute tail-call to an ARM function: ldr ip,[pc,#0]; bx ip; .word target."""
    return (bytes.fromhex("00c09fe5" "1cff2fe1")
            + target_va.to_bytes(4, "little"))


# Arxan-obfuscated createGameInstance -> plain GameMain::getInstance (community recipe)
GAMEMAIN_GETINSTANCE = 0x002897e4

# (virtual_address, stub, label). va == file offset for this segment.
TARGETS = [
    # --- background detection/guard thread spawners: stop them starting ---
    (0x0036ecf0, RET0, "lv1_detect_frida_library::create_guard_thread"),
    (0x00512f58, RET0, "lv1_detect_frida_library::create_detection_thread"),
    (0x005cfaf0, RET0, "lv1_detect_frida_library::create_library_guard_thread"),
    (0x002582f0, RET0, "lv1_detect_frida_server::create_guard_thread"),
    (0x004d00f0, RET0, "lv1_detect_frida_server::create_detection_thread"),
    (0x00348890, RET0, "lv1_detect_frida_server::create_library_guard_thread"),
    (0x00678bfc, RET0, "lv1_frida_killer::create_guard_thread"),
    (0x003c1d14, RET0, "lv1_frida_killer::create_detection_thread"),
    (0x005e922c, RET0, "lv1_frida_killer::create_library_guard_thread"),
    (0x0049721c, RET0, "lv1_root_detect_non_tamper_action::init"),
    (0x004b62d0, RET0, "lv1_root_detect_tamper_action::init"),
    (0x000c827c, RET0, "lv1_memory_guard::_GLOBAL__sub_I_AndroidMemoryAnomalyGuard"),
    # --- Arxan obfuscated entry + tamper-response handler ---
    (0x0033bcb8, thunk(GAMEMAIN_GETINSTANCE), "g_createGameInstance -> GameMain::getInstance"),
    (0x008339f4, RET0, "AntiCheat::guard_callback -> noop"),
    # --- integrity gates that matter once the APK is repacked/resigned ---
    (0x00595c20, RET1, "GameApp::isSignatureValid -> true"),
    (0x0094c248, RET0, "AntiCheat::isTampered -> false"),
    (0x0034466c, RET0, "AntiCheat::isSessionTampered -> false"),
    (0x003ad6d0, RET0, "AntiCheat::isRepackaged -> false"),
    (0x005169bc, RET0, "AntiCheat::isEmulator -> false"),
    (0x00247670, RET0, "GameApp::isEmulator -> false"),
    (0x0050ab84, RET0, "AntiCheat::isRooted -> false"),
    (0x006b7cd0, RET0, "AntiCheat::getAntihackFlags -> 0"),
    (0x00287814, RET0, "AntiCheat::hasHackerTools -> false"),
    (0x004695a0, RET0, "AntiCheat::isSideLoaded -> false"),
]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    buf = bytearray(open(src, "rb").read())

    got = hashlib.sha256(buf).hexdigest()
    if got != EXPECTED_SHA256:
        sys.exit(f"ERROR: input sha256 {got} != expected {EXPECTED_SHA256}")

    for va, stub, label in TARGETS:
        before = bytes(buf[va:va + len(stub)])
        buf[va:va + len(stub)] = stub
        print(f"  @0x{va:08x}  {before.hex()} -> {stub.hex()}  {label}")

    open(dst, "wb").write(buf)
    print(f"\nPatched {len(TARGETS)} functions.")
    print(f"out sha256: {hashlib.sha256(buf).hexdigest()}")
    print(f"size: {len(buf)} (unchanged: {len(buf) == len(open(src, 'rb').read())})")


if __name__ == "__main__":
    main()
