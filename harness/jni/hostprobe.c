// hostprobe.c — minimal 32-bit loader to test that an authentic libg.so (v29 or v36)
// loads headlessly in OUR OWN process (no injection, no APK) and survives its
// own .init_array, and that we can address it by base+offset.
//
// Deps are preloaded by ABSOLUTE path in dependency order, so DT_NEEDED resolves to the
// already-loaded soname and LD_LIBRARY_PATH / linker-namespace lookup no longer matters.
//
// Build: harness/build_probe.sh. Run on device as root (absolute path to the binary):
//   /data/local/tmp/bsh/hostprobe <libdir> [--peek 0xOFF ...] [--call 0xOFF a b c ...] [--hold]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <link.h>

static uintptr_t g_base = 0;
static int cb(struct dl_phdr_info *info, size_t sz, void *data) {
    if (info->dlpi_name && strstr(info->dlpi_name, "libg.so")) {
        g_base = (uintptr_t)info->dlpi_addr;
        printf("[probe] libg.so load base = %p  (%s)\n", (void*)g_base, info->dlpi_name);
    }
    return 0;
}

static void* try_open(const char *path) {
    // RTLD_GLOBAL so inter-lib deps resolve; RTLD_NOW to surface missing symbols now
    void *h = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (!h) { printf("[probe] dlopen(%s) FAILED: %s\n", path, dlerror()); }
    else    { printf("[probe] dlopen(%s) ok -> %p\n", path, h); }
    fflush(stdout);
    return h;
}

// libg.so mappings + perms: shows whether an Arxan build has flipped .text to r--
static void dump_maps(void) {
    FILE *f = fopen("/proc/self/maps", "r");
    if (!f) return;
    char line[512];
    while (fgets(line, sizeof line, f))
        if (strstr(line, "libg.so")) printf("[probe] maps: %s", line);
    fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 2) { printf("usage: %s <libdir> [--peek 0xOFF ...] [--hold]\n", argv[0]); return 1; }
    const char *dir = argv[1];
    printf("[probe] start (pid=%d) libdir=%s\n", getpid(), dir);

    // private deps first (leaves before dependents), then the target; absent ones are skipped
    const char *order[] = {"libsentry.so", "libsentry-android.so", "libnova_cpp.so", "libfmod.so"};
    char path[512];
    for (size_t i = 0; i < sizeof order / sizeof *order; i++) {
        snprintf(path, sizeof path, "%s/%s", dir, order[i]);
        if (access(path, R_OK) == 0) try_open(path);
    }
    snprintf(path, sizeof path, "%s/libg.so", dir);
    void *g = try_open(path);
    if (!g) { printf("[probe] RESULT: libg.so did NOT load\n"); return 2; }

    dl_iterate_phdr(cb, NULL);
    if (!g_base) { printf("[probe] RESULT: loaded but base not found\n"); return 3; }
    dump_maps();

    int hold = 0;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--hold")) { hold = 1; continue; }
        if (!strcmp(argv[i], "--peek") && i + 1 < argc) {
            unsigned long off = strtoul(argv[++i], NULL, 0);
            unsigned char *p = (unsigned char*)(g_base + off);
            printf("[probe] peek 0x%lx @ %p =", off, (void*)p);
            for (int j = 0; j < 12; j++) printf(" %02x", p[j]);
            printf("\n");
        }
        // call int f(int,int,int) at base+OFF (ARM-mode fn; unused args are ignored under AAPCS)
        if (!strcmp(argv[i], "--call") && i + 4 < argc) {
            unsigned long off = strtoul(argv[i + 1], NULL, 0);
            int a = (int)strtol(argv[i + 2], NULL, 0), b = (int)strtol(argv[i + 3], NULL, 0),
                c = (int)strtol(argv[i + 4], NULL, 0);
            i += 4;
            int (*fn)(int, int, int) = (int (*)(int, int, int))(g_base + off);
            printf("[probe] call 0x%lx(%d, %d, %d) = ", off, a, b, c);
            fflush(stdout);
            printf("%d\n", fn(a, b, c));
        }
    }
    printf("[probe] RESULT: libg.so LOADED + .init_array survived + addressable\n");

    // hold mode: keep process alive so a debugger (our OWN process, no anti-tamper) can attach
    if (hold) {
        printf("[probe] HOLDING (pid=%d, libg base=%p) — attach now\n", getpid(), (void*)g_base);
        fflush(stdout);
        for (;;) sleep(1);
    }
    return 0;
}
