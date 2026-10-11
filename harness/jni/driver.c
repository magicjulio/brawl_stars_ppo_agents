// driver.c — headless offline battle driver on the authentic v36.218 libg.so (armeabi-v7a).
// Calls Logic-layer functions by base+offset (offsets from the v36 .symtab, scratch/mapping/v36_syms.tsv).
//
// M1: load all data tables from plaintext CSVs, replicating DataTableResourcesLoader::loadDataTables
//     but with our own file read instead of ResourceManager::getCSV (no engine/asset/JNI path).
//
// Build: harness/build_probe.sh. Run on device as root (absolute path to the binary):
//   /data/local/tmp/bsh36/driver <libdir> <dataroot> list|m1
// <dataroot> is the APK assets dir layout, i.e. <dataroot>/csv_logic/globals.csv.
#define _GNU_SOURCE   // cpu_set_t / sched_setaffinity
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <sys/mman.h>
#include <dlfcn.h>
#include <link.h>
#include <sys/ucontext.h>
#include <math.h>
#include <sched.h>
#include <sys/uio.h>

// ---- v36 offsets ----
#define OFF_LogicDataTables_constructInstance          0x102538
#define OFF_LogicDataTables_isLoaded                   0x27b644
#define OFF_LogicDataTables_getGlobals                 0x6fd8cc
#define OFF_LogicDataTables_getTable                   0x4f2458
#define OFF_LogicDataTables_getTableCount              0x71a4d8
#define OFF_LogicResources_createDataTableResourcesArray 0x3a3f4c
#define OFF_LogicResources_load                        0x2802c4
#define OFF_String_ctor_cstr                           0x570c30
#define OFF_StringView_ctor_cstr_len                   0x4f645c
#define OFF_CSVNode_ctor_lines_name                    0x2577f4
// M2: battle construction (Logic-layer subset of BattleMode::createOfflineServer @0x3463ec)
#define OFF_LogicDataTables_getLocationByName          0x49408c
#define OFF_LogicDataTables_getCharacterByName         0x37612c
#define OFF_LogicDataTables_createDefaultDisplayData   0x86e5b8
#define OFF_LogicLocationData_getGameModeVariation     0x8ecc98
#define OFF_LBMS_getGameModeForOfflinePractiseFromVariation 0x80af2c
#define OFF_LogicGamePlayUtil_getPlayerCountWithGameMode 0x8ed0a4
#define OFF_LBMS_ctor                                  0x3bf880
#define OFF_LBMS_setRandomSeed                         0x5d8398
#define OFF_LBMS_setLocation                           0x3aabe8
#define OFF_LBMS_generateTileMap                       0x84d504
#define OFF_LBMS_getLogicRandom                        0x31b4c0
#define OFF_LBMS_addPlayer                             0x70b558
#define OFF_LBMS_setBotDifficulty                      0x292104
#define OFF_LBMS_setControlledByBot                    0x6aec9c   // (playerIndex, bool): player+0x31, its hero +0x125
#define OFF_LBMS_setRaidDifficulty                     0x6037cc
#define OFF_LBMS_addGameObjects                        0x3274e4
#define OFF_LBMS_getTick                               0x913168
#define OFF_LBMS_getTicksGone                          0x4f122c
#define OFF_LogicRandom_setIteratedRandomSeed          0x1f0388
#define OFF_LogicGameObjectManagerServer_setLogicBattleModeServer 0x8c4640
#define OFF_LogicLong_ctor                             0x4c4f2c
#define OFF_LogicData_getName                          0x356b74
#define OFF_LogicGameObjectServer_getGlobalID          0x82c034
#define OFF_LogicGameObjectServer_getX                 0x7fc5dc
#define OFF_LogicGameObjectServer_getY                 0x7587ac
// M3: tick loop (tickUpdate @0x1304b0 calls executeOneTick(NULL, NULL) n times — no input/output needed)
#define OFF_LBMS_executeOneTick                        0x8a26d8
#define OFF_LBMS_preTick                               0x3afb10   // manager->preTick(players): removes dead objects
#define OFF_LBMS_executeOfflineInputMessages           0x27b4dc
// per-tick order from the client's offline loop BattleMode::update @0x8c6e90:
//   preTick() -> executeOfflineInputMessages() -> tickUpdate(1) == executeOneTick(NULL, NULL)
#define OFF_LBMS_isGameOverSet                         0x5a3388
#define OFF_LBMS_isGameOverOrRoundResetting            0x5039fc
#define OFF_LBMS_getTeamScore                          0x513c88
#define OFF_BitStream_destruct                         0x9063b4
#define OFF_vtable_LogicGameListener                   0xc1421c   // _ZTV17LogicGameListener, 356 B
#define SLOT_sendOfflineVisionUpdate                   0xfc       // byte offset from the vptr
#define OFF_LBMS_destruct                              0x57b170   // teardown as BattleMode::~BattleMode: destruct, ~LBMS, delete
#define OFF_LBMS_dtor                                  0x97cfc4
// Player input: executeOfflineInputMessages pops queued ClientInputMessages (+0x178 msgs, +0x184 player idx) and
// calls handleClientInput(input, playerIndex) for each ClientInput. We call handleClientInput directly.
#define OFF_LBMS_handleClientInput                     0x75c168
#define OFF_LGOMS_getGameObjectByID                    0x2a6df4
#define OFF_LCS_getWeaponSkill                         0x6fd204   // -> LogicSkillData* (CharacterData+0x118)
#define OFF_LCS_getUltiSkill                           0x82ff4c   // -> LogicSkillData* (CharacterData+0x11c)
#define OFF_LCS_getSkill                               0x86ab74   // (LogicSkillData*) -> LogicSkillServer*
#define OFF_LCS_isTargetVisibleToAttack                0x8db424   // (target, bool) -> bush/invisibility check
#define OFF_LSS_getMaxCharge                           0x672bd0   // (character) -> ammo slots
#define OFF_LSD_getCastingRangeTiles                   0x1c5f08   // skills.csv CastingRange, in 1/3 tiles (100 units)
#define OFF_LSD_isPositionalTargeted                   0x965288   // throwers: aim is an absolute ground point
#define OFF_LCS_getCharacterData                       0x108ea8   // -> LogicCharacterData* (+0xb0 speed)
#define OFF_LData_getInstanceID                        0x509ec8
#define OFF_LGMU_getBattleTicks                        0x3c6b14   // (server+0xa4) -> tick limit of the mode
#define OFF_LGMU_getRespawnSeconds                     0x8c6bd8   // (server+0xa4)
#define OFF_LGOMS_isPositionVisibleToPlayerTeam        0xcf5bc    // (mgr, tilemap, x, y, team members, tx, ty, mirrored)
#define OFF_LBMS_getTeamMemberPositions                0x2253a4   // (server, team) -> server+0x7c list, not a copy
#define OFF_LBMS_isPlayFieldMirroredForPlayer          0x8be73c   // static (team, server+0xa4)
// LogicPlayer battle stats: each getter looks its stat id up in the player's stat list (+0x54 entries, +0x5c count)
#define OFF_LP_getKills                                0x506a80   // stat 2
#define OFF_LP_getDamageDoneToBrawlers                 0x1422a0   // stat 3
#define OFF_LP_getDeaths                               0x90cbd4   // stat 13
#define TABLE_BOSSES 22
#define TABLE_CHARACTERS 16                                // the table getRandomHeroForBot draws bot heroes from
#define OFF_LogicCharacterData_isHero                  0x703770
#define SIZEOF_LBMS  744
// LogicBattleModeServer: +0 LogicGameObjectManagerServer*, +0x3c LogicBossData*, +0x40 playerCount,
//   +0x4c ArrayList<LogicPlayer*> players, +0xb4/+0xb7/+0xc4 flag bytes set by createOfflineServer
// LogicGameObjectManagerServer: +0 ArrayList<LogicGameObjectServer*> (getGameObjects returns this)
// LogicGameObjectServer: +8 LogicData*
// LogicCharacterServer: +0x28 player index (-1 = not a player hero), +0x2c team, +0x64 state, +0x90 hp, +0x94 max hp
// LogicPlayer: +0xc team, +0x10 character global id, +0x38 super charge, +0x3c super charge needed, +0xd0 gadget
// LogicSkillServer: +0 LogicSkillData*, +0xc ammo in 1/1000 shots

// ---- v36 layouts (from disasm) ----
// String (16 B): +0 codepoints, +4 byteLength, +8 inline chars if byteLength<=7 else char* heap
// LogicArrayList<T> (12 B): +0 T* data, +4 capacity, +8 size
// StringView (20 B), CSVNode (84 B)
// LogicDataTableResource: +8 String fileName, +0x18 tableIndex, +0x1c type (0 data table, 1 loc patch)
// LogicDataTable: +0xc itemCount
typedef struct { int codepoints, len; union { char inl[8]; char *heap; } u; } String;
typedef struct { void *data; int cap, size; } ArrayList;
#define SIZEOF_StringView 20
#define SIZEOF_CSVNode    84

static uintptr_t g_base;
#define FN(T, off) ((T)(g_base + (off)))

static const char *str_c(const String *s) { return s->len > 7 ? s->u.heap : s->u.inl; }

static int phdr_cb(struct dl_phdr_info *info, size_t sz, void *data) {
    if (info->dlpi_name && strstr(info->dlpi_name, "libg.so")) g_base = (uintptr_t)info->dlpi_addr;
    return 0;
}

static void on_fault(int sig, siginfo_t *si, void *ctx) {
    ucontext_t *uc = ctx;
    uintptr_t pc = uc->uc_mcontext.arm_pc, lr = uc->uc_mcontext.arm_lr;
    printf("\n[driver] FATAL signal %d addr=%p pc=%p (libg+0x%lx) lr=%p (libg+0x%lx)\n", sig, si->si_addr,
           (void*)pc, (unsigned long)(pc - g_base), (void*)lr, (unsigned long)(lr - g_base));
    // libg keeps APCS frames (push {..r11,lr}; add r11,sp,#N): [fp] = caller fp, [fp+4] = return address
    uintptr_t fp = uc->uc_mcontext.arm_fp;
    for (int i = 0; i < 16 && fp && !(fp & 3); i++) {
        uintptr_t ret = ((uintptr_t*)fp)[1], next = ((uintptr_t*)fp)[0];
        printf("[driver]   #%-2d ret=libg+0x%lx\n", i, (unsigned long)(ret - g_base));
        if (next <= fp) break;
        fp = next;
    }
    fflush(stdout);
    _exit(128 + sig);
}

static int load_libs(const char *dir) {
    const char *order[] = {"libsentry.so", "libsentry-android.so", "libnova_cpp.so", "libfmod.so", "libg.so"};
    char path[512];
    for (size_t i = 0; i < sizeof order / sizeof *order; i++) {
        snprintf(path, sizeof path, "%s/%s", dir, order[i]);
        if (!dlopen(path, RTLD_NOW | RTLD_GLOBAL)) { printf("[driver] dlopen(%s): %s\n", path, dlerror()); return -1; }
    }
    dl_iterate_phdr(phdr_cb, NULL);
    printf("[driver] libg.so base = %p\n", (void*)g_base);
    return g_base ? 0 : -1;
}

static char *read_file(const char *path, long *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = malloc(*n + 1);
    if (fread(buf, 1, *n, f) != (size_t)*n) { fclose(f); free(buf); return NULL; }
    buf[*n] = 0;
    fclose(f);
    return buf;
}

// Split like DataLoaderCSV::load: lines at '\n', drop a trailing '\r', skip empty lines.
// The StringViews point into buf, so buf must stay alive (we never free it).
static ArrayList *split_lines(char *buf, long n) {
    void (*sv_ctor)(void*, const char*, int) = FN(void(*)(void*, const char*, int), OFF_StringView_ctor_cstr_len);
    int cap = 64, count = 0;
    char *views = malloc(cap * SIZEOF_StringView);
    for (long pos = 0; pos < n;) {
        long end = pos;
        while (end < n && buf[end] != '\n') end++;
        long len = end - pos;
        if (len > 0 && buf[end - 1] == '\r') len--;
        if (len > 0) {
            if (count == cap) views = realloc(views, (cap *= 2) * SIZEOF_StringView);
            sv_ctor(views + count * SIZEOF_StringView, buf + pos, (int)len);
            count++;
        }
        pos = end + 1;
    }
    ArrayList *l = malloc(sizeof *l);
    l->data = views; l->cap = cap; l->size = count;
    return l;
}

static int g_quiet;   // m1 prints per-table lines only when run on its own

static int cmd_m1(const char *root, int do_load) {
    void (*construct)(void)            = FN(void(*)(void), OFF_LogicDataTables_constructInstance);
    ArrayList *(*create_arr)(void)     = FN(ArrayList*(*)(void), OFF_LogicResources_createDataTableResourcesArray);
    void (*res_load)(ArrayList*, int, void*, void*, int) =
        FN(void(*)(ArrayList*, int, void*, void*, int), OFF_LogicResources_load);
    void (*string_ctor)(String*, const char*) = FN(void(*)(String*, const char*), OFF_String_ctor_cstr);
    void (*node_ctor)(void*, ArrayList*, const String*) =
        FN(void(*)(void*, ArrayList*, const String*), OFF_CSVNode_ctor_lines_name);
    int (*is_loaded)(void)            = FN(int(*)(void), OFF_LogicDataTables_isLoaded);
    void *(*get_globals)(void)        = FN(void*(*)(void), OFF_LogicDataTables_getGlobals);
    void *(*get_table)(int)           = FN(void*(*)(int), OFF_LogicDataTables_getTable);
    int (*table_count)(void)          = FN(int(*)(void), OFF_LogicDataTables_getTableCount);

    construct();
    ArrayList *arr = create_arr();
    printf("[m1] resources: %d (cap %d)\n", arr->size, arr->cap);

    int missing = 0;
    for (int i = 0; i < arr->size; i++) {
        char *res = ((char**)arr->data)[i];
        const char *name = str_c((String*)(res + 8));
        int idx = *(int*)(res + 0x18), type = *(int*)(res + 0x1c);
        char path[512];
        snprintf(path, sizeof path, "%s/%s", root, name);
        long n = 0;
        char *buf = read_file(path, &n);
        if (!g_quiet || !buf) printf("[m1] %2d %-40s idx=%2d type=%d %s", i, name, idx, type, buf ? "" : "MISSING");
        if (!buf) { printf("\n"); missing++; continue; }
        ArrayList *lines = split_lines(buf, n);
        if (!g_quiet) printf("lines=%d", lines->size);
        if (!do_load) { printf("\n"); continue; }

        String *sname = malloc(sizeof *sname);
        string_ctor(sname, name);
        void *node = calloc(1, SIZEOF_CSVNode);
        node_ctor(node, lines, sname);
        res_load(arr, i, node, NULL, 0);   // last index also runs LogicDataTables::createReferences()
        if (g_quiet) continue;
        if (type == 0) {
            char *t = get_table(idx);
            printf(" -> items=%d", t ? *(int*)(t + 0xc) : -1);
        }
        printf("\n");
    }
    if (!do_load) { printf("[m1] list done, %d missing\n", missing); return missing ? 1 : 0; }

    int loaded = is_loaded();
    void *globals = get_globals();
    int nonnull = 0;
    for (int t = 0; t < table_count(); t++) nonnull += get_table(t) != NULL;
    printf("[m1] isLoaded=%d globals=%p tables=%d/%d missing_files=%d\n", loaded, globals, nonnull,
           table_count(), missing);
    printf("[m1] RESULT: %s\n", loaded && globals ? "M1 PASS — data tables loaded" : "M1 FAIL");
    return loaded && globals ? 0 : 1;
}

// ---- M2: construct an offline bot battle ----
// key=value args (defaults = the game's offline-practice path, HomePage::startGame @0x511000):
//   loc=Gemgrab1 hero=ShotgunGirl gt=<from location's mode> map=1 pm=3 diff=0 seed=12345 team=0
static const char *kv(int argc, char **argv, const char *key, const char *def) {
    size_t n = strlen(key);
    for (int i = 0; i < argc; i++)
        if (!strncmp(argv[i], key, n) && argv[i][n] == '=') return argv[i] + n + 1;
    return def;
}

static const char *data_name(void *data) {
    const String *(*get_name)(void*) = FN(const String*(*)(void*), OFF_LogicData_getName);
    return data ? str_c(get_name(data)) : "(null)";
}

// The offline server (+0xb4 set) encodes a fresh 1.5 KB "offline vision" BitStream every tick and hands it to
// server+0x18 ->sendOfflineVisionUpdate(tick, bs). The real client (BattleScreen) consumes it; the default
// LogicGameListener's version is `bx lr`, so it leaks ~1.8 KB/tick. We swap in a listener whose vtable is a
// copy of LogicGameListener's with that one slot pointing at a function that frees the stream.
static void on_vision_update(void *self, int tick, void *bs) {
    FN(void(*)(void*), OFF_BitStream_destruct)(bs);   // frees the owned buffer at +0x10
    free(bs);                                         // the 28-byte object came from operator new
}

static void install_listener(char *server) {
    static void *vtbl[356 / sizeof(void*)];
    static void *listener[1];
    memcpy(vtbl, (void*)(g_base + OFF_vtable_LogicGameListener), sizeof vtbl);
    void **vptr = vtbl + 2;                           // vptr skips offset-to-top + typeinfo
    vptr[SLOT_sendOfflineVisionUpdate / sizeof(void*)] = (void*)on_vision_update;
    listener[0] = vptr;
    *(void**)(server + 0x18) = listener;
}

static void destroy_battle(char *server);

// Builds one battle (data tables must be loaded). verbose=0 for serve mode, where stdout is the protocol channel.
static char *build_battle(int argc, char **argv, int verbose) {
    void *(*loc_by_name)(const String*, void*)  = FN(void*(*)(const String*, void*), OFF_LogicDataTables_getLocationByName);
    void *(*char_by_name)(const String*, void*) = FN(void*(*)(const String*, void*), OFF_LogicDataTables_getCharacterByName);
    void *(*default_dd)(const String*)          = FN(void*(*)(const String*), OFF_LogicDataTables_createDefaultDisplayData);
    int (*loc_variation)(void*)                 = FN(int(*)(void*), OFF_LogicLocationData_getGameModeVariation);
    int (*practise_gm)(int)                     = FN(int(*)(int), OFF_LBMS_getGameModeForOfflinePractiseFromVariation);
    int (*player_count_for)(int)                = FN(int(*)(int), OFF_LogicGamePlayUtil_getPlayerCountWithGameMode);
    void (*server_ctor)(void*, int, int)        = FN(void(*)(void*, int, int), OFF_LBMS_ctor);
    void (*set_seed)(void*, int)                = FN(void(*)(void*, int), OFF_LBMS_setRandomSeed);
    void (*set_location)(void*, void*)          = FN(void(*)(void*, void*), OFF_LBMS_setLocation);
    void (*gen_tilemap)(void*, int, int, int)   = FN(void(*)(void*, int, int, int), OFF_LBMS_generateTileMap);
    void *(*get_random)(void*)                  = FN(void*(*)(void*), OFF_LBMS_getLogicRandom);
    void (*set_iter_seed)(void*, int)           = FN(void(*)(void*, int), OFF_LogicRandom_setIteratedRandomSeed);
    void (*mgr_set_server)(void*, void*)        = FN(void(*)(void*, void*), OFF_LogicGameObjectManagerServer_setLogicBattleModeServer);
    void (*long_ctor)(void*)                    = FN(void(*)(void*), OFF_LogicLong_ctor);
    void (*add_player)(void*, void*, void*, void*, void*, int, void*, int, void*) =
        FN(void(*)(void*, void*, void*, void*, void*, int, void*, int, void*), OFF_LBMS_addPlayer);
    void (*set_bot_diff)(void*, int)            = FN(void(*)(void*, int), OFF_LBMS_setBotDifficulty);
    void (*set_raid_diff)(void*, int)           = FN(void(*)(void*, int), OFF_LBMS_setRaidDifficulty);
    void (*add_objects)(void*, int, int, ArrayList*, int, ArrayList*) =
        FN(void(*)(void*, int, int, ArrayList*, int, ArrayList*), OFF_LBMS_addGameObjects);
    int (*get_tick)(void*)                      = FN(int(*)(void*), OFF_LBMS_getTick);
    void *(*get_table)(int)                     = FN(void*(*)(int), OFF_LogicDataTables_getTable);
    void (*string_ctor)(String*, const char*)   = FN(void(*)(String*, const char*), OFF_String_ctor_cstr);

    const char *loc_name = kv(argc, argv, "loc", "Gemgrab1"), *hero_name = kv(argc, argv, "hero", "ShotgunGirl");
    int map_type = atoi(kv(argc, argv, "map", "1")), pmode = atoi(kv(argc, argv, "pm", "3"));
    int diff = atoi(kv(argc, argv, "diff", "0")), seed = atoi(kv(argc, argv, "seed", "12345"));
    int team = atoi(kv(argc, argv, "team", "0"));

    String s;
    string_ctor(&s, loc_name);
    void *loc = loc_by_name(&s, NULL);
    string_ctor(&s, hero_name);
    void *hero = char_by_name(&s, NULL);
    if (!loc || !hero) {
        printf(verbose ? "[m2] unknown loc=%s (%p) or hero=%s (%p)\n" : "{\"error\":\"unknown loc=%s (%p) or hero=%s (%p)\"}\n",
               loc_name, loc, hero_name, hero);
        return NULL;
    }
    int variation = loc_variation(loc);
    int gm = atoi(kv(argc, argv, "gt", "-1"));
    if (gm < 0) gm = practise_gm(variation);
    int pc = player_count_for(gm);
    if (verbose)
        printf("[m2] loc=%s variation=%d gameType=%d playerCount=%d map=%d pm=%d diff=%d seed=%d hero=%s\n",
               data_name(loc), variation, gm, pc, map_type, pmode, diff, seed, data_name(hero));

    char *server = calloc(1, SIZEOF_LBMS);
    server_ctor(server, gm, pc);
    server[0xc4] = 0;                                  // sm_offlineMaxedHeroes
    server[0xb4] = 1;
    set_seed(server, seed);
    set_location(server, loc);
    void *bosses = get_table(TABLE_BOSSES);            // server[0x3c] = bosses->vtbl[5](bosses, 0)
    void *(*vfn)(void*, int) = (*(void*(***)(void*, int))bosses)[0x14 / 4];
    *(void**)(server + 0x3c) = vfn(bosses, 0);
    mgr_set_server(*(void**)server, server);
    gen_tilemap(server, pc, map_type, 0);              // campaignProgress 0
    set_iter_seed(get_random(server), seed);           // game uses rand() here; we stay deterministic
    if (verbose) printf("[m2] server constructed + tilemap generated\n");

    char *id = calloc(1, 16);
    long_ctor(id);
    string_ctor(&s, "Agent");
    void *dd = default_dd(&s);
    add_player(server, id, hero, NULL, NULL, team, dd, 0, NULL);  // no hero upgrades/emotes/skin (as the tutorial)
    // mates=<Name>[,<Name>...]: player 0's teammates get these brawlers (cycled) instead of the bots' random picks. We
    // add them like player 0, before addGameObjects, which then only fills the remaining (enemy) slots with bots.
    // Teammates beyond agents=<k> are handed to the bot AI below. Only for two-team modes (3v3: 2 teammates).
    const char *mates = kv(argc, argv, "mates", "");
    int n_mates = 0;
    if (*mates && pc == 6) {
        for (const char *p = mates; n_mates < pc / 2 - 1; n_mates++) {
            char name[64];
            size_t len = strcspn(p, ",");
            snprintf(name, sizeof name, "%.*s", (int)len, p);
            string_ctor(&s, name);
            void *mate = char_by_name(&s, NULL);
            if (!mate) {
                printf(verbose ? "[m2] unknown mates hero %s\n" : "{\"error\":\"unknown mates hero %s\"}\n", name);
                destroy_battle(server);
                return NULL;
            }
            char *mid = calloc(1, 16);
            long_ctor(mid);
            string_ctor(&s, "Agent");
            add_player(server, mid, mate, NULL, NULL, team, default_dd(&s), 0, NULL);  // own display data (owned)
            p = p[len] == ',' ? p + len + 1 : mates;          // next name, or start over
        }
    }
    server[0xb7] = (unsigned)(pmode - 1) < 2;
    set_bot_diff(server, diff);
    set_raid_diff(server, 0);
    // bots=<Name>[,<Name>...]: the bots' brawlers are drawn only from these. addBots picks each bot's hero with
    // LogicDataTables::getRandomHeroForBot @0x108414, which skips every hero on the list passed here (normally
    // empty): we list all heroes except the allowed ones.
    ArrayList banned = {0};
    const char *bots = kv(argc, argv, "bots", "");
    if (*bots) {
        void *chars = get_table(TABLE_CHARACTERS);
        int count = (*(int(***)(void*))chars)[0x10 / 4](chars);
        banned.data = calloc(count, sizeof(void*));
        banned.cap = count;
        int n_allowed = 0;
        for (int i = 0; i < count; i++) {
            void *c = (*(void*(***)(void*, int))chars)[0x14 / 4](chars, i);
            if (!FN(int(*)(void*), OFF_LogicCharacterData_isHero)(c)) continue;
            const char *name = data_name(c), *p = bots;
            size_t n = strlen(name);
            int allowed = 0;
            for (; *p; p += strcspn(p, ",") + (p[strcspn(p, ",")] == ','))
                if (strcspn(p, ",") == n && !strncmp(p, name, n)) allowed = 1;
            if (allowed) n_allowed++;
            else ((void**)banned.data)[banned.size++] = c;
        }
        if (verbose) printf("[m2] bots=%s: %d heroes allowed, %d banned\n", bots, n_allowed, banned.size);
        if (!n_allowed) {                              // every hero banned: the game would have no bot hero to pick
            printf(verbose ? "[m2] bots=%s names no hero\n" : "{\"error\":\"bots=%s names no hero\"}\n", bots);
            free(banned.data);
            destroy_battle(server);
            return NULL;
        }
    }
    add_objects(server, 0, 1, &banned, 0, NULL);       // spawns the bots
    free(banned.data);
    // agents=<k>: we steer player 0 and its first k-1 teammates (player order, see team_players()); enemies=<m>: also the
    // first m players of the other team (serve mode: play against our agents); the rest stay bots.
    // This is the game's own switch for a player taken over by / given back from a bot (LogicCharacterServer::tickAI
    // runs tickBot only for heroes with +0x125 set); respawns keep it, as tickSpawnHeroes passes the player's +0x31.
    ArrayList *plist = (ArrayList*)(server + 0x4c);
    int agents = atoi(kv(argc, argv, "agents", "1")), enemies = atoi(kv(argc, argv, "enemies", "0"));
    int team0 = *(int*)(((char**)plist->data)[0] + 0xc);
    for (int p = 1, k = 1, m = 0; p < plist->size; p++)
        if (*(int*)(((char**)plist->data)[p] + 0xc) == team0 ? k++ < agents : m++ < enemies)
            FN(void(*)(void*, int, int), OFF_LBMS_setControlledByBot)(server, p, 0);
        else if (p <= n_mates)                         // one of our added teammates that we don't steer: a bot
            FN(void(*)(void*, int, int), OFF_LBMS_setControlledByBot)(server, p, 1);
    install_listener(server);
    if (!verbose) return server;
    printf("[m2] player added, game objects added\n");

    ArrayList *players = (ArrayList*)(server + 0x4c);
    ArrayList *objs = *(ArrayList**)server;            // LogicGameObjectManagerServer: list at +0
    int (*gid)(void*) = FN(int(*)(void*), OFF_LogicGameObjectServer_getGlobalID);
    int (*gx)(void*) = FN(int(*)(void*), OFF_LogicGameObjectServer_getX);
    int (*gy)(void*) = FN(int(*)(void*), OFF_LogicGameObjectServer_getY);
    printf("[m2] tick=%d players=%d (expected %d) gameObjects=%d\n", get_tick(server), players->size, pc, objs->size);
    for (int i = 0; i < objs->size; i++) {
        void *o = ((void**)objs->data)[i];
        printf("[m2]   obj %2d gid=0x%08x data=%-24s x=%d y=%d\n", i, gid(o), data_name(*(void**)((char*)o + 8)), gx(o), gy(o));
    }
    int ok = get_tick(server) == 0 && players->size == pc && objs->size > 0;
    printf("[m2] RESULT: %s\n", ok ? "M2 PASS — battle constructed" : "M2 FAIL");
    return ok ? server : NULL;
}

static char *cmd_m2(const char *root, int argc, char **argv) {
    g_quiet = 1;
    if (cmd_m1(root, 1)) return NULL;
    return build_battle(argc, argv, 1);
}

static void destroy_battle(char *server) {
    FN(void(*)(void*), OFF_LBMS_destruct)(server);
    FN(void(*)(void*), OFF_LBMS_dtor)(server);
    free(server);
}

// ---- player actions ----
// ClientInput (36 B; ctor @0x8353b8, wire format ClientInput::decode @0x864100). Types handled by
// LogicBattleModeServer::handleClientInput:
//   0 attack, 1 super      x,y = aim point: offset from the hero for straight shots, absolute ground point for
//                          throwers (LogicSkillData::isPositionalTargeted) and for auto-aim. With autoshoot+has_target
//                          the server re-aims at target->getX/YForAutoshoot(). Aim is clamped to the visible area and
//                          to >= 100 units from the hero.
//   2 move                 x,y = absolute target point; the server clamps it to 900 units (3 tiles) from the hero.
//   3 stop                 stops a move in progress.
//   5/6 super aim on/off (UI state), 7 carryable aim, 8 gadget (x,y raw), 9 emote/pin, 10 steer controlled
//   projectile, 12 showdown give-up. carry (+0x20) must equal "is carrying something" for 0/1 to be accepted.
typedef struct {
    int seq, type, x, y, target;
    unsigned char autoshoot, has_target, pad[2];
    int pin, v1c;
    unsigned char carry, pad2[3];
} ClientInput;
enum { IN_ATTACK = 0, IN_SUPER = 1, IN_MOVE = 2, IN_STOP = 3, IN_GADGET = 8 };

// One agent decision. Attack/super/gadget are button presses applied once; movement holds until the next action.
typedef struct {
    float mx, my;   // move direction (any length); length < 0.3 = stand still
    int atk;        // 0 none, 1 auto-aim (closest visible enemy in range), 2 aim along (ax, ay)
    float ax, ay;   // aim direction; length 0..1 = fraction of max range (only throwers care about the length)
    int sup;        // super, same encoding as atk, aim (sx, sy); ignored until the super is charged
    float sx, sy;
    int gadget;     // 1 = use gadget (only if the hero has one equipped; our default loadout has none)
} Action;

typedef struct { float fx, fy; int moving, seq; } Ctl;   // per-agent controller state (last facing direction)

// A move input walks the hero to a point 900 units (25 ticks of walking) ahead, pathfinding around walls once (A*).
// Steps send it on their first tick and repeat it only every STEER_TICKS ticks: re-sending it every tick cost ~17% of
// the tick time in A* without changing where the hero goes.
#define STEER_TICKS 16

static int obj_x(void *o) { return FN(int(*)(void*), OFF_LogicGameObjectServer_getX)(o); }
static int obj_y(void *o) { return FN(int(*)(void*), OFF_LogicGameObjectServer_getY)(o); }

static char *player_at(char *server, int pidx) {
    ArrayList *players = (ArrayList*)(server + 0x4c);
    return pidx >= 0 && pidx < players->size ? ((char**)players->data)[pidx] : NULL;
}

static char *player_char(char *server, int pidx) {   // the player's hero, NULL while dead/respawning
    char *p = player_at(server, pidx);
    if (!p) return NULL;
    char *c = FN(char*(*)(void*, int), OFF_LGOMS_getGameObjectByID)(*(void**)server, *(int*)(p + 0x10));
    return c && *(int*)(c + 0x90) > 0 ? c : NULL;
}

// Player indices of the brawlers we steer in one team (build_battle's agents=<k> / enemies=<m>): its first `want`
// players in player order (player 0's team: player 0, then its teammates).
#define MAX_AGENTS 3
static int team_players(char *server, int team, int want, int *pidx) {
    ArrayList *players = (ArrayList*)(server + 0x4c);
    int k = 0;
    for (int p = 0; p < players->size && k < want && k < MAX_AGENTS; p++)
        if (*(int*)(player_at(server, p) + 0xc) == team) pidx[k++] = p;
    return k;
}

static int player_team(char *server, int pidx) {
    char *p = player_at(server, pidx);
    return p ? *(int*)(p + 0xc) : 0;
}

static int skill_range(void *skill) {   // logic units
    return skill ? FN(int(*)(void*), OFF_LSD_getCastingRangeTiles)(skill) * 100 : 0;
}

// Closest living enemy character within `range` (and visible to attack, if check_vis), like the in-game auto-aim.
static char *closest_enemy(char *server, char *self, int range, int check_vis) {
    int (*visible)(void*, void*, int) = FN(int(*)(void*, void*, int), OFF_LCS_isTargetVisibleToAttack);
    ArrayList *chars = (ArrayList*)(*(char**)server + 24);
    int sx = obj_x(self), sy = obj_y(self), team = *(int*)(self + 0x2c);
    long long best_d = (long long)range * range;
    char *best = NULL;
    for (int i = 0; i < chars->size; i++) {
        char *c = ((char**)chars->data)[i];
        if (c == self || *(int*)(c + 0x2c) == team || *(int*)(c + 0x90) <= 0) continue;
        long long dx = obj_x(c) - sx, dy = obj_y(c) - sy, d = dx * dx + dy * dy;
        if (d > best_d || (check_vis && !visible(self, c, 0))) continue;
        best = c; best_d = d;
    }
    return best;
}

// Inputs sent since the last serve reply, for the viewer: [type, world x, world y, auto-aim target gid or 0]
static int g_sent[32][4], g_nsent, g_log_inputs = 1;

static void send_input(char *server, int pidx, Ctl *ctl, ClientInput *in, int wx, int wy) {
    in->seq = ++ctl->seq & 0x7fff;
    FN(void(*)(void*, ClientInput*, int), OFF_LBMS_handleClientInput)(server, in, pidx);
    if (g_log_inputs && pidx == 0 && g_nsent < 32) {   // the frame shows player 0's inputs (drawn from its hero)
        int *e = g_sent[g_nsent++];
        e[0] = in->type; e[1] = wx; e[2] = wy; e[3] = in->has_target ? in->target : 0;
    }
}

static void fire(char *server, int pidx, char *c, Ctl *ctl, int type, int mode, float ax, float ay) {
    void *skill = (type == IN_SUPER ? FN(void*(*)(void*), OFF_LCS_getUltiSkill) : FN(void*(*)(void*), OFF_LCS_getWeaponSkill))(c);
    if (!skill) return;
    int range = skill_range(skill), px = obj_x(c), py = obj_y(c);
    ClientInput in = {.type = type};
    if (mode == 1) {
        char *t = closest_enemy(server, c, range, 1);
        if (t) {
            in.autoshoot = in.has_target = 1;
            in.target = FN(int(*)(void*), OFF_LogicGameObjectServer_getGlobalID)(t);
            in.x = obj_x(t); in.y = obj_y(t);
            send_input(server, pidx, ctl, &in, in.x, in.y);
            return;
        }
        ax = ctl->fx; ay = ctl->fy;          // nothing in range: fire where the hero faces, full range
    }
    float len = hypotf(ax, ay);
    if (len < 1e-3f) { ax = ctl->fx; ay = ctl->fy; len = 1; }
    if (len > 1) { ax /= len; ay /= len; }
    int dx = (int)(ax * range), dy = (int)(ay * range);
    int positional = FN(int(*)(void*), OFF_LSD_isPositionalTargeted)(skill);
    in.x = positional ? px + dx : dx;
    in.y = positional ? py + dy : dy;
    send_input(server, pidx, ctl, &in, px + dx, py + dy);
}

static void apply_action(char *server, int pidx, const Action *a, Ctl *ctl) {
    char *c = player_char(server, pidx);
    if (!c) { ctl->moving = 0; return; }
    int px = obj_x(c), py = obj_y(c);
    if (ctl->fx == 0 && ctl->fy == 0) {      // first use: face the map centre
        char *tm = *(char**)(server + 0x48);
        float cx = *(int*)(tm + 0x68) * 150 - px, cy = *(int*)(tm + 0x6c) * 150 - py, l = hypotf(cx, cy);
        ctl->fx = l > 0 ? cx / l : 0; ctl->fy = l > 0 ? cy / l : -1;
    }
    float m = hypotf(a->mx, a->my);
    if (m >= 0.3f) {
        char *tm = *(char**)(server + 0x48);
        int maxx = *(int*)(tm + 0x68) * 300 - 2, maxy = *(int*)(tm + 0x6c) * 300 - 2;
        ctl->fx = a->mx / m; ctl->fy = a->my / m;
        int tx = px + (int)(ctl->fx * 900), ty = py + (int)(ctl->fy * 900);
        ClientInput in = {.type = IN_MOVE, .x = tx < 1 ? 1 : tx > maxx ? maxx : tx, .y = ty < 1 ? 1 : ty > maxy ? maxy : ty};
        send_input(server, pidx, ctl, &in, in.x, in.y);
        ctl->moving = 1;
    } else if (ctl->moving) {
        ClientInput in = {.type = IN_STOP};
        send_input(server, pidx, ctl, &in, px, py);
        ctl->moving = 0;
    }
    if (a->atk) fire(server, pidx, c, ctl, IN_ATTACK, a->atk, a->ax, a->ay);
    if (a->sup) fire(server, pidx, c, ctl, IN_SUPER, a->sup, a->sx, a->sy);
    if (a->gadget) {
        ClientInput in = {.type = IN_GADGET, .x = px + (int)(ctl->fx * 300), .y = py + (int)(ctl->fy * 300)};
        send_input(server, pidx, ctl, &in, in.x, in.y);
    }
}

// Own-hero state not visible in the object list: [ammo (1/1000 shots), ammo slots, super charge, super needed, gid]
static void own_state(char *server, int pidx, int out[5]) {
    memset(out, 0, 5 * sizeof *out);
    char *p = player_at(server, pidx), *c = player_char(server, pidx);
    if (!p) return;
    out[2] = *(int*)(p + 0x38); out[3] = *(int*)(p + 0x3c);
    if (!c) return;
    out[4] = FN(int(*)(void*), OFF_LogicGameObjectServer_getGlobalID)(c);
    char *ss = FN(char*(*)(void*, void*), OFF_LCS_getSkill)(c, FN(void*(*)(void*), OFF_LCS_getWeaponSkill)(c));
    if (!ss) return;
    out[0] = *(int*)(ss + 0xc);
    out[1] = FN(int(*)(void*, void*), OFF_LSS_getMaxCharge)(ss, c);
}

// Scripted test policy (act=script): walk at the closest enemy until it is within 70% of weapon range, attack with
// auto-aim whenever someone is in range, use the super with auto-aim as soon as it is charged.
static void script_policy(char *server, int pidx, Action *a) {
    memset(a, 0, sizeof *a);
    char *c = player_char(server, pidx);
    if (!c) return;
    char *e = closest_enemy(server, c, 1 << 20, 0);
    if (!e) return;
    int range = skill_range(FN(void*(*)(void*), OFF_LCS_getWeaponSkill)(c));
    float dx = obj_x(e) - obj_x(c), dy = obj_y(e) - obj_y(c), d = hypotf(dx, dy);
    if (d > 0.7f * range) { a->mx = dx; a->my = dy; }
    a->atk = d <= range;
    a->sup = 1;
}

// ---- M3: run the battle headless to game over ----
// extra args: ticks=<max, default 6000> every=<print period, default 200>
static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static long rss_kb(void) {
    FILE *f = fopen("/proc/self/status", "r");
    char line[128];
    long kb = -1;
    while (f && fgets(line, sizeof line, f))
        if (!strncmp(line, "VmRSS:", 6)) kb = atol(line + 6);
    if (f) fclose(f);
    return kb;
}

static void dump_characters(char *server) {
    int (*gx)(void*) = FN(int(*)(void*), OFF_LogicGameObjectServer_getX);
    int (*gy)(void*) = FN(int(*)(void*), OFF_LogicGameObjectServer_getY);
    ArrayList *chars = (ArrayList*)(*(char**)server + 24);   // LogicGameObjectManagerServer::getCharacters()
    for (int i = 0; i < chars->size; i++) {
        char *c = ((char**)chars->data)[i];
        int *w = (int*)c;
        printf("[m3]     %-16s x=%5d y=%5d pidx=%2d +2c=%d +30=%d | +90=%d +94=%d +98=%d\n",
               data_name(*(void**)(c + 8)), gx(c), gy(c), w[0x28/4], w[0x2c/4], w[0x30/4],
               w[0x90/4], w[0x94/4], w[0x98/4]);
    }
}

// ---- recorder (rec=<file>): JSON replay for tools/viewer ----
// {"meta":{...},"frames":[[tick,scoreA,scoreB,[[gid,type,name,x,y,z,team,pidx,hp,maxhp],...],grid|0,own,inputs],...],
//  "names":[...]}   name/grid entries index "names"; grid is sent at tick 0 and whenever a tile changes.
// own = own_state() of meta.ownPlayer: [ammo/1000, ammo slots, super charge, super needed, hero gid (0 = dead)]
// inputs = what player 0 sent since the previous frame: [[type, world x, world y, auto-aim target gid], ...]
//          (type 0 attack, 1 super, 2 move, 3 stop, 8 gadget; x,y = move target / aim point in world units)
// Frames from tools/bsenv.py may carry an 8th element: the agent's per-step note (dict), shown by the viewer.
// type = LogicGameObjectServer vtable slot 0x20 getType(): 0 character, 1 projectile, 2 area effect, 3 item
// LogicTileMap (server+0x48): +0 LogicTile**, +0x68 width, +0x6c height; LogicTile +0 = current LogicTileData*
// (destroyTile swaps it to the Open tile). Positions are logic units, 300 per tile.
// The same frame writer feeds serve mode (one frame per protocol line). Name indices stay valid across battles
// because the data tables are loaded once; g_prev_tiles is reset per battle so the first frame carries the grid.
static FILE *g_rec;
static void *g_names[1024];
static int g_nnames, g_names_sent, g_rec_frames;
static void **g_prev_tiles;

static int name_idx(void *data) {
    for (int i = 0; i < g_nnames; i++) if (g_names[i] == data) return i;
    if (g_nnames == 1024) return -1;
    g_names[g_nnames] = data;
    return g_nnames++;
}

static void write_frame(FILE *f, char *server, int own_pidx) {
    int (*team_score)(void*, int, int) = FN(int(*)(void*, int, int), OFF_LBMS_getTeamScore);
    char *tm = *(char**)(server + 0x48);
    void **tiles = *(void***)tm;
    int n_tiles = *(int*)(tm + 0x68) * *(int*)(tm + 0x6c);

    fprintf(f, "[%d,%d,%d,[", FN(int(*)(void*), OFF_LBMS_getTick)(server), team_score(server, 0, 0), team_score(server, 1, 0));
    ArrayList *objs = *(ArrayList**)server;
    for (int i = 0; i < objs->size; i++) {
        char *o = ((char**)objs->data)[i];
        int *w = (int*)o;
        int type = (*(int(***)(void*))o)[0x20 / 4](o);
        char *pl = type == 0 ? player_at(server, w[0x28/4]) : NULL;      // gems carried: heroes only (not summons)
        int gems = pl && *(int*)(pl + 0x10) == w[0x4/4] ? *(int*)(pl + 0x20) : -1;
        fprintf(f, "%s[%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d]", i ? "," : "", w[0x4/4], type, name_idx(*(void**)(o + 8)),
                obj_x(o), obj_y(o), w[0x24/4], w[0x2c/4], w[0x28/4], type == 0 ? w[0x90/4] : -1, type == 0 ? w[0x94/4] : -1,
                gems);
    }
    fprintf(f, "],");
    int changed = !g_prev_tiles;
    if (!g_prev_tiles) g_prev_tiles = calloc(n_tiles, sizeof(void*));
    for (int i = 0; i < n_tiles && !changed; i++) changed = *(void**)tiles[i] != g_prev_tiles[i];
    if (!changed) fprintf(f, "0");
    else {
        fprintf(f, "[");
        for (int i = 0; i < n_tiles; i++) {
            g_prev_tiles[i] = *(void**)tiles[i];
            fprintf(f, "%s%d", i ? "," : "", name_idx(g_prev_tiles[i]));
        }
        fprintf(f, "]");
    }
    int own[5];
    own_state(server, own_pidx, own);
    // Gem Grab win countdown (as in vec_obs): ticks left and the leading team, while the leader still has enough gems
    int tick = FN(int(*)(void*), OFF_LBMS_getTick)(server), cd_end = *(int*)(server + 0x98), need = *(int*)(server + 0xac);
    int s0 = team_score(server, 0, 0), s1 = team_score(server, 1, 0), lead = s0 > s1 ? 0 : s1 > s0 ? 1 : -1;
    int cd_on = *(int*)(server + 0xa4) == 0 && cd_end > 0 && lead >= 0 && (lead ? s1 : s0) >= need && cd_end > tick;
    fprintf(f, ",[%d,%d,%d,%d,%d,%d,%d],[", own[0], own[1], own[2], own[3], own[4], cd_on ? cd_end - tick : 0,
            cd_on ? lead : -1);
    for (int i = 0; i < g_nsent; i++)
        fprintf(f, "%s[%d,%d,%d,%d]", i ? "," : "", g_sent[i][0], g_sent[i][1], g_sent[i][2], g_sent[i][3]);
    g_nsent = 0;
    fprintf(f, "]]");
}

static void rec_frame(char *server) {
    if (g_rec_frames++) fprintf(g_rec, ",\n");
    write_frame(g_rec, server, 0);
}

static int rec_open(const char *path, char *server, int argc, char **argv) {
    if (!(g_rec = fopen(path, "w"))) { printf("[rec] cannot open %s\n", path); return -1; }
    char *tm = *(char**)(server + 0x48);
    fprintf(g_rec, "{\"meta\":{\"loc\":\"%s\",\"hero\":\"%s\",\"seed\":%s,\"tileSize\":300,\"w\":%d,\"h\":%d,"
            "\"ownPlayer\":0,\"tickRate\":20},\n\"frames\":[\n", kv(argc, argv, "loc", "Gemgrab1"),
            kv(argc, argv, "hero", "ShotgunGirl"), kv(argc, argv, "seed", "12345"), *(int*)(tm + 0x68), *(int*)(tm + 0x6c));
    return 0;
}

static void rec_close(void) {
    fprintf(g_rec, "\n],\n\"names\":[");
    for (int i = 0; i < g_nnames; i++) fprintf(g_rec, "%s\"%s\"", i ? "," : "", data_name(g_names[i]));
    fprintf(g_rec, "]}\n");
    fclose(g_rec);
    printf("[rec] wrote %d frames, %d names\n", g_rec_frames, g_nnames);
}

// One simulation tick in the order of the client's offline loop BattleMode::update @0x8c6e90.
// Player inputs (handleClientInput) go between preTick and executeOneTick, where executeOfflineInputMessages runs them.
// One game tick; agent j (player pidx[j], controller ctl[j]) gets action a[j] first (NULL = no new input).
static void sim_tick_agents(char *server, int k, const int *pidx, const Action *const *a, Ctl *const *ctl) {
    FN(void(*)(void*), OFF_LBMS_preTick)(server);
    for (int j = 0; j < k; j++) if (a[j]) apply_action(server, pidx[j], a[j], ctl[j]);
    FN(void(*)(void*), OFF_LBMS_executeOfflineInputMessages)(server);
    FN(void(*)(void*, void*, void*), OFF_LBMS_executeOneTick)(server, NULL, NULL);
}

static void sim_tick(char *server, const Action *a, Ctl *ctl) {   // player 0 only
    static const int p0 = 0;
    sim_tick_agents(server, 1, &p0, &a, &ctl);
}

static int cmd_m3(const char *root, int argc, char **argv) {
    int (*get_tick)(void*)          = FN(int(*)(void*), OFF_LBMS_getTick);
    int (*game_over_set)(void*)     = FN(int(*)(void*), OFF_LBMS_isGameOverSet);
    int (*game_over_or_reset)(void*) = FN(int(*)(void*), OFF_LBMS_isGameOverOrRoundResetting);
    int (*team_score)(void*, int, int) = FN(int(*)(void*, int, int), OFF_LBMS_getTeamScore);

    char *server = cmd_m2(root, argc, argv);
    if (!server) return 1;
    // trap=0xOFF: plant an ARM `udf` at libg+OFF (our own process) -> SIGILL handler prints the call chain
    const char *trap = kv(argc, argv, "trap", NULL);
    if (trap) {
        uintptr_t at = g_base + strtoul(trap, NULL, 0), page = at & ~(uintptr_t)0xfff;
        mprotect((void*)page, 0x2000, PROT_READ | PROT_WRITE | PROT_EXEC);
        *(volatile uint32_t*)at = 0xe7f000f0;
        __builtin___clear_cache((char*)at, (char*)at + 4);
        printf("[m3] trap planted at libg+%s\n", trap);
    }
    int max_ticks = atoi(kv(argc, argv, "ticks", "6000")), every = atoi(kv(argc, argv, "every", "200"));
    // act=script: player 0 runs script_policy, deciding every think=<ticks> (default 4); otherwise it idles
    int script = !strcmp(kv(argc, argv, "act", "none"), "script"), think = atoi(kv(argc, argv, "think", "4"));
    Ctl ctl = {0};
    Action act;
    int peek_off = (int)strtol(kv(argc, argv, "peek", "-1"), NULL, 0), peek_n = atoi(kv(argc, argv, "peekn", "300"));
    int peek_last = 0x7fffffff;

    const char *rec = kv(argc, argv, "rec", NULL);
    if (rec && rec_open(rec, server, argc, argv)) return 1;
    if (g_rec) rec_frame(server);

    printf("[m3] tick %d: rss=%ldKB\n", get_tick(server), rss_kb());
    dump_characters(server);
    double t0 = now_s(), sim = 0;
    int n = 0;
    for (; n < max_ticks; n++) {
        double a = now_s();
        int decide = script && n % think == 0;
        if (decide) script_policy(server, 0, &act);
        sim_tick(server, decide ? &act : NULL, &ctl);
        sim += now_s() - a;
        if (g_rec) rec_frame(server);
        if (peek_off >= 0 && n < peek_n) {       // peek=0xOFF: report changes of the int at server+OFF
            int v = *(int*)(server + peek_off);
            if (v != peek_last) printf("[peek] tick %d server+0x%x = %d (0x%x)\n", get_tick(server), peek_off, v, v);
            peek_last = v;
        }
        if (game_over_set(server)) { n++; break; }
        if (every > 0 && (n + 1) % every == 0) {
            int own[5];
            own_state(server, 0, own);
            printf("[m3] tick %d: score %d-%d over/reset=%d objs=%d rss=%ldKB | own ammo=%d/%d super=%d/%d\n",
                   get_tick(server), team_score(server, 0, 0), team_score(server, 1, 0), game_over_or_reset(server),
                   (*(ArrayList**)server)->size, rss_kb(), own[0], own[1] * 1000, own[2], own[3]);
            dump_characters(server);
        }
    }
    double wall = now_s() - t0;
    printf("[m3] stopped after %d ticks: tick=%d gameOverSet=%d score %d-%d rss=%ldKB inputs=%d\n", n, get_tick(server),
           game_over_set(server), team_score(server, 0, 0), team_score(server, 1, 0), rss_kb(), ctl.seq);
    dump_characters(server);
    printf("[m3] speed: %.0f ticks/s pure sim (%.3f ms/tick), %.0f ticks/s incl. printing\n", n / sim,
           1e3 * sim / n, n / wall);
    if (g_rec) rec_close();
    int ok = get_tick(server) > 0 && game_over_set(server);
    printf("[m3] RESULT: %s\n", ok ? "M3 PASS — ran to game over" : "M3 INCOMPLETE — no game over yet");
    return ok ? 0 : 1;
}

// ---- reward pieces (RL step 3), shared by vec and serve mode ----
// Every step reports RC_DIM separate numbers for the own player (pidx 0) and its team; the host weights them
// (tools/bsvec.py REWARD_WEIGHTS), so shaping can be changed without touching the driver. Signs are natural (a death
// is +1 in "death"); the weights carry the sign. Per step:
//   win           +1 / -1 / 0 at game over (by final team scores), else 0
//   gem_diff      change of (own team gems - enemy team gems)             (sums to the final difference)
//   countdown     change of the win-countdown potential: +progress while the own team's countdown runs, -progress
//                 while the enemy's runs, 0 otherwise (progress 0 -> 1); sums to +-1 when a countdown ends the game
//   gems_carried  change of the gems the own brawler carries (pickups +, drops on death -)
//   kill, death   own brawler's kills / deaths this step (LogicPlayer stats)
//   dmg_dealt     own damage to enemy brawlers / 1000 (LogicPlayer stat)
//   dmg_taken     own hp lost, in lives and weighted by how low it already was: losing hp from h0 to h1 (fractions,
//                 dead = 0) costs F(h0) - F(h1), F(h) = 1.5h - h^2/2, i.e. 0.5 per life at full hp rising to 1.5 near
//                 zero; full hp -> dead is exactly 1. Healing is not rewarded.
//   low_hp        seconds of the step spent alive below LOW_HP of max hp (sampled at the step end)
enum { RC_WIN, RC_GEM_DIFF, RC_COUNTDOWN, RC_GEMS_CARRIED, RC_KILL, RC_DEATH, RC_DMG_DEALT, RC_DMG_TAKEN, RC_LOW_HP,
       RC_DIM };
#define RC_NAMES "win,gem_diff,countdown,gems_carried,kill,death,dmg_dealt,dmg_taken,low_hp"
#define LOW_HP 0.1f
typedef struct { int tick, gem_diff, carried, kills, deaths, dmg; float cd, hp; } RewState;

static float hp_potential(float h) { return 1.5f * h - 0.5f * h * h; }

static void rew_read(char *s, int team, int pidx, RewState *r) {
    int (*score)(void*, int, int) = FN(int(*)(void*, int, int), OFF_LBMS_getTeamScore);
    int own = score(s, team, 0), opp = score(s, 1 - team, 0), tick = FN(int(*)(void*), OFF_LBMS_getTick)(s);
    int cd_end = *(int*)(s + 0x98), cd_len = 20 * *(int*)(s + 0xb0), need = *(int*)(s + 0xac);
    float prog = *(int*)(s + 0xa4) == 0 && cd_end > 0 && cd_len > 0 ? 1 - (float)(cd_end - tick) / cd_len : 0;   // Gem Grab only
    prog = prog < 0 ? 0 : prog > 1 ? 1 : prog;
    char *p = player_at(s, pidx), *c = player_char(s, pidx);
    r->tick = tick;
    r->gem_diff = own - opp;
    r->cd = own > opp && own >= need ? prog : opp > own && opp >= need ? -prog : 0;
    r->carried = p ? *(int*)(p + 0x20) : 0;
    r->kills = p ? FN(int(*)(void*), OFF_LP_getKills)(p) : 0;
    r->deaths = p ? FN(int(*)(void*), OFF_LP_getDeaths)(p) : 0;
    r->dmg = p ? FN(int(*)(void*), OFF_LP_getDamageDoneToBrawlers)(p) : 0;
    int hp = c ? *(int*)(c + 0x90) : 0, max = c ? *(int*)(c + 0x94) : 0;
    r->hp = c && max > 0 ? (hp > 0 ? (float)hp / max : 0) : -1;     // -1 = no hero (dead / respawning)
}

// Fills out[RC_DIM] for player pidx's step since *prev and makes the current state the new *prev. over: game over now.
// win, gem_diff and countdown are the team's; the other pieces are the player's own.
static void rew_step(char *s, int team, int pidx, RewState *prev, int over, float *out) {
    RewState cur;
    rew_read(s, team, pidx, &cur);
    memset(out, 0, RC_DIM * sizeof *out);
    out[RC_WIN] = over ? (cur.gem_diff > 0) - (cur.gem_diff < 0) : 0;
    out[RC_GEM_DIFF] = cur.gem_diff - prev->gem_diff;
    out[RC_COUNTDOWN] = cur.cd - prev->cd;
    out[RC_GEMS_CARRIED] = cur.carried - prev->carried;
    out[RC_KILL] = cur.kills - prev->kills;
    out[RC_DEATH] = cur.deaths - prev->deaths;
    out[RC_DMG_DEALT] = (cur.dmg - prev->dmg) / 1000.f;
    float h0 = prev->hp, h1 = cur.hp < 0 ? 0 : cur.hp;
    out[RC_DMG_TAKEN] = h0 >= 0 && h1 < h0 ? hp_potential(h0) - hp_potential(h1) : 0;
    out[RC_LOW_HP] = cur.hp >= 0 && cur.hp < LOW_HP ? (cur.tick - prev->tick) / 20.f : 0;
    *prev = cur;
}

// ---- serve: step-wise control over stdin/stdout (used by tools/bsenv.py and tools/play.py) ----
// Loads the data tables once, then reads one command per line:
//   reset [key=value...]   (re)build a battle; keys as for m2, falling back to the serve command line's keys.
//                          agents=<k>: we steer player 0 and its first k-1 teammates; enemies=<m>: also the first m
//                          players of the other team. Those are the rows of step and obs, in this order (meta "rows" =
//                          their player indices); the other players stay bots.
//   step <repeat> <row>... per row 9 numbers <mx> <my> <atk> <ax> <ay> <sup> <sx> <sy> <gadget> (an Action), or "-" =
//                          no input this step (as between an agent's decisions); rows left out stand still. Runs
//                          <repeat> ticks (presses only on the first tick). Rows of team 1 are in its turned-around
//                          frame (MIRROR_*), as obs shows it to them. rc stays player 0's.
//   obs                    the observation of every row as vec mode builds it (OBS_BLOCKS; team 1 turned around):
//                          {"obs":[[OBS_DIM numbers],...],"grid":["<GRID*GRID tile flag bytes as hex>",...]}.
//                          Velocities are measured since the previous obs: ask once per decision, as training did.
//   quit
// and answers reset/step with one JSON line: {"frame":<recorder frame>,"names":[[idx,"name"],...new...],"done":0|1,
// "rc":[reward pieces of this step, RC_NAMES order]} (reset adds "meta" with "rc_names", "rows" and "obs_layout"; its rc
// is all zero). Lines not starting with '{' may come from libg's own logging and are to be ignored.
static void serve_reply(char *server, const char *meta, const float *rc) {
    printf("{%s\"rc\":[", meta ? meta : "");
    for (int i = 0; i < RC_DIM; i++) printf(i ? ",%g" : "%g", rc[i]);
    printf("],\"frame\":");
    write_frame(stdout, server, 0);
    printf(",\"names\":[");
    for (int i = g_names_sent; i < g_nnames; i++) printf("%s[%d,\"%s\"]", i > g_names_sent ? "," : "", i, data_name(g_names[i]));
    g_names_sent = g_nnames;
    printf("],\"done\":%d}\n", FN(int(*)(void*), OFF_LBMS_isGameOverSet)(server));
    fflush(stdout);
}

// ---- vec: N battles in one process behind a binary protocol (tools/bsvec.py), for RL training ----
// driver <libdir> <assets> vec [n=16] [agents=1] [repeat=4] [seed=1] [skip=120] [maxticks=12000] [vision=0]
//                            [cpus=<hex mask>] [bench=0] [battle keys as for m2]
// n = rows; agents=<k> (1..3) brawlers we steer per battle (player 0 + its first k-1 teammates, the rest stay bots),
// so n/k battles; row r = agent r % k of battle r / k, each with its own observation, action and reward pieces
// (v5; with agents=1 rows are battles exactly as in v4).
// bench=<batches> [policy=random|idle|script] times the stepping on the device without the protocol.
// All fields are little-endian 32-bit ints/floats. stdout carries only protocol bytes; everything printed (table
// loading, libg logging, the fault handler) goes to stderr.
//   driver -> host at start:  int32 hello[11] = {VEC_MAGIC, VEC_VERSION, n, OBS_DIM, ACT_DIM, INFO_DIM, repeat, GRID,
//                             layout_len, RC_DIM, names_len}, the observation layout string (layout_len bytes, see
//                             obs_layout()), the reward piece names (RC_NAMES, names_len bytes),
//                             then a batch reply with the first observation of every battle
//   host -> driver:           int32 req[4] = {VEC_MAGIC, cmd, arg, count}
//       VEC_STEP   arg = ticks per action (<= 0: the default repeat), count = c, then int32 start, then
//                  float32 act[c][ACT_DIM]: steps battles start .. start+c-1 (any range; the host pipelines two
//                  halves so the driver simulates one while the other's reply/actions cross USB)
//       VEC_RESET  arg = base seed (battle i gets seed+i; -1 = every battle's next seed), count = 0
//                  (VEC_STEP ranges are rows and must cover whole battles: start and c multiples of agents)
//       VEC_QUIT
//   driver -> host per batch: int32 hdr[4] = {VEC_MAGIC, c, busy_us, reset_us}, float32 reward[c],
//                             float32 rc[c][RC_DIM], int32 info[c][INFO_DIM], float32 obs[c][OBS_DIM],
//                             uint8 grid[c][GRID*GRID]   (c = the stepped range; all n battles at start/reset)
// reward = the "win" piece only (+1/-1/0 at game over); rc = all reward pieces of the step (see rew_step()), which
// the host weights into the training reward. On the step a battle ends, rc belongs to the finished battle.
// act row = mx, my, attack (0 none/1 auto/2 aimed), ax, ay, super (same), sx, sy, gadget (> 0.5), unused: the
// fields of Action. Buttons press on the first tick of the step, movement is sent then and every STEER_TICKS.
// A battle that ends (game over = terminated, tick >= maxticks = truncated) is rebuilt at once with seed += n/agents;
// its rows then carry the done flag, final tick/score/result in info and the first observation of the new battle.
// busy_us = time spent simulating + resetting for this batch, reset_us = the resetting part.
#define VEC_MAGIC   0x31565342   // "BSV1"
#define VEC_VERSION 5
enum { VEC_STEP = 1, VEC_RESET = 2, VEC_QUIT = 3 };
enum { INFO_FLAGS, INFO_TICK, INFO_SCORE_OWN, INFO_SCORE_OPP, INFO_SEED, INFO_RESULT, INFO_DIM };
#define FLAG_TERMINATED 1
#define FLAG_TRUNCATED  2
#define ACT_DIM 10
// Observation v2: one float32 vector + a uint8 tile grid per battle, egocentric (relative to the own hero, or to where
// it died while respawning). Blocks and field order are OBS_BLOCKS (sent to the host as the layout string); vec_obs()
// writes them in exactly this order. Scales: positions relative / 3000 (10 tiles), absolute x/W, y/H; velocities in
// logic units per tick / 40; hp as fraction + max hp / 5000; ranges / 3000; speed / 1000; time and timers as fractions.
// Fog of war: an enemy counts as visible when the game's own check LogicCharacterServer::isTargetVisibleToAttack(viewer,
// enemy, true) says so (bushes + team vision, no line-of-sight test; viewer = own hero, else a living ally). Hidden
// enemies keep their last seen position/hp/gems (seen_age > 0) and zero velocity. alive/respawn/brawler stats stay
// known, as on the in-game scoreboard. Allies are always visible. Players are slotted allies first, then enemies, by
// player index (3v3: 2 + 3; other player counts are cut to these slots). Assumes we play team 0's side.
// Items, projectiles and area effects are listed only where the team can see them
// (LogicGameObjectManagerServer::isPositionVisibleToPlayerTeam: inside a team member's visibility rect, ~+-14 tiles
// wide, +11/-9 tall, and not hidden in a bush unless close), as the game's own vision encoding does; the gem mine is
// always known.
//   hero = 1 + LogicData instance id of the brawler in characters.csv (categorical; 0 = unknown)
//   item rows: nearest type-3 objects except the gem mine (gems: z > 0 while flying out of the mine)
//   projectile rows: nearest projectiles with their target point (straight shots fly start -> target)
//   area rows: nearest area effects (spawn effects, clouds, ...)
// grid: GRID x GRID tiles centred on the own hero's tile, row-major (rows = y), each a bit set of TILE_* flags.
#define GRID 21
enum { TILE_BLOCKS_MOVE = 1, TILE_BLOCKS_SHOTS = 2, TILE_BUSH = 4, TILE_DESTRUCTIBLE = 8, TILE_OUTSIDE = 16,
       TILE_SPECIAL = 32 };   // special = damage / healing / speed tiles
#define N_ALLY 2
#define N_ENEMY 3
#define N_ITEM 8
#define N_PROJ 12
#define N_AREA 4
#define MAX_PLAYERS 16
static const struct { const char *name; int count; const char *fields; } OBS_BLOCKS[] = {
    {"global", 1, "time,own_score,enemy_score,own_countdown,enemy_countdown,seen_items,mine,mine_dx,mine_dy"},
    {"self", 1, "alive,x,y,vx,vy,hp,max_hp,ammo,ammo_max,super,super_ready,gems,respawn,range,super_range,"
                "face_x,face_y,speed,hero"},
    {"ally", N_ALLY, "present,alive,visible,dx,dy,seen_age,vx,vy,hp,max_hp,gems,respawn,range,speed,hero"},
    {"enemy", N_ENEMY, "present,alive,visible,dx,dy,seen_age,vx,vy,hp,max_hp,gems,respawn,range,speed,hero"},
    {"item", N_ITEM, "present,dx,dy,z"},
    {"projectile", N_PROJ, "present,dx,dy,target_dx,target_dy,enemy"},
    {"area", N_AREA, "present,dx,dy,enemy"},
};
#define OBS_DIM (9 + 19 + (N_ALLY + N_ENEMY) * 15 + N_ITEM * 4 + N_PROJ * 6 + N_AREA * 4)

// "global:1:time,own_score,...;self:1:alive,...;..." + ";grid:17:<flag names>"; checks OBS_DIM against the table.
static const char *obs_layout(void) {
    static char buf[2048];
    int len = 0, dim = 0;
    for (size_t b = 0; b < sizeof OBS_BLOCKS / sizeof *OBS_BLOCKS; b++) {
        int nf = 1;
        for (const char *c = OBS_BLOCKS[b].fields; *c; c++) nf += *c == ',';
        dim += nf * OBS_BLOCKS[b].count;
        len += snprintf(buf + len, sizeof buf - len, "%s:%d:%s;", OBS_BLOCKS[b].name, OBS_BLOCKS[b].count, OBS_BLOCKS[b].fields);
    }
    snprintf(buf + len, sizeof buf - len, "grid:%d:blocks_move,blocks_shots,bush,destructible,outside,special", GRID);
    return dim == OBS_DIM ? buf : NULL;
}

// Tile flags by LogicTileData, from csv_logic/tiles.csv columns (looked up by tile name once per tile type).
static struct { char name[48]; int flags; } g_tile_rows[128];
static int g_ntile_rows;
static struct { void *data; int flags; } g_tile_cache[128];
static int g_ntile_cache;

static int csv_split(char *line, char **out, int max) {   // splits in place, strips quotes
    int n = 0;
    for (char *p = line; n < max;) {
        int q = *p == '"';
        if (q) p++;
        out[n++] = p;
        char *end = q ? strchr(p, '"') : p + strcspn(p, ",\r\n");
        if (!end) break;
        char *next = q ? end + 1 : end, sep = *next;
        *end = 0;
        if (sep != ',') break;
        p = next + 1;
    }
    return n;
}

static int load_tile_flags(const char *root) {
    char path[512];
    snprintf(path, sizeof path, "%s/csv_logic/tiles.csv", root);
    long n;
    char *buf = read_file(path, &n), *line, *save = NULL, *f[64];
    if (!buf) return -1;
    int col[7] = {-1, -1, -1, -1, -1, -1, -1};
    const char *want[7] = {"Name", "BlocksMovement", "BlocksProjectiles", "HidesHero", "IsDestructible", "Damage", "SpeedChange"};
    int row = 0;
    for (line = strtok_r(buf, "\n", &save); line; line = strtok_r(NULL, "\n", &save), row++) {
        int nf = csv_split(line, f, 64);
        if (row == 0) {
            for (int i = 0; i < nf; i++) for (int k = 0; k < 7; k++) if (!strcmp(f[i], want[k])) col[k] = i;
            continue;
        }
        if (row == 1 || col[0] < 0 || col[0] >= nf || g_ntile_rows == 128) continue;   // row 1 = column types
        int flags = 0;
#define COL(k) (col[k] >= 0 && col[k] < nf ? f[col[k]] : "")
        if (!strcmp(COL(1), "true")) flags |= TILE_BLOCKS_MOVE;
        if (!strcmp(COL(2), "true")) flags |= TILE_BLOCKS_SHOTS;
        if (!strcmp(COL(3), "true")) flags |= TILE_BUSH;
        if (!strcmp(COL(4), "true")) flags |= TILE_DESTRUCTIBLE;
        if (atoi(COL(5)) || atoi(COL(6))) flags |= TILE_SPECIAL;
#undef COL
        snprintf(g_tile_rows[g_ntile_rows].name, sizeof g_tile_rows[0].name, "%s", f[col[0]]);
        g_tile_rows[g_ntile_rows++].flags = flags;
    }
    free(buf);
    return col[0] >= 0 && col[1] >= 0 && col[2] >= 0 && col[3] >= 0 ? 0 : -1;
}

static int tile_flags(void *data) {
    for (int i = 0; i < g_ntile_cache; i++) if (g_tile_cache[i].data == data) return g_tile_cache[i].flags;
    const char *name = data_name(data);
    int flags = 0;
    for (int i = 0; i < g_ntile_rows; i++) if (!strcmp(g_tile_rows[i].name, name)) { flags = g_tile_rows[i].flags; break; }
    if (g_ntile_cache < 128) { g_tile_cache[g_ntile_cache].data = data; g_tile_cache[g_ntile_cache++].flags = flags; }
    return flags;
}

static int is_gem_mine(void *data) {   // cached by LogicData pointer
    static void *mine, *other[64];
    static int nother;
    if (data == mine) return 1;
    for (int i = 0; i < nother; i++) if (other[i] == data) return 0;
    if (!strcmp(data_name(data), "OrbSpawner")) { mine = data; return 1; }
    if (nother < 64) other[nother++] = data;
    return 0;
}

static int g_proto_fd = -1;   // the real stdout in vec mode (fd 1 is pointed at stderr)

static void die_vec(const char *msg) {
    fprintf(stderr, "[vec] FATAL: %s\n", msg);
    exit(1);
}

// Every tick executeOneTick encodes a ~1.5 KB "vision" network update of all objects for the client (offline server:
// handed to listener->sendOfflineVisionUpdate, which we free unread). That is ~35% of the tick time. vision=0 (default
// in vec mode) jumps over it, in our process's in-memory copy of libg only: the `ldrb r0,[r4,#0xb4]` that starts the
// encoding branch becomes `b <LogicTime::increaseTick call>`. isotest novision=1 checks that battles play out
// identically with and without it.
#define OFF_EOT_vision_start 0x8a2768   // executeOneTick: ldrb r0, [r4, #0xb4]
#define OFF_EOT_vision_end   0x8a2af4   // executeOneTick: mov r0, r9; bl LogicTime::increaseTick
static int skip_vision_encoding(void) {
    volatile uint32_t *at = (uint32_t*)(g_base + OFF_EOT_vision_start);
    uint32_t want = 0xe5d400b4, jump = 0xea000000 | ((OFF_EOT_vision_end - OFF_EOT_vision_start - 8) >> 2);
    if (*at == jump) return 0;
    if (*at != want) { fprintf(stderr, "[driver] vision skip: unexpected instruction 0x%08x\n", *at); return -1; }
    uintptr_t page = (uintptr_t)at & ~(uintptr_t)0xfff;
    if (mprotect((void*)page, 0x1000, PROT_READ | PROT_WRITE | PROT_EXEC)) return -1;
    *at = jump;
    __builtin___clear_cache((char*)at, (char*)at + 4);
    mprotect((void*)page, 0x1000, PROT_READ | PROT_EXEC);
    return 0;
}

typedef struct {   // static facts about a player's brawler, remembered while it is dead
    float range, speed, max_hp, hero;
} HeroInfo;
typedef struct {   // one brawler we steer in a battle
    int pidx;                                 // its player index
    Ctl ctl;
    int lx, ly;                               // its hero's position (last known while dead): the observation's origin
    RewState rew;                             // state at the end of the previous step, for the reward pieces
} Agent;
typedef struct {
    char *server;
    int k;                                    // agents (brawlers we steer), ag[0] = player 0 (vec mode)
    Agent ag[MAX_AGENTS];
    int seed, team;
    int mirror, W, H;                         // how the team's view is turned (MIRROR_*), map size in logic units
    int battle_ticks, respawn_ticks;          // mode's tick limit, 20 * respawn seconds
    int prev_tick, px[MAX_PLAYERS], py[MAX_PLAYERS];   // hero positions at the previous observation (for velocity)
    unsigned char pvalid[MAX_PLAYERS];
    int alive_tick[MAX_PLAYERS];              // last observation tick the player's hero was alive
    int seen_tick[MAX_PLAYERS], sx[MAX_PLAYERS], sy[MAX_PLAYERS], sgems[MAX_PLAYERS];   // last sighting of enemies
    float shp[MAX_PLAYERS];
    HeroInfo hero[MAX_PLAYERS];
} Env;
// n = rows = battles * k: row r is agent r % k of battle r / k
typedef struct { int n, k, repeat, skip, maxticks, argc; char **argv; Env *env; } Vec;

static int read_full(int fd, void *buf, size_t n) {
    for (size_t got = 0; got < n;) {
        ssize_t r = read(fd, (char*)buf + got, n - got);
        if (r <= 0) return -1;
        got += r;
    }
    return 0;
}

static int write_full(int fd, const void *buf, size_t n) {
    for (size_t put = 0; put < n;) {
        ssize_t r = write(fd, (const char*)buf + put, n - put);
        if (r <= 0) return -1;
        put += r;
    }
    return 0;
}

static int team_score_of(char *server, int team) {
    return FN(int(*)(void*, int, int), OFF_LBMS_getTeamScore)(server, team, 0);
}

// Our agents learned the battle from team 0's side (own spawn at the bottom). Team 1 (spawn at the top) gets the battle
// turned around so that its spawn is at the bottom too: rotated by 180 degrees when the map is point-symmetric
// (Gemgrab1 is), else flipped top to bottom. Its observations are written in that frame and its actions turned back.
enum { MIRROR_NONE, MIRROR_ROT180, MIRROR_FLIPY };

static int map_symmetry(char *tm) {
    void **tiles = *(void***)tm;
    int n = *(int*)(tm + 0x68) * *(int*)(tm + 0x6c);
    for (int i = 0; i < n; i++)               // tile (x, y) <-> (w-1-x, h-1-y) is index i <-> n-1-i
        if (*(void**)tiles[i] != *(void**)tiles[n - 1 - i]) return MIRROR_FLIPY;
    return MIRROR_ROT180;
}

static int mir_x(Env *e, int x) { return e->mirror == MIRROR_ROT180 ? e->W - x : x; }
static int mir_y(Env *e, int y) { return e->mirror ? e->H - y : y; }

static void mirror_action(Action *a, int mirror) {   // a direction in the team's frame -> world
    if (mirror == MIRROR_ROT180) { a->mx = -a->mx; a->ax = -a->ax; a->sx = -a->sx; }
    if (mirror) { a->my = -a->my; a->ay = -a->ay; a->sy = -a->sy; }
}

// Sets e up for steering players pidx[0 .. k) of `team` in the battle e->server (built and fast-forwarded already).
static void env_init(Env *e, int team, int k, const int *pidx) {
    char *s = e->server, *tm = *(char**)(s + 0x48);
    memset(e->ag, 0, sizeof e->ag);
    e->k = k;
    e->team = team;
    e->W = *(int*)(tm + 0x68) * 300;
    e->H = *(int*)(tm + 0x6c) * 300;
    e->mirror = team == 1 ? map_symmetry(tm) : MIRROR_NONE;
    for (int j = 0; j < k; j++) {
        Agent *a = &e->ag[j];
        char *c = player_char(s, a->pidx = pidx[j]);
        a->lx = c ? mir_x(e, obj_x(c)) : 0; a->ly = c ? mir_y(e, obj_y(c)) : 0;
    }
    int variation = *(int*)(s + 0xa4);
    e->battle_ticks = FN(int(*)(int), OFF_LGMU_getBattleTicks)(variation);
    e->respawn_ticks = 20 * FN(int(*)(int), OFF_LGMU_getRespawnSeconds)(variation);
    e->prev_tick = 0;
    memset(e->pvalid, 0, sizeof e->pvalid);
    memset(e->alive_tick, 0, sizeof e->alive_tick);
    memset(e->hero, 0, sizeof e->hero);
    for (int i = 0; i < MAX_PLAYERS; i++) e->seen_tick[i] = -1;
    for (int j = 0; j < k; j++) rew_read(s, team, e->ag[j].pidx, &e->ag[j].rew);
}

// (Re)builds e->server with seed e->seed and fast-forwards `skip` ticks (the 3v3 start countdown).
static int vec_build(Vec *v, Env *e) {
    char seed[24], *args[160];
    int na = 0;
    snprintf(seed, sizeof seed, "seed=%d", e->seed);
    args[na++] = seed;                       // kv() takes the first match, so this overrides a seed= on the command line
    for (int i = 0; i < v->argc && na < 160; i++) args[na++] = v->argv[i];
    if (e->server) destroy_battle(e->server);
    memset(e->ag, 0, sizeof e->ag);
    if (!(e->server = build_battle(na, args, 0))) return -1;
    int want = v->k > 0 ? v->k : 1, pidx[MAX_AGENTS], team = player_team(e->server, 0);
    if (team_players(e->server, team, want, pidx) != want) return -1;   // the mode has fewer teammates
    Ctl ctl = {0};
    for (int i = 0; i < v->skip; i++) sim_tick(e->server, NULL, &ctl);
    env_init(e, team, want, pidx);
    return 0;
}

static float hp_frac(char *c) {
    int hp = *(int*)(c + 0x90), max = *(int*)(c + 0x94);
    return max > 0 ? (float)hp / max : 0;
}

typedef struct { char *o; int d; } Near;

// keeps the k nearest (smallest d) in out[0..k), sorted; returns how many
static int nearest(Near *all, int n, Near *out, int k) {
    int m = 0;
    for (int i = 0; i < n; i++) {
        int j = m < k ? m++ : k;
        if (j == k && all[i].d >= out[k - 1].d) continue;
        if (j == k) j = k - 1;
        while (j > 0 && out[j - 1].d > all[i].d) { out[j] = out[j - 1]; j--; }
        out[j] = all[i];
    }
    return m;
}

static void hero_info(char *c, HeroInfo *h) {
    void *data = FN(void*(*)(void*), OFF_LCS_getCharacterData)(c);
    h->range = skill_range(FN(void*(*)(void*), OFF_LCS_getWeaponSkill)(c)) / 3000.f;
    h->speed = *(int*)((char*)data + 0xb0) / 1000.f;
    h->max_hp = *(int*)(c + 0x94) / 5000.f;
    h->hero = FN(int(*)(void*), OFF_LData_getInstanceID)(data) + 1;   // 0 = unknown
}

static float clampf(float v, float lo, float hi) { return v < lo ? lo : v > hi ? hi : v; }

// Fraction of the respawn time still to wait (0 while alive). The game writes LogicPlayer+0x28 = death tick + 5 about
// 5 ticks after the death and brings the hero back at +0x28 + 20*respawn seconds (tickSpawnHeroes starts the 40-tick
// spawn effect before that). Until +0x28 is written for this death, estimate it from the last tick we saw the hero.
static float respawn_frac(Env *e, char *pl, int p, int tick) {
    if (e->respawn_ticks <= 0) return 0;
    int p28 = *(int*)(pl + 0x28);
    int at = (p28 > e->alive_tick[p] ? p28 : e->alive_tick[p] + 5) + e->respawn_ticks;
    return clampf((float)(at - tick) / e->respawn_ticks, 0, 1);
}

// What the game's vision would show the viewer's team at (tx, ty), as isTargetVisibleToAttack does it.
static int pos_visible(char *server, char *viewer, int tx, int ty) {
    if (!viewer) return 0;
    int team = *(int*)(viewer + 0x2c);
    void *members = FN(void*(*)(void*, int), OFF_LBMS_getTeamMemberPositions)(server, team);
    int mirrored = FN(int(*)(int, int), OFF_LBMS_isPlayFieldMirroredForPlayer)(team, *(int*)(server + 0xa4));
    return FN(int(*)(void*, void*, int, int, void*, int, int, int), OFF_LGOMS_isPositionVisibleToPlayerTeam)(
        *(void**)(viewer + 0xc), *(void**)(server + 0x48), obj_x(viewer), obj_y(viewer), members, tx, ty, mirrored);
}

// Observation of agent j. Call vec_remember() once after all agents of the battle (velocities need the old positions).
static void vec_obs(Env *e, int j, float *o, unsigned char *grid) {
    int (*visible)(void*, void*, int) = FN(int(*)(void*, void*, int), OFF_LCS_isTargetVisibleToAttack);
    char *s = e->server, *tm = *(char**)(s + 0x48);
    int tick = FN(int(*)(void*), OFF_LBMS_getTick)(s), tw = *(int*)(tm + 0x68), th = *(int*)(tm + 0x6c);
    float W = tw * 300.f, H = th * 300.f, dt = tick > e->prev_tick ? tick - e->prev_tick : 1;
    ArrayList *players = (ArrayList*)(s + 0x4c);
    int np = players->size < MAX_PLAYERS ? players->size : MAX_PLAYERS;
    char *heroes[MAX_PLAYERS], *viewer = NULL;
    for (int p = 0; p < np; p++) {
        heroes[p] = player_char(s, p);
        if (heroes[p]) hero_info(heroes[p], &e->hero[p]);
        if (heroes[p] && !viewer && *(int*)(player_at(s, p) + 0xc) == e->team) viewer = heroes[p];
    }
    Agent *ag = &e->ag[j];
    int self = ag->pidx;
    char *me = heroes[self];
    if (me) { ag->lx = mir_x(e, obj_x(me)); ag->ly = mir_y(e, obj_y(me)); }
    int ox = ag->lx, oy = ag->ly;

    // one pass over all objects: gem mine, items, projectiles, area effects
    static Near items[256], projs[256], areas[256], near_i[N_ITEM], near_p[N_PROJ], near_a[N_AREA];
    int ni = 0, npj = 0, na = 0;
    char *mine = NULL;
    ArrayList *objs = *(ArrayList**)s;
    for (int i = 0; i < objs->size; i++) {
        char *ob = ((char**)objs->data)[i];
        int type = (*(int(***)(void*))ob)[0x20 / 4](ob);
        if (type == 0) continue;
        int x = obj_x(ob), y = obj_y(ob), dx = mir_x(e, x) - ox, dy = mir_y(e, y) - oy;
        Near nr = {ob, dx * dx + dy * dy};
        if (type == 3 && is_gem_mine(*(void**)(ob + 8))) { mine = ob; continue; }
        if (!pos_visible(s, viewer, x, y)) continue;
        if (type == 3) { if (ni < 256) items[ni++] = nr; }
        else if (type == 1) { if (npj < 256) projs[npj++] = nr; }
        else if (type == 2) { if (na < 256) areas[na++] = nr; }
    }

    float *q = o;
    memset(o, 0, OBS_DIM * sizeof *o);
    // global
    int own_score = team_score_of(s, e->team), opp_score = team_score_of(s, 1 - e->team);
    // Gem Grab win countdown (calculateIsGameOver): server+0x98 = end tick (0 = none), +0xb0 = its seconds, +0xac = gems
    // needed. It is re-checked at the start of the next tick, so it is shown only while the leader still has enough.
    int cd_end = *(int*)(s + 0x98), cd_len = 20 * *(int*)(s + 0xb0), need = *(int*)(s + 0xac);
    float cd = cd_end > 0 && cd_len > 0 ? clampf((float)(cd_end - tick) / cd_len, 0, 1) : 0;
    *q++ = e->battle_ticks > 0 ? (float)tick / e->battle_ticks : 0;
    *q++ = own_score / 10.f;
    *q++ = opp_score / 10.f;
    *q++ = own_score > opp_score && own_score >= need ? cd : 0;
    *q++ = opp_score > own_score && opp_score >= need ? cd : 0;
    *q++ = ni / 10.f;
    *q++ = mine != NULL;
    *q++ = mine ? (mir_x(e, obj_x(mine)) - ox) / 3000.f : 0;
    *q++ = mine ? (mir_y(e, obj_y(mine)) - oy) / 3000.f : 0;

    // self
    int own[5];
    own_state(s, self, own);
    char *p0 = player_at(s, self);
    HeroInfo *hi = &e->hero[self];
    *q++ = me != NULL;
    *q++ = ox / W;
    *q++ = oy / H;
    *q++ = me && e->pvalid[self] ? (ox - e->px[self]) / dt / 40.f : 0;
    *q++ = me && e->pvalid[self] ? (oy - e->py[self]) / dt / 40.f : 0;
    *q++ = me ? hp_frac(me) : 0;
    *q++ = hi->max_hp;
    *q++ = own[0] / 3000.f;
    *q++ = own[1] / 3.f;
    *q++ = own[3] > 0 ? clampf((float)own[2] / own[3], 0, 1) : 0;
    *q++ = own[3] > 0 && own[2] >= own[3];
    *q++ = p0 ? *(int*)(p0 + 0x20) / 10.f : 0;
    *q++ = p0 && !me ? respawn_frac(e, p0, self, tick) : 0;
    *q++ = hi->range;
    *q++ = me ? skill_range(FN(void*(*)(void*), OFF_LCS_getUltiSkill)(me)) / 3000.f : 0;
    *q++ = e->mirror == MIRROR_ROT180 ? -ag->ctl.fx : ag->ctl.fx;   // facing is kept in world directions
    *q++ = e->mirror ? -ag->ctl.fy : ag->ctl.fy;
    *q++ = hi->speed;
    *q++ = hi->hero;

    // other players: allies, then enemies
    for (int enemy = 0; enemy < 2; enemy++) {
        int k = 0, slots = enemy ? N_ENEMY : N_ALLY;
        for (int p = 0; p < np && k < slots; p++) {
            if (p == self) continue;
            char *pl = player_at(s, p), *h = heroes[p];
            if ((*(int*)(pl + 0xc) != e->team) != enemy) continue;
            int vis = h && (!enemy || (viewer && visible(viewer, h, 1)));
            int gems = *(int*)(pl + 0x20);
            if (vis) { e->seen_tick[p] = tick; e->sx[p] = mir_x(e, obj_x(h)); e->sy[p] = mir_y(e, obj_y(h)); e->shp[p] = hp_frac(h); e->sgems[p] = gems; }
            int seen = e->seen_tick[p] >= 0;
            *q++ = 1;
            *q++ = h != NULL;
            *q++ = vis;
            *q++ = seen ? (e->sx[p] - ox) / 3000.f : 0;
            *q++ = seen ? (e->sy[p] - oy) / 3000.f : 0;
            *q++ = vis ? 0 : seen ? clampf((tick - e->seen_tick[p]) / 200.f, 0, 1) : 1;
            *q++ = vis && e->pvalid[p] ? (e->sx[p] - e->px[p]) / dt / 40.f : 0;
            *q++ = vis && e->pvalid[p] ? (e->sy[p] - e->py[p]) / dt / 40.f : 0;
            *q++ = seen ? e->shp[p] : 0;
            *q++ = e->hero[p].max_hp;
            *q++ = (enemy ? (seen ? e->sgems[p] : 0) : gems) / 10.f;
            *q++ = h ? 0 : respawn_frac(e, pl, p, tick);
            *q++ = e->hero[p].range;
            *q++ = e->hero[p].speed;
            *q++ = e->hero[p].hero;
            k++;
        }
        q += (slots - k) * 15;                 // unused slots stay zero
    }

    // items, projectiles, area effects
    int m = nearest(items, ni, near_i, N_ITEM);
    for (int i = 0; i < N_ITEM; i++, q += 4) if (i < m) {
        char *ob = near_i[i].o;
        q[0] = 1; q[1] = (mir_x(e, obj_x(ob)) - ox) / 3000.f; q[2] = (mir_y(e, obj_y(ob)) - oy) / 3000.f; q[3] = *(int*)(ob + 0x24) / 1000.f;
    }
    m = nearest(projs, npj, near_p, N_PROJ);
    for (int i = 0; i < N_PROJ; i++, q += 6) if (i < m) {
        char *ob = near_p[i].o;
        q[0] = 1; q[1] = (mir_x(e, obj_x(ob)) - ox) / 3000.f; q[2] = (mir_y(e, obj_y(ob)) - oy) / 3000.f;
        q[3] = (mir_x(e, *(int*)(ob + 0x4c)) - ox) / 3000.f; q[4] = (mir_y(e, *(int*)(ob + 0x50)) - oy) / 3000.f;
        q[5] = *(int*)(ob + 0x2c) != e->team;
    }
    m = nearest(areas, na, near_a, N_AREA);
    for (int i = 0; i < N_AREA; i++, q += 4) if (i < m) {
        char *ob = near_a[i].o;
        q[0] = 1; q[1] = (mir_x(e, obj_x(ob)) - ox) / 3000.f; q[2] = (mir_y(e, obj_y(ob)) - oy) / 3000.f; q[3] = *(int*)(ob + 0x2c) != e->team;
    }
    if (q - o != OBS_DIM) die_vec("observation layout mismatch");

    // tile grid around the own hero's tile (in the team's frame: tile (tx, ty) there is real tile (rx, ry))
    void **tiles = *(void***)tm;
    int cx = ox / 300, cy = oy / 300;
    for (int gy = 0; gy < GRID; gy++)
        for (int gx = 0; gx < GRID; gx++) {
            int tx = cx - GRID / 2 + gx, ty = cy - GRID / 2 + gy;
            int rx = e->mirror == MIRROR_ROT180 ? tw - 1 - tx : tx, ry = e->mirror ? th - 1 - ty : ty;
            grid[gy * GRID + gx] = tx < 0 || ty < 0 || tx >= tw || ty >= th
                ? TILE_OUTSIDE | TILE_BLOCKS_MOVE | TILE_BLOCKS_SHOTS : tile_flags(*(void**)tiles[ry * tw + rx]);
        }

}

// After the observations of a step: remember hero positions for the next velocities.
static void vec_remember(Env *e) {
    char *s = e->server;
    ArrayList *players = (ArrayList*)(s + 0x4c);
    int np = players->size < MAX_PLAYERS ? players->size : MAX_PLAYERS, tick = FN(int(*)(void*), OFF_LBMS_getTick)(s);
    for (int p = 0; p < np; p++) {
        char *h = player_char(s, p);
        e->pvalid[p] = h != NULL;
        if (h) { e->px[p] = mir_x(e, obj_x(h)); e->py[p] = mir_y(e, obj_y(h)); e->alive_tick[p] = tick; }
    }
    e->prev_tick = tick;
}

// Observations of all agents of a battle into rows o[0 .. k), grid[0 .. k).
static void vec_obs_all(Env *e, float *o, unsigned char *grid) {
    for (int j = 0; j < e->k; j++) vec_obs(e, j, o + j * OBS_DIM, grid + j * GRID * GRID);
    vec_remember(e);
}

static void vec_info(Env *e, int *info, int flags) {
    int own = team_score_of(e->server, e->team), opp = team_score_of(e->server, 1 - e->team);
    info[INFO_FLAGS] = flags;
    info[INFO_TICK] = FN(int(*)(void*), OFF_LBMS_getTick)(e->server);
    info[INFO_SCORE_OWN] = own;
    info[INFO_SCORE_OPP] = opp;
    info[INFO_SEED] = e->seed;
    info[INFO_RESULT] = flags ? (own > opp) - (own < opp) : 0;
}

static int clamp_btn(float f) { int b = (int)lrintf(f); return b < 0 ? 0 : b > 2 ? 2 : b; }

typedef struct { int *hdr; float *rew, *rc; int *info; float *obs; unsigned char *grid; size_t size; } Batch;

static Batch batch_alloc(int n) {
    Batch b;
    int words = 4 + n + n * RC_DIM + n * INFO_DIM + n * OBS_DIM;
    b.size = words * 4 + n * GRID * GRID;
    char *p = calloc(1, b.size);
    b.hdr = (int*)p;
    b.rew = (float*)(b.hdr + 4);
    b.rc = b.rew + n;
    b.info = (int*)(b.rc + n * RC_DIM);
    b.obs = (float*)(b.info + n * INFO_DIM);
    b.grid = (unsigned char*)(p + words * 4);
    b.hdr[0] = VEC_MAGIC; b.hdr[1] = n;
    return b;
}

// rows start .. start+count-1 of the batch: header + the matching slice of every section
static int write_batch(Batch *b, int start, int count) {
    b->hdr[1] = count;
    struct iovec io[6] = {
        {b->hdr, 16},
        {b->rew + start, count * sizeof *b->rew},
        {b->rc + start * RC_DIM, count * RC_DIM * sizeof *b->rc},
        {b->info + start * INFO_DIM, count * INFO_DIM * sizeof *b->info},
        {b->obs + start * OBS_DIM, count * OBS_DIM * sizeof *b->obs},
        {b->grid + start * GRID * GRID, count * GRID * GRID}};
    int k = 0;
    while (k < 6) {
        ssize_t r = writev(g_proto_fd, io + k, 6 - k);
        if (r <= 0) return -1;
        while (k < 6 && (size_t)r >= io[k].iov_len) r -= io[k++].iov_len;
        if (k < 6) { io[k].iov_base = (char*)io[k].iov_base + r; io[k].iov_len -= r; }
    }
    return 0;
}

// act holds rows start .. start+count-1 (whole battles: both multiples of v->k)
static void vec_step(Vec *v, Batch *b, const float *act, int repeat, int start, int count) {
    int (*over_set)(void*) = FN(int(*)(void*), OFF_LBMS_isGameOverSet);
    double t0 = now_s(), reset_t = 0;
    int K = v->k, battles = v->n / K;
    for (int bi = start / K; bi < (start + count) / K; bi++) {
        Env *e = &v->env[bi];
        int r0 = bi * K, pidx[MAX_AGENTS];
        Action a[MAX_AGENTS], hold[MAX_AGENTS];
        const Action *now[MAX_AGENTS];
        Ctl *ctl[MAX_AGENTS];
        for (int j = 0; j < K; j++) {
            const float *f = act + (r0 + j - start) * ACT_DIM;
            a[j] = (Action){f[0], f[1], clamp_btn(f[2]), f[3], f[4], clamp_btn(f[5]), f[6], f[7], f[8] > 0.5f};
            mirror_action(&a[j], e->mirror);
            hold[j] = (Action){a[j].mx, a[j].my};
            pidx[j] = e->ag[j].pidx;
            ctl[j] = &e->ag[j].ctl;
        }
        int over = 0;
        for (int t = 0; t < repeat && !over; t++) {
            for (int j = 0; j < K; j++) now[j] = t == 0 ? &a[j] : t % STEER_TICKS == 0 ? &hold[j] : NULL;
            sim_tick_agents(e->server, K, pidx, now, ctl);
            over = over_set(e->server);
        }
        int trunc = !over && v->maxticks > 0 && FN(int(*)(void*), OFF_LBMS_getTick)(e->server) >= v->maxticks;
        for (int j = 0; j < K; j++) {
            int i = r0 + j;
            vec_info(e, b->info + i * INFO_DIM, over ? FLAG_TERMINATED : trunc ? FLAG_TRUNCATED : 0);
            float *rc = b->rc + i * RC_DIM;
            rew_step(e->server, e->team, e->ag[j].pidx, &e->ag[j].rew, over, rc);
            b->rew[i] = rc[RC_WIN];
        }
        if (over || trunc) {
            double r0t = now_s();
            e->seed += battles;
            if (vec_build(v, e)) die_vec("rebuilding a battle failed");
            reset_t += now_s() - r0t;
        }
        vec_obs_all(e, b->obs + r0 * OBS_DIM, b->grid + r0 * GRID * GRID);
    }
    b->hdr[2] = (int)((now_s() - t0) * 1e6);
    b->hdr[3] = (int)(reset_t * 1e6);
}

static void vec_reset_all(Vec *v, Batch *b, int base_seed) {
    double t0 = now_s();
    int K = v->k, battles = v->n / K;
    for (int bi = 0; bi < battles; bi++) {
        Env *e = &v->env[bi];
        if (base_seed != -1) e->seed = base_seed + bi;
        else if (e->server) e->seed += battles;
        if (vec_build(v, e)) die_vec("building a battle failed (unknown loc/hero, or fewer teammates than agents?)");
        for (int i = bi * K; i < (bi + 1) * K; i++) {
            b->rew[i] = 0;
            memset(b->rc + i * RC_DIM, 0, RC_DIM * sizeof *b->rc);
            vec_info(e, b->info + i * INFO_DIM, 0);
        }
        vec_obs_all(e, b->obs + bi * K * OBS_DIM, b->grid + bi * K * GRID * GRID);
    }
    b->hdr[2] = b->hdr[3] = (int)((now_s() - t0) * 1e6);
}

// serve mode (protocol: comment above serve_reply)
static int cmd_serve(const char *root, int argc, char **argv) {
    g_quiet = 1;
    int saved = dup(1);                      // keep table-loading chatter off the protocol channel
    dup2(2, 1);
    int fail = cmd_m1(root, 1) || load_tile_flags(root);
    fflush(stdout);
    dup2(saved, 1);
    close(saved);
    if (fail) { printf("{\"error\":\"data tables failed to load\"}\n"); return 1; }
    printf("{\"ready\":1}\n");
    fflush(stdout);

    char line[8192], *server = NULL;
    static Env env[2];                       // the brawlers we steer: [0] in player 0's team, [1] in the other team
    RewState rew = {0};
    float rc[RC_DIM] = {0};
    while (fgets(line, sizeof line, stdin)) {
        char *words[128];
        int n = 0;
        for (char *t = strtok(line, " \t\r\n"); t && n < 128; t = strtok(NULL, " \t\r\n")) words[n++] = t;
        if (!n) continue;
        if (!strcmp(words[0], "quit")) break;
        int rows = env[0].k + env[1].k;
        if (!strcmp(words[0], "reset")) {
            char *args[160];
            int na = 0;
            for (int i = 1; i < n; i++) args[na++] = words[i];
            for (int i = 0; i < argc && na < 160; i++) args[na++] = argv[i];
            if (server) destroy_battle(server);
            free(g_prev_tiles);
            g_prev_tiles = NULL;
            memset(env, 0, sizeof env);
            g_nsent = 0;
            if (!(server = build_battle(na, args, 0))) continue;   // build_battle printed the error line
            // skip=<ticks>: fast-forward the start countdown (3v3 modes ignore movement for the first 120 ticks)
            Ctl ctl = {0};
            for (int i = atoi(kv(na, args, "skip", "0")); i > 0; i--) sim_tick(server, NULL, &ctl);
            int team0 = player_team(server, 0), pidx[MAX_AGENTS], len = 0;
            char rowlist[64] = "";
            for (int t = 0; t < 2; t++) {
                int team = t ? 1 - team0 : team0, want = atoi(kv(na, args, t ? "enemies" : "agents", t ? "0" : "1"));
                env[t].server = server;
                env_init(&env[t], team, team_players(server, team, want, pidx), pidx);
                for (int j = 0; j < env[t].k; j++)
                    len += snprintf(rowlist + len, sizeof rowlist - len, "%s%d", len ? "," : "", env[t].ag[j].pidx);
            }
            char *tm = *(char**)(server + 0x48), *hero = player_char(server, 0), meta[4096];
            int wr = hero ? skill_range(FN(void*(*)(void*), OFF_LCS_getWeaponSkill)(hero)) : 0;
            int sr = hero ? skill_range(FN(void*(*)(void*), OFF_LCS_getUltiSkill)(hero)) : 0;
            snprintf(meta, sizeof meta, "\"meta\":{\"loc\":\"%s\",\"hero\":\"%s\",\"seed\":%s,\"tileSize\":300,\"w\":%d,"
                     "\"h\":%d,\"ownPlayer\":0,\"tickRate\":20,\"players\":%d,\"range\":[%d,%d],\"rssKB\":%ld,"
                     "\"rc_names\":\"" RC_NAMES "\",\"rows\":[%s],\"obs_layout\":\"%s\"},",
                     kv(na, args, "loc", "Gemgrab1"), kv(na, args, "hero", "ShotgunGirl"), kv(na, args, "seed", "12345"),
                     *(int*)(tm + 0x68), *(int*)(tm + 0x6c), ((ArrayList*)(server + 0x4c))->size, wr, sr, rss_kb(),
                     rowlist, obs_layout());
            rew_read(server, team0, 0, &rew);
            memset(rc, 0, sizeof rc);
            serve_reply(server, meta, rc);
        } else if (!server) {
            printf("{\"error\":\"%s before reset\"}\n", words[0]);
            fflush(stdout);
        } else if (!strcmp(words[0], "step")) {
            Action a[2 * MAX_AGENTS] = {{0}}, hold[2 * MAX_AGENTS] = {{0}};
            const Action *now[2 * MAX_AGENTS];
            Ctl *ctl[2 * MAX_AGENTS];
            int pidx[2 * MAX_AGENTS], skip[2 * MAX_AGENTS] = {0}, r = 0, bad = n < 3;
            for (int w = 2; w < n && !bad; r++) {   // per row: 9 numbers, or "-" = no input this step
                if (r >= rows || (strcmp(words[w], "-") && w + 9 > n)) { bad = 1; break; }
                if (!strcmp(words[w], "-")) { skip[r] = 1; w++; continue; }
                char **f = words + w;
                a[r] = (Action){strtof(f[0], NULL), strtof(f[1], NULL), atoi(f[2]), strtof(f[3], NULL), strtof(f[4], NULL),
                                atoi(f[5]), strtof(f[6], NULL), strtof(f[7], NULL), atoi(f[8])};
                w += 9;
            }
            if (bad) {
                printf("{\"error\":\"step needs <repeat> and per row 9 numbers or '-' (%d rows)\"}\n", rows);
                fflush(stdout);
                continue;
            }
            for (int t = 0, i = 0; t < 2; t++)
                for (int j = 0; j < env[t].k; j++, i++) {
                    mirror_action(&a[i], env[t].mirror);
                    hold[i] = (Action){a[i].mx, a[i].my};   // long steps re-steer every STEER_TICKS, without presses
                    pidx[i] = env[t].ag[j].pidx;
                    ctl[i] = &env[t].ag[j].ctl;
                }
            int repeat = atoi(words[1]);
            for (int i = 0; i < (repeat < 1 ? 1 : repeat); i++) {
                for (int j = 0; j < rows; j++) now[j] = skip[j] ? NULL : i == 0 ? &a[j] : i % STEER_TICKS == 0 ? &hold[j] : NULL;
                sim_tick_agents(server, rows, pidx, now, ctl);
                if (FN(int(*)(void*), OFF_LBMS_isGameOverSet)(server)) break;
            }
            rew_step(server, env[0].team, 0, &rew, FN(int(*)(void*), OFF_LBMS_isGameOverSet)(server), rc);
            serve_reply(server, NULL, rc);
        } else if (!strcmp(words[0], "obs")) {
            static float o[2 * MAX_AGENTS][OBS_DIM];
            static unsigned char g[2 * MAX_AGENTS][GRID * GRID];
            int r = 0;
            for (int t = 0; t < 2; t++) {
                for (int j = 0; j < env[t].k; j++, r++) vec_obs(&env[t], j, o[r], g[r]);
                vec_remember(&env[t]);
            }
            // built in memory and sent with one write(): line-buffered stdout sent the ~6 KB line in 1 KB pieces,
            // and adb then held it back ~40 ms every other call (the agents' decision delayed play.py's frames)
            static char out[2 * MAX_AGENTS * (OBS_DIM * 16 + GRID * GRID * 2 + 8) + 64];
            int n = sprintf(out, "{\"obs\":[");
            for (int i = 0; i < r; i++) {
                n += sprintf(out + n, i ? ",[" : "[");
                for (int f = 0; f < OBS_DIM; f++) n += sprintf(out + n, f ? ",%.6g" : "%.6g", o[i][f]);
                n += sprintf(out + n, "]");
            }
            n += sprintf(out + n, "],\"grid\":[");
            for (int i = 0; i < r; i++) {
                n += sprintf(out + n, i ? ",\"" : "\"");
                for (int c = 0; c < GRID * GRID; c++) n += sprintf(out + n, "%02x", g[i][c]);
                n += sprintf(out + n, "\"");
            }
            n += sprintf(out + n, "]}\n");
            fflush(stdout);
            for (int w = 0; w < n; ) {
                int k = write(1, out + w, n - w);
                if (k <= 0) break;
                w += k;
            }
        } else {
            printf("{\"error\":\"unknown command %s\"}\n", words[0]);
            fflush(stdout);
        }
    }
    if (server) destroy_battle(server);
    return 0;
}

static int cmd_vec(const char *root, int argc, char **argv) {
    g_quiet = 1;
    g_log_inputs = 0;
    if (cmd_m1(root, 1)) die_vec("data tables failed to load");
    if (load_tile_flags(root)) die_vec("cannot read csv_logic/tiles.csv");
    const char *layout = obs_layout();
    if (!layout) die_vec("OBS_DIM does not match OBS_BLOCKS");
    if (!atoi(kv(argc, argv, "vision", "0")) && skip_vision_encoding()) die_vec("cannot skip the vision encoding");
    // cpus=<hex mask>: best-effort pinning. The vendor power management isolates cores now and then (e.g. cpu 4 at
    // idle), and pinning to an isolated core fails, so a failure only warns.
    unsigned long mask = strtoul(kv(argc, argv, "cpus", "0"), NULL, 16);
    if (mask) {
        cpu_set_t set;
        CPU_ZERO(&set);
        for (int c = 0; c < 32; c++) if (mask >> c & 1) CPU_SET(c, &set);
        if (sched_setaffinity(0, sizeof set, &set)) fprintf(stderr, "[vec] warning: cannot pin to cpus %lx\n", mask);
    }
    Vec v = {0};
    v.n = atoi(kv(argc, argv, "n", "16"));
    v.k = atoi(kv(argc, argv, "agents", "1"));
    v.repeat = atoi(kv(argc, argv, "repeat", "4"));
    v.skip = atoi(kv(argc, argv, "skip", "120"));
    v.maxticks = atoi(kv(argc, argv, "maxticks", "12000"));
    v.argc = argc; v.argv = argv;
    if (v.n < 1 || v.n > 4096 || v.repeat < 1) die_vec("bad n or repeat");
    if (v.k < 1 || v.k > MAX_AGENTS || v.n % v.k) die_vec("agents must be 1..3 and divide n");
    v.env = calloc(v.n / v.k, sizeof *v.env);
    Batch b = batch_alloc(v.n);
    float *act = malloc(v.n * ACT_DIM * sizeof *act);
    vec_reset_all(&v, &b, atoi(kv(argc, argv, "seed", "1")));
    int bench = atoi(kv(argc, argv, "bench", "0"));
    if (bench > 0) {   // bench=<batches> [policy=random|idle|script]: time vec_step on the device, no protocol
        const char *policy = kv(argc, argv, "policy", "random");
        unsigned rnd = 12345;
        double busy = 0, resets = 0;
        int episodes = 0;
        for (int s = 0; s < bench; s++) {
            for (int i = 0; i < v.n; i++) {
                float *f = act + i * ACT_DIM;
                memset(f, 0, ACT_DIM * sizeof *f);
                if (!strcmp(policy, "random")) {
                    rnd = rnd * 1103515245u + 12345u; f[0] = (rnd >> 8 & 0xffff) / 32768.f - 1;
                    rnd = rnd * 1103515245u + 12345u; f[1] = (rnd >> 8 & 0xffff) / 32768.f - 1;
                    rnd = rnd * 1103515245u + 12345u; f[2] = (rnd >> 8 & 0xffff) < 0x4ccc;
                    f[5] = 1;
                } else if (!strcmp(policy, "script")) {
                    Action a;
                    Env *e = &v.env[i / v.k];
                    script_policy(e->server, e->ag[i % v.k].pidx, &a);
                    f[0] = a.mx; f[1] = a.my; f[2] = a.atk; f[5] = a.sup;
                }
            }
            vec_step(&v, &b, act, v.repeat, 0, v.n);
            busy += b.hdr[2]; resets += b.hdr[3];
            for (int i = 0; i < v.n; i += v.k) episodes += b.info[i * INFO_DIM] != 0;
        }
        double ticks = (double)bench * v.n * v.repeat;
        fprintf(stderr, "[vec] bench n=%d repeat=%d policy=%s: %.0f ticks/s, %.4f ms/tick (resets %.1f%%), %d episodes, rss=%ldKB\n",
                v.n, v.repeat, policy, ticks / (busy * 1e-6), busy * 1e-3 / ticks, 100 * resets / busy, episodes, rss_kb());
        return 0;
    }
    int hello[11] = {VEC_MAGIC, VEC_VERSION, v.n, OBS_DIM, ACT_DIM, INFO_DIM, v.repeat, GRID, (int)strlen(layout),
                     RC_DIM, (int)strlen(RC_NAMES)};
    if (write_full(g_proto_fd, hello, sizeof hello) || write_full(g_proto_fd, layout, hello[8]) ||
        write_full(g_proto_fd, RC_NAMES, hello[10]) || write_batch(&b, 0, v.n)) return 1;
    fprintf(stderr, "[vec] ready: n=%d repeat=%d obs=%d rss=%ldKB\n", v.n, v.repeat, OBS_DIM, rss_kb());
    for (;;) {
        int req[4];
        if (read_full(0, req, sizeof req)) break;            // host closed the pipe
        if (req[0] != VEC_MAGIC) die_vec("bad request magic (protocol out of sync)");
        if (req[1] == VEC_QUIT) break;
        int start = 0, count = v.n;
        if (req[1] == VEC_STEP) {
            count = req[3];
            if (read_full(0, &start, sizeof start)) break;
            if (count < 1 || start < 0 || start + count > v.n || start % v.k || count % v.k)
                die_vec("step: row range out of bounds or not whole battles");
            if (read_full(0, act, count * ACT_DIM * sizeof *act)) break;
            vec_step(&v, &b, act, req[2] > 0 ? req[2] : v.repeat, start, count);
        } else if (req[1] == VEC_RESET) {
            vec_reset_all(&v, &b, req[2]);
        } else die_vec("unknown request");
        if (write_batch(&b, start, count)) break;
    }
    for (int i = 0; i < v.n / v.k; i++) if (v.env[i].server) destroy_battle(v.env[i].server);
    fprintf(stderr, "[vec] exit rss=%ldKB\n", rss_kb());
    return 0;
}

// ---- isotest: battles sharing one process must not influence each other ----
// isotest [n=6] [ticks=3000] [seed=1] [think=4] [decoy=Ball1] [novision=0]: runs n seeds solo (one after another, destroying each),
// then all n in lockstep, then staggered (battle i starts i*41 ticks late, reverse stepping order, a decoy battle on
// another map rebuilt every 97 ticks). Every battle runs the scripted policy; a hash of all objects per tick must
// match the solo run exactly. novision=1: the solo runs encode vision, the others skip it (must not change anything).
static unsigned state_hash(char *s) {
    unsigned h = 2166136261u;
#define MIX(v) (h = (h ^ (unsigned)(v)) * 16777619u)
    MIX(FN(int(*)(void*), OFF_LBMS_getTick)(s));
    MIX(team_score_of(s, 0)); MIX(team_score_of(s, 1));
    ArrayList *objs = *(ArrayList**)s;
    for (int i = 0; i < objs->size; i++) {
        char *o = ((char**)objs->data)[i];
        int *w = (int*)o, type = (*(int(***)(void*))o)[0x20 / 4](o);
        MIX(w[1]); MIX(type); MIX(obj_x(o)); MIX(obj_y(o)); MIX(w[0x24/4]);
        if (type == 0) MIX(w[0x90/4]);
    }
#undef MIX
    return h;
}

static void iso_tick(Env *e, int t, int think) {
    Action a;
    int decide = t % think == 0;
    if (decide) script_policy(e->server, 0, &a);
    sim_tick(e->server, decide ? &a : NULL, &e->ag[0].ctl);
}

static int cmd_isotest(const char *root, int argc, char **argv) {
    int (*over_set)(void*) = FN(int(*)(void*), OFF_LBMS_isGameOverSet);
    g_quiet = 1;
    g_log_inputs = 0;
    if (cmd_m1(root, 1)) return 1;
    int n = atoi(kv(argc, argv, "n", "6")), T = atoi(kv(argc, argv, "ticks", "3000"));
    int seed0 = atoi(kv(argc, argv, "seed", "1")), think = atoi(kv(argc, argv, "think", "4"));
    char decoy_loc[64];
    snprintf(decoy_loc, sizeof decoy_loc, "loc=%s", kv(argc, argv, "decoy", "Ball1"));
    Vec v = {.n = n, .argc = argc, .argv = argv};
    unsigned *ref = calloc((size_t)n * T, sizeof *ref);
    int *len = calloc(n, sizeof *len), fails = 0;
    Env *env = calloc(n, sizeof *env);

    double t0 = now_s();
    for (int i = 0; i < n; i++) {                   // solo reference runs
        Env e = {.seed = seed0 + i};
        if (vec_build(&v, &e)) return 1;
        for (int t = 0; t < T; t++) {
            iso_tick(&e, t, think);
            ref[(size_t)i * T + t] = state_hash(e.server);
            len[i] = t + 1;
            if (over_set(e.server)) break;
        }
        printf("[iso] solo seed %d: %d ticks, score %d-%d\n", e.seed, len[i], team_score_of(e.server, 0), team_score_of(e.server, 1));
        destroy_battle(e.server);
    }
    printf("[iso] solo runs: %.2fs, rss=%ldKB\n", now_s() - t0, rss_kb());
    if (atoi(kv(argc, argv, "novision", "0"))) {   // compare the runs below, without vision encoding, to the solo runs
        if (skip_vision_encoding()) return 1;
        printf("[iso] vision encoding skipped from here on\n");
    }

    for (int mode = 0; mode < 2; mode++) {          // 0 lockstep, 1 staggered + reverse order + decoy
        int first_bad = -1, bad_env = -1, done = 0;
        char *decoy = NULL;
        int decoy_seed = 1000;
        memset(env, 0, n * sizeof *env);
        for (int i = 0; i < n; i++) env[i].seed = seed0 + i;
        if (mode == 0) for (int i = 0; i < n; i++) if (vec_build(&v, &env[i])) return 1;
        int *t_env = calloc(n, sizeof *t_env);
        for (int g = 0; done < n; g++) {
            if (mode == 1 && g % 97 == 0) {
                if (decoy) destroy_battle(decoy);
                char seed[24], *args[2] = {decoy_loc, seed};
                snprintf(seed, sizeof seed, "seed=%d", decoy_seed++);
                decoy = build_battle(2, args, 0);
            }
            if (decoy) { Ctl dc = {0}; sim_tick(decoy, NULL, &dc); }
            for (int j = 0; j < n; j++) {
                int i = mode == 1 ? n - 1 - j : j;
                Env *e = &env[i];
                if (mode == 1 && !e->server && g >= i * 41 && t_env[i] == 0) { if (vec_build(&v, e)) return 1; }
                if (!e->server || t_env[i] >= len[i]) continue;
                iso_tick(e, t_env[i], think);
                unsigned h = state_hash(e->server);
                if (h != ref[(size_t)i * T + t_env[i]] && first_bad < 0) { first_bad = t_env[i]; bad_env = i; }
                if (++t_env[i] == len[i]) { destroy_battle(e->server); e->server = NULL; done++; }
            }
        }
        if (decoy) destroy_battle(decoy);
        free(t_env);
        printf("[iso] %s: %s", mode ? "staggered+decoy" : "lockstep", first_bad < 0 ? "all ticks identical to solo\n" : "MISMATCH");
        if (first_bad >= 0) { printf(" battle %d (seed %d) first differs at tick %d\n", bad_env, seed0 + bad_env, first_bad); fails++; }
    }
    printf("[iso] rss=%ldKB\n[iso] RESULT: %s\n", rss_kb(), fails ? "FAIL - battles interfere" : "PASS - battles are isolated");
    return fails ? 1 : 0;
}

// movetest [dir=up]: one move input after the countdown, then report how long the hero keeps walking
static int cmd_movetest(const char *root, int argc, char **argv) {
    g_quiet = 1;
    if (cmd_m1(root, 1)) return 1;
    Vec v = {.n = 1, .skip = 120, .argc = argc, .argv = argv};
    Env e = {.seed = atoi(kv(argc, argv, "seed", "1"))};
    if (vec_build(&v, &e)) return 1;
    Action a = {0, -1};
    for (int t = 0; t < 60; t++) {
        sim_tick(e.server, t == 0 ? &a : NULL, &e.ag[0].ctl);
        char *c = player_char(e.server, 0);
        if (c) printf("[move] t=%2d x=%d y=%d +0x14=%d tick=%d\n", t, obj_x(c), obj_y(c), *(int*)(c + 0x14),
                      FN(int(*)(void*), OFF_LBMS_getTick)(e.server));
    }
    return 0;
}

// deathtest [seed=1]: scripted battle; prints LogicPlayer ints 0x00..0x6c whenever a player's hero dies or respawns
static int cmd_deathtest(const char *root, int argc, char **argv) {
    g_quiet = 1;
    g_log_inputs = 0;
    if (cmd_m1(root, 1)) return 1;
    Vec v = {.n = 1, .skip = 120, .argc = argc, .argv = argv};
    Env e = {.seed = atoi(kv(argc, argv, "seed", "1"))};
    if (vec_build(&v, &e)) return 1;
    printf("[death] respawn_ticks=%d battle_ticks=%d variation=%d\n", e.respawn_ticks, e.battle_ticks, *(int*)(e.server + 0xa4));
    int was[MAX_PLAYERS], np = ((ArrayList*)(e.server + 0x4c))->size, events = 0;
    for (int p = 0; p < np; p++) was[p] = 1;
    for (int t = 0; t < 3000 && events < 12; t++) {
        iso_tick(&e, t, 4);
        int tick = FN(int(*)(void*), OFF_LBMS_getTick)(e.server);
        for (int p = 0; p < np; p++) {
            int alive = player_char(e.server, p) != NULL;
            if (alive == was[p]) continue;
            was[p] = alive;
            events++;
            int *w = (int*)player_at(e.server, p);
            printf("[death] tick %d player %d %s:", tick, p, alive ? "RESPAWNED" : "died");
            for (int k = 0; k < 0x70 / 4; k++) printf(" %x=%d", k * 4, w[k]);
            printf("\n");
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 4 && !strcmp(argv[3], "vec")) {   // binary protocol on the real stdout; all printing to stderr
        g_proto_fd = dup(1);
        dup2(2, 1);
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    if (argc < 4) { printf("usage: %s <libdir> <dataroot> list|m1|m2|m3|serve|vec|isotest [key=value...]|str OFF...\n", argv[0]); return 1; }
    struct sigaction sa = {0};
    sa.sa_sigaction = on_fault;
    sa.sa_flags = SA_SIGINFO;
    int sigs[] = {SIGSEGV, SIGBUS, SIGILL, SIGABRT, SIGFPE};
    for (size_t i = 0; i < sizeof sigs / sizeof *sigs; i++) sigaction(sigs[i], &sa, NULL);

    if (load_libs(argv[1])) return 2;
    if (!strcmp(argv[3], "list")) return cmd_m1(argv[2], 0);
    if (!strcmp(argv[3], "m1"))   return cmd_m1(argv[2], 1);
    if (!strcmp(argv[3], "m2"))   return cmd_m2(argv[2], argc - 4, argv + 4) ? 0 : 1;
    if (!strcmp(argv[3], "m3"))   return cmd_m3(argv[2], argc - 4, argv + 4);
    if (!strcmp(argv[3], "serve")) return cmd_serve(argv[2], argc - 4, argv + 4);
    if (!strcmp(argv[3], "vec"))   return cmd_vec(argv[2], argc - 4, argv + 4);
    if (!strcmp(argv[3], "isotest")) return cmd_isotest(argv[2], argc - 4, argv + 4);
    if (!strcmp(argv[3], "movetest")) return cmd_movetest(argv[2], argc - 4, argv + 4);
    if (!strcmp(argv[3], "deathtest")) return cmd_deathtest(argv[2], argc - 4, argv + 4);
    if (!strcmp(argv[3], "str")) {  // libg string literals are encrypted on disk; .init_array decodes them
        for (int i = 4; i < argc; i++) {
            unsigned long off = strtoul(argv[i], NULL, 0);
            printf("[str] 0x%lx = \"%.120s\"\n", off, (const char*)(g_base + off));
        }
        return 0;
    }
    printf("unknown command %s\n", argv[3]);
    return 1;
}
