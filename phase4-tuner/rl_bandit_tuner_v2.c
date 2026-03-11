// Deterministic multi-node-safe bandit NCCL tuner plugin (v2).
//
// Fixes critical bugs in v1 (rl_bandit_tuner_plugin.c):
//   1. RNG divergence across ranks → replaced with deterministic round-robin
//   2. Reward file isolation across nodes → shared volume path + rank-aware writes
//   3. Missing AUTO arm → added as 4th arm (don't modify cost table)
//   4. Size band mismatch → aligned with workload_aware_tuner_v3.c (7 bands)
//   5. No outlier handling → IQR trimming when picking exploitation arm
//
// Two phases:
//   EXPLORE: Deterministic round-robin across 4 arms, M rounds each.
//            All ranks compute the same arm from iteration count (no RNG).
//   EXPLOIT: Pick arm with lowest trimmed mean latency, stick with it.
//
// Reward logging:
//   Application writes: allreduce,{nBytes},{nNodes},{nRanks},{latency_ms}
//   Only rank 0 writes (set NCCL_TUNER_RANK=0 on rank 0).
//   All ranks read the same file at the explore→exploit transition.
//
// Environment variables:
//   NCCL_TUNER_REWARD_FILE  — path to reward log (default: /results/bandit_rewards.log)
//   NCCL_TUNER_RANK         — set to "0" on rank 0 for reward writing
//   NCCL_TUNER_EXPLORE_ROUNDS — rounds per arm during exploration (default: 5)
//   NCCL_TUNER_LOG          — set to "1" for verbose logging

#include "tuner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define __hidden __attribute__ ((visibility("hidden")))

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define MAX_LINE_LENGTH 256
#define MAX_KEYS        64
#define NUM_ARMS         4
#define MAX_REWARDS_PER_ARM 256

// Arms: the candidate (algo, proto) pairs.
// Arm 3 (AUTO) uses algo=-1, proto=-1 to signal "don't modify cost table".
static const int ARM_ALGO[NUM_ARMS]  = { NCCL_ALGO_TREE, NCCL_ALGO_TREE, NCCL_ALGO_RING, -1 };
static const int ARM_PROTO[NUM_ARMS] = { NCCL_PROTO_SIMPLE, NCCL_PROTO_LL128, NCCL_PROTO_SIMPLE, -1 };
static const char* ARM_NAMES[NUM_ARMS] = { "Tree+Simple", "Tree+LL128", "Ring+Simple", "AUTO" };

typedef struct {
    ncclFunc_t collType;
    int sizeBand;
    int nNodes;
    int nRanks;
} BanditKey;

typedef struct {
    double latencies[MAX_REWARDS_PER_ARM];
    int count;
} ArmStats;

typedef struct {
    BanditKey key;
    ArmStats arms[NUM_ARMS];
    int callCount;       // total getCollInfo calls for this key
    int exploitArmIdx;   // -1 during explore, >=0 during exploit
} BanditEntry;

typedef struct {
    BanditEntry entries[MAX_KEYS];
    int numKeys;

    char rewardFile[PATH_MAX];
    int exploreRounds;   // M rounds per arm
    int verbose;

    size_t nRanks;
    size_t nNodes;
    ncclDebugLogger_t logFunction;
} TunerContext;

// ---- Size banding (7 bands, aligned with workload_aware_tuner_v3.c) ----

static int sizeBand(size_t nBytes) {
    if (nBytes < 1024ULL)              return 0;  // < 1 KB
    if (nBytes < 256ULL * 1024)        return 1;  // 1 KB - 256 KB
    if (nBytes < 1024ULL * 1024)       return 2;  // 256 KB - 1 MB
    if (nBytes < 4ULL * 1024 * 1024)   return 3;  // 1 MB - 4 MB
    if (nBytes < 32ULL * 1024 * 1024)  return 4;  // 4 MB - 32 MB
    if (nBytes < 128ULL * 1024 * 1024) return 5;  // 32 MB - 128 MB
    return 6;                                      // >= 128 MB
}

// ---- Key management ----

static int keysEqual(const BanditKey* a, const BanditKey* b) {
    return a->collType == b->collType &&
           a->sizeBand == b->sizeBand &&
           a->nNodes   == b->nNodes &&
           a->nRanks   == b->nRanks;
}

static int getOrAddKey(TunerContext* ctx, const BanditKey* key) {
    for (int i = 0; i < ctx->numKeys; ++i) {
        if (keysEqual(&ctx->entries[i].key, key))
            return i;
    }
    if (ctx->numKeys >= MAX_KEYS) return -1;

    int idx = ctx->numKeys++;
    ctx->entries[idx].key = *key;
    ctx->entries[idx].callCount = 0;
    ctx->entries[idx].exploitArmIdx = -1;
    for (int a = 0; a < NUM_ARMS; ++a) {
        ctx->entries[idx].arms[a].count = 0;
    }
    return idx;
}

// ---- IQR trimmed mean ----

static int cmpDouble(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

static double trimmedMean(const double* vals, int n) {
    if (n <= 0) return 1e30;
    if (n <= 4) {
        double sum = 0;
        for (int i = 0; i < n; ++i) sum += vals[i];
        return sum / n;
    }

    // Sort a copy
    double sorted[MAX_REWARDS_PER_ARM];
    int m = (n < MAX_REWARDS_PER_ARM) ? n : MAX_REWARDS_PER_ARM;
    for (int i = 0; i < m; ++i) sorted[i] = vals[i];
    qsort(sorted, m, sizeof(double), cmpDouble);

    // IQR
    double q1 = sorted[m / 4];
    double q3 = sorted[(3 * m) / 4];
    double iqr = q3 - q1;
    double lo = q1 - 1.5 * iqr;
    double hi = q3 + 1.5 * iqr;

    double sum = 0;
    int count = 0;
    for (int i = 0; i < m; ++i) {
        if (sorted[i] >= lo && sorted[i] <= hi) {
            sum += sorted[i];
            count++;
        }
    }
    return (count > 0) ? (sum / count) : (sorted[m / 2]);
}

// ---- Reward file parsing ----

static ncclFunc_t parseCollType(const char* str) {
    if (strcmp(str, "broadcast") == 0) return ncclFuncBroadcast;
    if (strcmp(str, "reduce") == 0) return ncclFuncReduce;
    if (strcmp(str, "allgather") == 0) return ncclFuncAllGather;
    if (strcmp(str, "reducescatter") == 0) return ncclFuncReduceScatter;
    if (strcmp(str, "allreduce") == 0) return ncclFuncAllReduce;
    return ncclFuncAllReduce;
}

// Ingest all rewards from file and attribute to the correct arm based on
// the deterministic round-robin schedule.
static void ingestAllRewards(TunerContext* ctx) {
    FILE* f = fopen(ctx->rewardFile, "r");
    if (!f) return;

    // We need to attribute each reward to the arm that was active when it
    // was generated. Since exploration is deterministic round-robin:
    //   call i → arm = (i % NUM_ARMS)
    // We track per-key call counts to map rewards to arms.

    // Reset all arm stats for fresh ingestion
    for (int k = 0; k < ctx->numKeys; ++k) {
        for (int a = 0; a < NUM_ARMS; ++a) {
            ctx->entries[k].arms[a].count = 0;
        }
    }

    // Per-key call counter for attribution
    int keyCallCounts[MAX_KEYS];
    memset(keyCallCounts, 0, sizeof(keyCallCounts));

    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;

        // Parse: collType,nBytes,nNodes,nRanks,latency_ms
        char buf[MAX_LINE_LENGTH];
        strncpy(buf, line, sizeof(buf));
        buf[sizeof(buf)-1] = '\0';

        char* saveptr = NULL;
        char* tok;

        tok = strtok_r(buf, ",", &saveptr);
        if (!tok) continue;
        ncclFunc_t collType = parseCollType(tok);

        tok = strtok_r(NULL, ",", &saveptr);
        if (!tok) continue;
        size_t nBytes = (size_t)strtoull(tok, NULL, 10);

        tok = strtok_r(NULL, ",", &saveptr);
        if (!tok) continue;
        int nNodes = atoi(tok);

        tok = strtok_r(NULL, ",", &saveptr);
        if (!tok) continue;
        int nRanks = atoi(tok);

        tok = strtok_r(NULL, ",", &saveptr);
        if (!tok) continue;
        double latencyMs = strtod(tok, NULL);

        BanditKey key;
        key.collType = collType;
        key.sizeBand = sizeBand(nBytes);
        key.nNodes = nNodes;
        key.nRanks = nRanks;

        int kIdx = getOrAddKey(ctx, &key);
        if (kIdx < 0) continue;

        // Determine which arm was active for this reward
        int callNum = keyCallCounts[kIdx]++;
        int armIdx = callNum % NUM_ARMS;

        ArmStats* arm = &ctx->entries[kIdx].arms[armIdx];
        if (arm->count < MAX_REWARDS_PER_ARM) {
            arm->latencies[arm->count] = latencyMs;
            arm->count++;
        }
    }

    fclose(f);
}

// Pick the best arm for a key based on trimmed mean of collected rewards.
static int pickBestArm(TunerContext* ctx, BanditEntry* entry) {
    double bestMean = 1e30;
    int bestIdx = NUM_ARMS - 1; // Default to AUTO

    for (int a = 0; a < NUM_ARMS; ++a) {
        ArmStats* arm = &entry->arms[a];
        if (arm->count < 2) continue; // Need at least 2 samples

        double mean = trimmedMean(arm->latencies, arm->count);

        if (ctx->verbose && ctx->logFunction) {
            ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                "BANDIT-v2: key(coll=%d,band=%d) arm %d (%s): trimmed_mean=%.2f ms (n=%d)",
                entry->key.collType, entry->key.sizeBand,
                a, ARM_NAMES[a], mean, arm->count);
        }

        if (mean < bestMean) {
            bestMean = mean;
            bestIdx = a;
        }
    }

    return bestIdx;
}

// ---- NCCL tuner API ----

__hidden ncclResult_t pluginInit(void** context, uint64_t commId, size_t nRanks, size_t nNodes,
                                 ncclDebugLogger_t logFunction,
                                 ncclNvlDomainInfo_v5_t* nvlDomainInfo,
                                 ncclTunerConstants_v5_t* constants) {
    (void)nvlDomainInfo;
    (void)constants;
    (void)commId;

    TunerContext* ctx = (TunerContext*)calloc(1, sizeof(TunerContext));
    if (!ctx) return ncclSystemError;

    ctx->nRanks = nRanks;
    ctx->nNodes = nNodes;
    ctx->logFunction = logFunction;
    ctx->numKeys = 0;
    ctx->exploreRounds = 5; // default: 5 rounds per arm = 20 explore iterations per key

    // Parse env vars
    const char* rewardEnv = getenv("NCCL_TUNER_REWARD_FILE");
    if (rewardEnv && rewardEnv[0] != '\0') {
        strncpy(ctx->rewardFile, rewardEnv, sizeof(ctx->rewardFile));
        ctx->rewardFile[sizeof(ctx->rewardFile)-1] = '\0';
    } else {
        snprintf(ctx->rewardFile, sizeof(ctx->rewardFile),
                 "/results/bandit_rewards_%llu.log",
                 (unsigned long long)commId);
    }

    const char* roundsEnv = getenv("NCCL_TUNER_EXPLORE_ROUNDS");
    if (roundsEnv) {
        int val = atoi(roundsEnv);
        if (val > 0 && val <= 50) ctx->exploreRounds = val;
    }

    const char* logEnv = getenv("NCCL_TUNER_LOG");
    ctx->verbose = (logEnv && logEnv[0] == '1') ? 1 : 0;

    if (logFunction) {
        logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
            "BANDIT-v2: init for %zu nodes, %zu ranks, exploreRounds=%d, rewardFile=%s",
            nNodes, nRanks, ctx->exploreRounds, ctx->rewardFile);
    }

    *context = ctx;
    return ncclSuccess;
}

__hidden ncclResult_t pluginGetCollInfo(void* context, ncclFunc_t collType, size_t nBytes,
                                        int numPipeOps, float** collCostTable,
                                        int numAlgo, int numProto,
                                        int regBuff, int* nChannels) {
    (void)numPipeOps;
    (void)regBuff;

    TunerContext* ctx = (TunerContext*)context;
    if (!ctx) return ncclInternalError;

    // Only tune AllReduce (most impactful collective)
    if (collType != ncclFuncAllReduce) return ncclSuccess;

    BanditKey key;
    key.collType = collType;
    key.sizeBand = sizeBand(nBytes);
    key.nNodes = (int)ctx->nNodes;
    key.nRanks = (int)ctx->nRanks;

    int kIdx = getOrAddKey(ctx, &key);
    if (kIdx < 0) return ncclSuccess;

    BanditEntry* entry = &ctx->entries[kIdx];
    int callNum = entry->callCount++;
    int totalExploreIters = ctx->exploreRounds * NUM_ARMS;
    int armIdx;

    if (entry->exploitArmIdx >= 0) {
        // Already in exploitation phase
        armIdx = entry->exploitArmIdx;
    } else if (callNum < totalExploreIters) {
        // EXPLORE: deterministic round-robin
        armIdx = callNum % NUM_ARMS;
    } else {
        // Transition: explore → exploit
        // Ingest all rewards and pick the best arm
        ingestAllRewards(ctx);
        armIdx = pickBestArm(ctx, entry);
        entry->exploitArmIdx = armIdx;

        if (ctx->logFunction) {
            ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                "BANDIT-v2: EXPLOIT TRANSITION key(coll=%d,band=%d,nodes=%d,ranks=%d) → arm %d (%s) after %d explore iters",
                key.collType, key.sizeBand, key.nNodes, key.nRanks,
                armIdx, ARM_NAMES[armIdx], totalExploreIters);
        }
    }

    // Apply the selected arm
    int algo = ARM_ALGO[armIdx];
    int proto = ARM_PROTO[armIdx];

    if (algo < 0 || proto < 0) {
        // AUTO arm: don't modify cost table, let NCCL decide
        if (ctx->verbose && ctx->logFunction) {
            ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                "BANDIT-v2: iter %d → AUTO (no override)", callNum);
        }
        return ncclSuccess;
    }

    // Validate indices
    if (algo >= numAlgo || proto >= numProto) return ncclSuccess;
    if (collCostTable[algo][proto] == NCCL_ALGO_PROTO_IGNORE) return ncclSuccess;

    // Force this (algo, proto)
    collCostTable[algo][proto] = 0.0f;
    *nChannels = 0; // let NCCL decide channel count

    if (ctx->verbose && ctx->logFunction) {
        const char* phase = (entry->exploitArmIdx >= 0) ? "EXPLOIT" : "EXPLORE";
        ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
            "BANDIT-v2: %s iter %d → arm %d (%s) algo=%d proto=%d",
            phase, callNum, armIdx, ARM_NAMES[armIdx], algo, proto);
    }

    return ncclSuccess;
}

__hidden ncclResult_t pluginFinalize(void* context) {
    TunerContext* ctx = (TunerContext*)context;
    if (ctx) {
        if (ctx->logFunction) {
            ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                "BANDIT-v2: finalize (%d keys tracked)", ctx->numKeys);
            for (int k = 0; k < ctx->numKeys; ++k) {
                BanditEntry* e = &ctx->entries[k];
                if (e->exploitArmIdx >= 0) {
                    ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                        "BANDIT-v2: key(coll=%d,band=%d) → exploit arm %d (%s)",
                        e->key.collType, e->key.sizeBand,
                        e->exploitArmIdx, ARM_NAMES[e->exploitArmIdx]);
                }
            }
        }
        free(ctx);
    }
    return ncclSuccess;
}

#define PLUGIN_NAME "DeterministicBandit_v2"

const ncclTuner_v5_t ncclTunerPlugin_v5 = {
    .name = PLUGIN_NAME,
    .init = pluginInit,
    .getCollInfo = pluginGetCollInfo,
    .finalize = pluginFinalize
};
