// RL-style (multi-armed bandit) NCCL tuner plugin.
// This is based on the NCCL ext-tuner example plugin, but replaces the
// static CSV policy with an online bandit that learns from observed
// latencies written by the application to a reward log.
//
// The application is responsible for logging lines of the form:
//   collType,nBytes,nNodes,nRanks,latency_ms
// e.g.
//   allreduce,4194304,1,8,12.34
//
// The plugin:
// - Groups collectives into "keys" = (collType, sizeBand, nNodes, nRanks)
// - For each key, maintains a small set of candidate (algo, proto) arms
// - Supports three selection strategies (set NCCL_TUNER_STRATEGY):
//     "eps_greedy" (default) - epsilon-greedy with configurable epsilon
//     "ucb1"                 - Upper Confidence Bound (UCB1)
//     "thompson"             - Thompson Sampling (Normal model)
// - Updates arm statistics when it sees new reward log entries for a key,
//   attributing each latency to the arm last used for that key
//
// Environment variables:
// - NCCL_TUNER_REWARD_FILE (optional): path to reward log file.
//   If unset, defaults to /tmp/nccl_tuner_rewards_<commId>.log
// - NCCL_TUNER_EPS (optional): epsilon for epsilon-greedy in [0,1].
//   Default: 0.1
// - NCCL_TUNER_STRATEGY (optional): "eps_greedy", "ucb1", or "thompson".
//   Default: "eps_greedy"
// - NCCL_TUNER_UCB_C (optional): exploration constant for UCB1.
//   Default: 1.414 (sqrt(2))

#include "tuner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <limits.h>

#define __hidden __attribute__ ((visibility("hidden")))

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define MAX_LINE_LENGTH 256

// Bandit configuration limits
#define MAX_KEYS   64
#define MAX_ARMS    4

// Selection strategy
typedef enum {
  STRATEGY_EPS_GREEDY = 0,
  STRATEGY_UCB1       = 1,
  STRATEGY_THOMPSON   = 2,
} SelectionStrategy;

typedef struct {
  ncclFunc_t collType;
  int sizeBand;
  int nNodes;
  int nRanks;
} BanditKey;

typedef struct {
  int algo;
  int proto;
  int count;
  double sumLatencyMs;
} BanditArm;

typedef struct {
  BanditKey key;
  BanditArm arms[MAX_ARMS];
  int numArms;
  int lastArmIdx; // index of arm last selected for this key, or -1 if none yet
} BanditEntry;

typedef struct {
  BanditEntry entries[MAX_KEYS];
  int numKeys;

  char rewardFile[PATH_MAX];
  long rewardOffset;
  double epsilon;
  double ucbC;               // exploration constant for UCB1
  int totalPulls;            // total arm pulls across all keys (for UCB1)
  SelectionStrategy strategy;

  size_t nRanks;
  size_t nNodes;
  ncclDebugLogger_t logFunction;
} TunerContext;

// ---- Helpers copied / adapted from example plugin ----

static ncclFunc_t parseCollType(const char* str) {
  if (strcmp(str, "broadcast") == 0) return ncclFuncBroadcast;
  if (strcmp(str, "reduce") == 0) return ncclFuncReduce;
  if (strcmp(str, "allgather") == 0) return ncclFuncAllGather;
  if (strcmp(str, "reducescatter") == 0) return ncclFuncReduceScatter;
  if (strcmp(str, "allreduce") == 0) return ncclFuncAllReduce;
  return ncclFuncAllReduce;
}

static const char* collTypeToString(ncclFunc_t collType) {
  switch (collType) {
    case ncclFuncBroadcast:    return "broadcast";
    case ncclFuncReduce:       return "reduce";
    case ncclFuncAllGather:    return "allgather";
    case ncclFuncReduceScatter:return "reducescatter";
    case ncclFuncAllReduce:    return "allreduce";
    default:                   return "unknown";
  }
}

// Size banding: bucket nBytes into a small number of bands.
// This must be replicated by the application if it wants to precompute bands,
// but for rewards we log raw nBytes and recompute the band here.
static int sizeBandFromBytes(size_t nBytes) {
  if (nBytes < 1024ULL) return 0;                // < 1 KB
  if (nBytes < 16ULL * 1024) return 1;           // 1 KB - 16 KB
  if (nBytes < 256ULL * 1024) return 2;          // 16 KB - 256 KB
  if (nBytes < 1ULL * 1024 * 1024) return 3;     // 256 KB - 1 MB
  if (nBytes < 8ULL * 1024 * 1024) return 4;     // 1 MB - 8 MB
  return 5;                                      // >= 8 MB
}

static int keysEqual(const BanditKey* a, const BanditKey* b) {
  return a->collType == b->collType &&
         a->sizeBand == b->sizeBand &&
         a->nNodes   == b->nNodes &&
         a->nRanks   == b->nRanks;
}

// Find an existing key or create a new one (if capacity allows).
static int getOrAddKey(TunerContext* ctx, const BanditKey* key) {
  for (int i = 0; i < ctx->numKeys; ++i) {
    if (keysEqual(&ctx->entries[i].key, key)) {
      return i;
    }
  }
  if (ctx->numKeys >= MAX_KEYS) {
    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                       "RL-TUNER: Reached MAX_KEYS=%d, cannot add new key for collType=%s, sizeBand=%d, nodes=%d, ranks=%d",
                       MAX_KEYS, collTypeToString(key->collType), key->sizeBand, key->nNodes, key->nRanks);
    }
    return -1;
  }

  int idx = ctx->numKeys++;
  ctx->entries[idx].key = *key;
  ctx->entries[idx].numArms = 0;
  ctx->entries[idx].lastArmIdx = -1;
  if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                     "RL-TUNER: Added new key idx=%d collType=%s sizeBand=%d nodes=%d ranks=%d",
                     idx, collTypeToString(key->collType), key->sizeBand, key->nNodes, key->nRanks);
  }
  return idx;
}

// Initialize candidate arms for a key. For now we use a fixed set of
// (algo, proto) pairs that we know are interesting on modern GPUs.
static void initDefaultArmsForKey(BanditEntry* entry) {
  if (entry->numArms > 0) return;

  // Arm 0: tree + simple
  entry->arms[0].algo = NCCL_ALGO_TREE;
  entry->arms[0].proto = NCCL_PROTO_SIMPLE;
  entry->arms[0].count = 0;
  entry->arms[0].sumLatencyMs = 0.0;

  // Arm 1: tree + ll128
  entry->arms[1].algo = NCCL_ALGO_TREE;
  entry->arms[1].proto = NCCL_PROTO_LL128;
  entry->arms[1].count = 0;
  entry->arms[1].sumLatencyMs = 0.0;

  // Arm 2: ring + simple
  entry->arms[2].algo = NCCL_ALGO_RING;
  entry->arms[2].proto = NCCL_PROTO_SIMPLE;
  entry->arms[2].count = 0;
  entry->arms[2].sumLatencyMs = 0.0;

  entry->numArms = 3;
}

// Parse one reward log line:
//   collType,nBytes,nNodes,nRanks,latency_ms
static int parseRewardLine(const char* line, ncclFunc_t* collType, size_t* nBytes,
                           int* nNodes, int* nRanks, double* latencyMs) {
  char buf[MAX_LINE_LENGTH];
  strncpy(buf, line, sizeof(buf));
  buf[sizeof(buf)-1] = '\0';

  char* saveptr = NULL;
  char* token = strtok_r(buf, ",", &saveptr);
  if (!token) return 0;
  *collType = parseCollType(token);

  token = strtok_r(NULL, ",", &saveptr);
  if (!token) return 0;
  *nBytes = (size_t)strtoull(token, NULL, 10);

  token = strtok_r(NULL, ",", &saveptr);
  if (!token) return 0;
  *nNodes = atoi(token);

  token = strtok_r(NULL, ",", &saveptr);
  if (!token) return 0;
  *nRanks = atoi(token);

  token = strtok_r(NULL, ",", &saveptr);
  if (!token) return 0;
  *latencyMs = strtod(token, NULL);

  return 1;
}

// Ingest new reward lines from the reward file and update arm statistics.
static void ingestRewards(TunerContext* ctx) {
  if (ctx->rewardFile[0] == '\0') return;

  FILE* f = fopen(ctx->rewardFile, "r");
  if (!f) {
    // File might not exist yet; that's fine.
    return;
  }

  if (ctx->rewardOffset > 0) {
    if (fseek(f, ctx->rewardOffset, SEEK_SET) != 0) {
      // If seek fails, reset to beginning.
      fseek(f, 0, SEEK_SET);
      ctx->rewardOffset = 0;
    }
  }

  char line[MAX_LINE_LENGTH];
  while (fgets(line, sizeof(line), f)) {
    // Track new offset as we go.
    ctx->rewardOffset = ftell(f);

    // Skip comments / empty lines
    if (line[0] == '#' || line[0] == '\n') continue;

    ncclFunc_t collType;
    size_t nBytes;
    int nNodes, nRanks;
    double latencyMs;
    if (!parseRewardLine(line, &collType, &nBytes, &nNodes, &nRanks, &latencyMs)) {
      continue;
    }

    BanditKey key;
    key.collType = collType;
    key.sizeBand = sizeBandFromBytes(nBytes);
    key.nNodes   = nNodes;
    key.nRanks   = nRanks;

    // Find matching key
    int kIdx = -1;
    for (int i = 0; i < ctx->numKeys; ++i) {
      if (keysEqual(&ctx->entries[i].key, &key)) {
        kIdx = i;
        break;
      }
    }
    if (kIdx < 0) {
      // Reward for a key we haven't seen in this communicator; ignore.
      continue;
    }

    BanditEntry* entry = &ctx->entries[kIdx];
    if (entry->lastArmIdx < 0 || entry->lastArmIdx >= entry->numArms) {
      // We don't know which arm was used last; ignore.
      continue;
    }

    BanditArm* arm = &entry->arms[entry->lastArmIdx];
    arm->count += 1;
    arm->sumLatencyMs += latencyMs;

    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                       "RL-TUNER: Reward for key(collType=%s,band=%d,nodes=%d,ranks=%d) arm(algo=%d,proto=%d) latency=%.3f ms (N=%d)",
                       collTypeToString(key.collType), key.sizeBand, key.nNodes, key.nRanks,
                       arm->algo, arm->proto, latencyMs, arm->count);
    }
  }

  fclose(f);
}

// --- Arm selection strategies ---

// Epsilon-greedy: with probability epsilon pick random, else pick best mean.
static int selectArmEpsGreedy(TunerContext* ctx, BanditEntry* entry) {
  double r = (double)rand() / (double)RAND_MAX;
  if (r < ctx->epsilon) {
    return rand() % entry->numArms;
  }
  double bestMean = 0.0;
  int bestIdx = 0;
  for (int i = 0; i < entry->numArms; ++i) {
    double mean = entry->arms[i].sumLatencyMs / (double)entry->arms[i].count;
    if (i == 0 || mean < bestMean) {
      bestMean = mean;
      bestIdx = i;
    }
  }
  return bestIdx;
}

// UCB1 for minimisation: score_i = mean_i - c * sqrt(ln(T) / n_i).
// Pick arm with lowest score (optimistic lower bound).
static int selectArmUCB1(TunerContext* ctx, BanditEntry* entry) {
  double lnT = log((double)(ctx->totalPulls > 0 ? ctx->totalPulls : 1));
  double bestScore = 1e30;
  int bestIdx = 0;
  for (int i = 0; i < entry->numArms; ++i) {
    int ni = entry->arms[i].count;
    double mean_i = entry->arms[i].sumLatencyMs / (double)ni;
    double score = mean_i - ctx->ucbC * sqrt(lnT / (double)ni);
    if (score < bestScore) {
      bestScore = score;
      bestIdx = i;
    }
  }
  return bestIdx;
}

// Box-Muller transform for standard normal sample (no external deps).
static double randNormal(void) {
  double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
  double u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
  return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

// Thompson Sampling (Normal model) for minimisation:
// Sample from N(mean_i, 1/n_i) per arm, pick lowest sample.
static int selectArmThompson(TunerContext* ctx, BanditEntry* entry) {
  (void)ctx;
  double bestSample = 1e30;
  int bestIdx = 0;
  for (int i = 0; i < entry->numArms; ++i) {
    int ni = entry->arms[i].count;
    double mean_i = entry->arms[i].sumLatencyMs / (double)ni;
    double scale = 1.0 / sqrt((double)ni);
    double sample = mean_i + scale * randNormal();
    if (sample < bestSample) {
      bestSample = sample;
      bestIdx = i;
    }
  }
  return bestIdx;
}

// Dispatch to the configured strategy.
static int selectArm(TunerContext* ctx, BanditEntry* entry) {
  if (entry->numArms == 0) return -1;

  // Always try untried arms first (common to all strategies).
  for (int i = 0; i < entry->numArms; ++i) {
    if (entry->arms[i].count == 0) {
      return i;
    }
  }

  switch (ctx->strategy) {
    case STRATEGY_UCB1:     return selectArmUCB1(ctx, entry);
    case STRATEGY_THOMPSON: return selectArmThompson(ctx, entry);
    default:                return selectArmEpsGreedy(ctx, entry);
  }
}

__hidden ncclResult_t pluginInit(void** context, uint64_t commId, size_t nRanks, size_t nNodes,
                                 ncclDebugLogger_t logFunction,
                                 ncclNvlDomainInfo_v5_t* nvlDomainInfo,
                                 ncclTunerConstants_v5_t* constants) {
  (void)nvlDomainInfo;
  (void)constants;

  TunerContext* ctx = (TunerContext*)malloc(sizeof(TunerContext));
  if (!ctx) return ncclSystemError;

  memset(ctx, 0, sizeof(TunerContext));
  ctx->nRanks = nRanks;
  ctx->nNodes = nNodes;
  ctx->logFunction = logFunction;
  ctx->rewardOffset = 0;
  ctx->totalPulls = 0;
  ctx->epsilon = 0.1;         // default for eps_greedy
  ctx->ucbC = 1.414;          // sqrt(2) default for UCB1
  ctx->strategy = STRATEGY_EPS_GREEDY;

  // Parse NCCL_TUNER_STRATEGY
  const char* stratEnv = getenv("NCCL_TUNER_STRATEGY");
  if (stratEnv) {
    if (strcmp(stratEnv, "ucb1") == 0)          ctx->strategy = STRATEGY_UCB1;
    else if (strcmp(stratEnv, "thompson") == 0)  ctx->strategy = STRATEGY_THOMPSON;
    // else stays eps_greedy
  }

  const char* epsEnv = getenv("NCCL_TUNER_EPS");
  if (epsEnv) {
    double val = strtod(epsEnv, NULL);
    if (val >= 0.0 && val <= 1.0) {
      ctx->epsilon = val;
    }
  }

  const char* ucbEnv = getenv("NCCL_TUNER_UCB_C");
  if (ucbEnv) {
    double val = strtod(ucbEnv, NULL);
    if (val > 0.0) ctx->ucbC = val;
  }

  const char* rewardEnv = getenv("NCCL_TUNER_REWARD_FILE");
  if (rewardEnv && rewardEnv[0] != '\0') {
    strncpy(ctx->rewardFile, rewardEnv, sizeof(ctx->rewardFile));
    ctx->rewardFile[sizeof(ctx->rewardFile)-1] = '\0';
  } else {
    snprintf(ctx->rewardFile, sizeof(ctx->rewardFile),
             "/tmp/nccl_tuner_rewards_%llu.log",
             (unsigned long long)commId);
  }

  // Seed RNG with time and communicator ID.
  unsigned int seed = (unsigned int)time(NULL) ^ (unsigned int)(commId & 0xffffffffULL);
  srand(seed);

  static const char* stratNames[] = {"eps_greedy", "ucb1", "thompson"};
  if (logFunction) {
    logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                "RL-TUNER: init for %zu nodes, %zu ranks, strategy=%s, rewardFile=%s, epsilon=%.3f, ucbC=%.3f",
                nNodes, nRanks, stratNames[ctx->strategy], ctx->rewardFile, ctx->epsilon, ctx->ucbC);
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

  if (ctx->logFunction) {
    ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                     "RL-TUNER: getCollInfo collType=%s nBytes=%zu",
                     collTypeToString(collType), nBytes);
  }

  // Ingest any new rewards for this communicator.
  ingestRewards(ctx);

  BanditKey key;
  key.collType = collType;
  key.sizeBand = sizeBandFromBytes(nBytes);
  key.nNodes   = (int)ctx->nNodes;
  key.nRanks   = (int)ctx->nRanks;

  int kIdx = getOrAddKey(ctx, &key);
  if (kIdx < 0) {
    return ncclSuccess;
  }

  BanditEntry* entry = &ctx->entries[kIdx];
  initDefaultArmsForKey(entry);

  int armIdx = selectArm(ctx, entry);
  if (armIdx < 0 || armIdx >= entry->numArms) {
    return ncclSuccess;
  }

  BanditArm* arm = &entry->arms[armIdx];

  // Validate algo/proto indices against NCCL's tables.
  if (arm->algo < 0 || arm->algo >= numAlgo ||
      arm->proto < 0 || arm->proto >= numProto) {
    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                       "RL-TUNER: Selected arm out of bounds algo=%d proto=%d (numAlgo=%d numProto=%d)",
                       arm->algo, arm->proto, numAlgo, numProto);
    }
    return ncclSuccess;
  }

  if (collCostTable[arm->algo][arm->proto] == NCCL_ALGO_PROTO_IGNORE) {
    if (ctx->logFunction) {
      ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                       "RL-TUNER: Selected arm algo=%d proto=%d is IGNORE; leaving defaults",
                       arm->algo, arm->proto);
    }
    return ncclSuccess;
  }

  // Prefer this (algo, proto) by setting its cost low.
  collCostTable[arm->algo][arm->proto] = 0.0f;
  *nChannels = 1; // let NCCL adjust if desired; we only steer algo/proto

  entry->lastArmIdx = armIdx;
  ctx->totalPulls++;

  if (ctx->logFunction) {
    double mean = (arm->count > 0) ? (arm->sumLatencyMs / (double)arm->count) : -1.0;
    ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                     "RL-TUNER: Selected arm for collType=%s band=%d nodes=%d ranks=%d -> algo=%d proto=%d (N=%d mean=%.3f ms)",
                     collTypeToString(key.collType), key.sizeBand, key.nNodes, key.nRanks,
                     arm->algo, arm->proto, arm->count, mean);
  }

  return ncclSuccess;
}

__hidden ncclResult_t pluginFinalize(void* context) {
  if (context) {
    free(context);
  }
  return ncclSuccess;
}

#define PLUGIN_NAME "RLBandit"

const ncclTuner_v5_t ncclTunerPlugin_v5 = {
  .name = PLUGIN_NAME,
  .init = pluginInit,
  .getCollInfo = pluginGetCollInfo,
  .finalize = pluginFinalize
};

