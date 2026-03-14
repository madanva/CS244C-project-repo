// =======================================================================
// Workload-Aware NCCL Tuner Plugin v2
// =======================================================================
//
// An NCCL tuner plugin that fixes three blind spots in NCCL's cost model:
//
//   1. TOPOLOGY: NCCL AUTO mispredicts inter-node costs — Tree+Simple
//      beats AUTO by up to 57% at 64MB multi-node because Ring forces
//      every hop across the slow inter-node link.
//
//   2. OVERLAP: When compute and communication run on concurrent CUDA
//      streams, SM contention changes the optimal (algo, proto, CTA).
//      6/9 message sizes "flip" their optimal config under overlap on
//      single-node NVLink.
//
//   3. CTA COUNT: NCCL uses a fixed CTA (channel) count. Our data shows
//      3-8% gains from size-dependent and mode-dependent CTA tuning.
//
// How it works:
//   - At init: auto-detects topology (single vs multi-node) from nNodes
//   - At runtime: auto-detects overlap via call timing heuristic
//   - Per collective: looks up optimal (algo, proto, nChannels) from
//     experimentally-derived tables indexed by (topology, overlap, size)
//   - Falls back gracefully: if preferred config is unavailable (IGNORE),
//     tries ranked alternatives instead of surrendering to AUTO
//
// Data sources:
//   - Single-node: 8x A100-SXM4-80GB NVLink, 9 sizes × 5 configs × 2 modes
//   - Multi-node: 2×4 A100 over network, 6 sizes × 5 configs × 2 modes
//   - CTA sweep: 8 CTA counts × 4 sizes × 5 configs × 2 modes
//
// Environment variables (all optional):
//   NCCL_OVERLAP_MODE: Override auto-detection. 0=sequential, 1=overlap.
//   NCCL_TUNER_LOG:    Set to 1 for verbose per-collective logging.
//
// Build:
//   gcc -shared -fPIC -o libnccl_tuner_v2.so workload_aware_tuner_v2.c -I.
//
// Usage:
//   NCCL_TUNER_PLUGIN=./libnccl_tuner_v2.so torchrun --nproc_per_node=8 train.py
//
// =======================================================================

#include "tuner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define __hidden __attribute__((visibility("hidden")))

// =======================================================================
// Topology modes (auto-detected from nNodes at init)
// =======================================================================
#define TOPO_SINGLE_NODE  0   // All GPUs on one node (NVLink)
#define TOPO_MULTI_NODE   1   // GPUs span multiple nodes (network)
#define NUM_TOPOS         2

// =======================================================================
// Overlap modes
// =======================================================================
#define MODE_SEQUENTIAL   0   // Compute then communicate
#define MODE_OVERLAP      1   // Compute and communicate concurrently
#define NUM_MODES         2

// =======================================================================
// Size bands — 8 bands for finer-grained control
// =======================================================================
#define NUM_SIZE_BANDS    8

// Band boundaries (bytes):
//   0: < 1 KB           (control messages, tiny)
//   1: 1 KB - 32 KB     (small, LL territory)
//   2: 32 KB - 256 KB   (medium-small, LL128 transition)
//   3: 256 KB - 1 MB    (medium, protocol transitions)
//   4: 1 MB - 4 MB      (medium-large)
//   5: 4 MB - 16 MB     (large)
//   6: 16 MB - 128 MB   (very large, bandwidth-bound)
//   7: >= 128 MB         (huge)

static int sizeBand(size_t nBytes) {
    if (nBytes < 1024ULL)                   return 0;
    if (nBytes < 32ULL * 1024)              return 1;
    if (nBytes < 256ULL * 1024)             return 2;
    if (nBytes < 1024ULL * 1024)            return 3;
    if (nBytes < 4ULL * 1024 * 1024)        return 4;
    if (nBytes < 16ULL * 1024 * 1024)       return 5;
    if (nBytes < 128ULL * 1024 * 1024)      return 6;
    return 7;
}

// =======================================================================
// Policy entry: what to recommend for a given (topo, mode, band)
// =======================================================================
typedef struct {
    int algo;        // NCCL_ALGO_*, or -1 for AUTO
    int proto;       // NCCL_PROTO_*, or -1 for AUTO
    int nChannels;   // Recommended CTA/channel count, 0 = let NCCL decide
} PolicyEntry;

// Fallback entry: if preferred is IGNORE, try these in order
typedef struct {
    PolicyEntry preferred;
    PolicyEntry fallback1;
    PolicyEntry fallback2;
} PolicyWithFallback;

// =======================================================================
// POLICY TABLES — derived from experimental data
// =======================================================================
//
// Notation: T=Tree, R=Ring, S=Simple, L=LL128
// AUTO = {-1, -1, 0} = let NCCL decide
//
// Each table is indexed: [overlap_mode][size_band]
//

// -----------------------------------------------------------------------
// SINGLE-NODE policy (8x A100 NVLink)
//
// Key findings:
//   - Sequential: varied winners, Ring+LL128 and Tree+LL128 trade off
//   - Overlap: Ring+Simple dominates >= 2MB (lower SM footprint)
//   - CTA tuning: 3-8% additional gain, mode-dependent
//   - 6/9 sizes flip between sequential and overlap
//   - Overall AUTO gaps: 1-3% (modest on NVLink)
// -----------------------------------------------------------------------
static const PolicyWithFallback singleNodePolicy[NUM_MODES][NUM_SIZE_BANDS] = {
    // --- SEQUENTIAL (single-node) ---
    {
        // Band 0 (<1KB): tiny, AUTO is fine
        { {-1, -1, 0},  {-1, -1, 0},  {-1, -1, 0} },
        // Band 1 (1-32KB): AUTO, marginal differences
        { {-1, -1, 0},  {-1, -1, 0},  {-1, -1, 0} },
        // Band 2 (32-256KB): Tree+LL128 wins (+1.5%), fallback Tree+Simple
        { {NCCL_ALGO_TREE, NCCL_PROTO_LL128, 0},
          {NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, 0},
          {-1, -1, 0} },
        // Band 3 (256KB-1MB): Ring+Simple wins (+3.0%), fallback Ring+LL128
        { {NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 0},
          {NCCL_ALGO_RING, NCCL_PROTO_LL128, 0},
          {-1, -1, 0} },
        // Band 4 (1-4MB): Ring+LL128 + 24 CTAs (+3.5%)
        { {NCCL_ALGO_RING, NCCL_PROTO_LL128, 24},
          {NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 0},
          {NCCL_ALGO_TREE, NCCL_PROTO_LL128, 0} },
        // Band 5 (4-16MB): Ring+Simple + 24 CTAs (+3.1%)
        { {NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 24},
          {NCCL_ALGO_RING, NCCL_PROTO_LL128, 0},
          {-1, -1, 0} },
        // Band 6 (16-128MB): Ring+Simple + 16 CTAs (+6.4% at 64MB)
        { {NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 16},
          {NCCL_ALGO_RING, NCCL_PROTO_LL128, 0},
          {-1, -1, 0} },
        // Band 7 (>=128MB): AUTO (near-optimal at 256MB)
        { {-1, -1, 0},  {-1, -1, 0},  {-1, -1, 0} },
    },
    // --- OVERLAP (single-node) ---
    {
        // Band 0 (<1KB): compute-bound, AUTO
        { {-1, -1, 0},  {-1, -1, 0},  {-1, -1, 0} },
        // Band 1 (1-32KB): compute-bound, AUTO
        { {-1, -1, 0},  {-1, -1, 0},  {-1, -1, 0} },
        // Band 2 (32-256KB): AUTO (gap <0.1% under overlap)
        { {-1, -1, 0},  {-1, -1, 0},  {-1, -1, 0} },
        // Band 3 (256KB-1MB): Tree+LL128 wins at 1MB overlap (+1.3%)
        { {NCCL_ALGO_TREE, NCCL_PROTO_LL128, 0},
          {-1, -1, 0},
          {-1, -1, 0} },
        // Band 4 (1-4MB): Ring+Simple + 4 CTAs (+3.3%)
        //   Low CTAs optimal — less SM contention with compute
        { {NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 4},
          {NCCL_ALGO_RING, NCCL_PROTO_LL128, 4},
          {-1, -1, 0} },
        // Band 5 (4-16MB): Ring+Simple + 12 CTAs
        { {NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 12},
          {NCCL_ALGO_TREE, NCCL_PROTO_LL128, 0},
          {-1, -1, 0} },
        // Band 6 (16-128MB): Ring+Simple + 24 CTAs (+3.0%)
        { {NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 24},
          {NCCL_ALGO_RING, NCCL_PROTO_LL128, 0},
          {-1, -1, 0} },
        // Band 7 (>=128MB): Ring+Simple (overlap makes this dominant)
        { {NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 0},
          {-1, -1, 0},
          {-1, -1, 0} },
    },
};

// -----------------------------------------------------------------------
// MULTI-NODE policy (2×4 A100, inter-node network)
//
// Key findings:
//   - Tree+Simple DOMINATES — beats AUTO by 9-57% across most sizes
//   - Ring is catastrophic: 1.6-2.9x slower than Tree at large messages
//     because every Ring hop crosses the slow inter-node link
//   - AUTO gap is 10-57% (vs 1-3% single-node) — massive
//   - The hierarchy (NVLink intra + network inter) strongly favors Tree
//   - Overlap has minimal extra effect — topology dominates
// -----------------------------------------------------------------------
static const PolicyWithFallback multiNodePolicy[NUM_MODES][NUM_SIZE_BANDS] = {
    // --- SEQUENTIAL (multi-node) ---
    {
        // Band 0 (<1KB): AUTO for tiny
        { {-1, -1, 0},  {-1, -1, 0},  {-1, -1, 0} },
        // Band 1 (1-32KB): AUTO
        { {-1, -1, 0},  {-1, -1, 0},  {-1, -1, 0} },
        // Band 2 (32-256KB): Tree+Simple (+9.1% at 256KB)
        { {NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, 0},
          {NCCL_ALGO_TREE, NCCL_PROTO_LL128, 0},
          {-1, -1, 0} },
        // Band 3 (256KB-1MB): Tree+Simple (+13.2% at 1MB)
        { {NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, 0},
          {NCCL_ALGO_TREE, NCCL_PROTO_LL128, 0},
          {-1, -1, 0} },
        // Band 4 (1-4MB): Tree+Simple (AUTO is near-optimal at 4MB, but
        //   Tree+Simple never hurts)
        { {NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, 0},
          {-1, -1, 0},
          {-1, -1, 0} },
        // Band 5 (4-16MB): Tree+Simple (+15.9% at 16MB)
        { {NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, 0},
          {NCCL_ALGO_TREE, NCCL_PROTO_LL128, 0},
          {-1, -1, 0} },
        // Band 6 (16-128MB): Tree+Simple (+57.2% at 64MB !!!)
        //   THE headline result. AUTO picks something 2.3x slower.
        { {NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, 0},
          {NCCL_ALGO_RING, NCCL_PROTO_LL128, 0},
          {-1, -1, 0} },
        // Band 7 (>=128MB): Tree+Simple (+0.5% at 256MB — marginal but safe)
        { {NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, 0},
          {NCCL_ALGO_TREE, NCCL_PROTO_LL128, 0},
          {-1, -1, 0} },
    },
    // --- OVERLAP (multi-node) ---
    {
        // Band 0 (<1KB): AUTO
        { {-1, -1, 0},  {-1, -1, 0},  {-1, -1, 0} },
        // Band 1 (1-32KB): AUTO
        { {-1, -1, 0},  {-1, -1, 0},  {-1, -1, 0} },
        // Band 2 (32-256KB): Tree+Simple (+10.9% at 256KB)
        { {NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, 0},
          {NCCL_ALGO_RING, NCCL_PROTO_LL128, 0},
          {-1, -1, 0} },
        // Band 3 (256KB-1MB): Tree+Simple (+13.5% at 1MB)
        { {NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, 0},
          {NCCL_ALGO_TREE, NCCL_PROTO_LL128, 0},
          {-1, -1, 0} },
        // Band 4 (1-4MB): Tree+Simple (+6.4% at 4MB — overlap creates gap)
        { {NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, 0},
          {NCCL_ALGO_TREE, NCCL_PROTO_LL128, 0},
          {-1, -1, 0} },
        // Band 5 (4-16MB): Tree+Simple (+11.5% at 16MB)
        { {NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, 0},
          {NCCL_ALGO_TREE, NCCL_PROTO_LL128, 0},
          {-1, -1, 0} },
        // Band 6 (16-128MB): Tree+Simple (+45.6% at 64MB)
        { {NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, 0},
          {NCCL_ALGO_TREE, NCCL_PROTO_LL128, 0},
          {-1, -1, 0} },
        // Band 7 (>=128MB): Tree+LL128 (wins at 256MB overlap, marginal)
        { {NCCL_ALGO_TREE, NCCL_PROTO_LL128, 0},
          {NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, 0},
          {-1, -1, 0} },
    },
};

// =======================================================================
// Overlap auto-detection via call timing
// =======================================================================
//
// Heuristic: In real training, the gap between consecutive allreduce calls
// follows one of two patterns:
//
//   Sequential: long gaps (compute finishes, then collective starts)
//     gap ~ compute_time + comm_time (tens of ms)
//
//   Overlap: short, consistent gaps (collective launched right after compute
//     starts on another stream — they're pipelined)
//     gap ~ max(compute_time, comm_time) — shorter and steadier
//
// We track the variance of inter-call gaps. Low variance = pipelined = overlap.
// High variance = sequential/irregular.
//
// This is a heuristic — the env var NCCL_OVERLAP_MODE overrides it.
//

#define TIMING_WINDOW    32    // Rolling window for gap tracking
#define OVERLAP_THRESHOLD 0.15 // CV (coeff of variation) below this → overlap

typedef struct {
    // Config
    int topoMode;           // TOPO_SINGLE_NODE or TOPO_MULTI_NODE
    int overlapMode;        // MODE_SEQUENTIAL or MODE_OVERLAP
    int overlapOverride;    // 1 if set via env var (skip auto-detection)
    int verboseLog;         // 1 for per-collective logging
    size_t nRanks;
    size_t nNodes;
    ncclDebugLogger_t logFn;

    // Overlap auto-detection state
    struct timespec lastCallTime;
    int hasLastCall;
    double gapHistory[TIMING_WINDOW];
    int gapCount;
    int gapIdx;
    int detectionDone;      // 1 once we've locked in a mode
    int callCount;          // total getCollInfo calls

    // Stats
    uint64_t overrideCount; // times we overrode NCCL
    uint64_t fallbackCount; // times preferred was IGNORE
    uint64_t autoCount;     // times we let NCCL decide
} TunerContext;

// =======================================================================
// Timing helpers
// =======================================================================

static double timespec_diff_ms(struct timespec* a, struct timespec* b) {
    return (b->tv_sec - a->tv_sec) * 1000.0 +
           (b->tv_nsec - a->tv_nsec) / 1000000.0;
}

static void updateOverlapDetection(TunerContext* ctx) {
    if (ctx->overlapOverride || ctx->detectionDone) return;
    if (ctx->gapCount < TIMING_WINDOW) return;

    // Compute mean and std of gap times
    double sum = 0, sum2 = 0;
    for (int i = 0; i < TIMING_WINDOW; i++) {
        sum += ctx->gapHistory[i];
        sum2 += ctx->gapHistory[i] * ctx->gapHistory[i];
    }
    double mean = sum / TIMING_WINDOW;
    double var = sum2 / TIMING_WINDOW - mean * mean;
    if (var < 0) var = 0;
    double std = 0;
    // Manual sqrt (avoid linking libm)
    if (var > 0) {
        std = var;
        for (int i = 0; i < 20; i++) std = 0.5 * (std + var / std);
    }
    double cv = (mean > 0.001) ? std / mean : 999.0;

    int detected = (cv < OVERLAP_THRESHOLD) ? MODE_OVERLAP : MODE_SEQUENTIAL;

    if (detected != ctx->overlapMode && ctx->logFn) {
        ctx->logFn(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                   "TUNER-V2: overlap auto-detect: cv=%.3f mean_gap=%.2fms -> %s",
                   cv, mean, detected == MODE_OVERLAP ? "OVERLAP" : "SEQUENTIAL");
    }

    ctx->overlapMode = detected;

    // Re-check every TIMING_WINDOW calls (don't lock in permanently,
    // workload phase can change)
    ctx->gapCount = 0;
    ctx->gapIdx = 0;
}

// =======================================================================
// Helper: apply a policy entry with fallback chain
// =======================================================================

static int tryApplyEntry(const PolicyEntry* entry,
                         float** collCostTable, int numAlgo, int numProto,
                         int* nChannels) {
    // AUTO entry — skip
    if (entry->algo < 0 || entry->proto < 0) return 0;

    // Bounds check
    if (entry->algo >= numAlgo || entry->proto >= numProto) return 0;

    // Check availability
    if (collCostTable[entry->algo][entry->proto] == NCCL_ALGO_PROTO_IGNORE) return 0;

    // Apply: set our preferred as lowest cost
    collCostTable[entry->algo][entry->proto] = 0.0f;

    // Set channel count if recommended
    if (entry->nChannels > 0) {
        *nChannels = entry->nChannels;
    }

    return 1;  // success
}

// =======================================================================
// Name helpers
// =======================================================================

static const char* algoName(int algo) {
    switch (algo) {
        case NCCL_ALGO_TREE:           return "Tree";
        case NCCL_ALGO_RING:           return "Ring";
        case NCCL_ALGO_COLLNET_DIRECT: return "CollNetDirect";
        case NCCL_ALGO_COLLNET_CHAIN:  return "CollNetChain";
        case NCCL_ALGO_NVLS:           return "NVLS";
        case NCCL_ALGO_NVLS_TREE:      return "NVLSTree";
        case NCCL_ALGO_PAT:            return "PAT";
        default:                        return "AUTO";
    }
}

static const char* protoName(int proto) {
    switch (proto) {
        case NCCL_PROTO_LL:     return "LL";
        case NCCL_PROTO_LL128:  return "LL128";
        case NCCL_PROTO_SIMPLE: return "Simple";
        default:                return "AUTO";
    }
}

static const char* topoName(int topo) {
    return topo == TOPO_MULTI_NODE ? "MULTI" : "SINGLE";
}

static const char* modeName(int mode) {
    return mode == MODE_OVERLAP ? "OVL" : "SEQ";
}

// =======================================================================
// Plugin API: init
// =======================================================================

__hidden ncclResult_t pluginInit(void** context, uint64_t commId,
                                 size_t nRanks, size_t nNodes,
                                 ncclDebugLogger_t logFunction,
                                 ncclNvlDomainInfo_v5_t* nvlDomainInfo,
                                 ncclTunerConstants_v5_t* constants) {
    (void)commId;
    (void)nvlDomainInfo;
    (void)constants;

    TunerContext* ctx = (TunerContext*)calloc(1, sizeof(TunerContext));
    if (!ctx) return ncclSystemError;

    ctx->nRanks = nRanks;
    ctx->nNodes = nNodes;
    ctx->logFn = logFunction;

    // --- Auto-detect topology ---
    ctx->topoMode = (nNodes > 1) ? TOPO_MULTI_NODE : TOPO_SINGLE_NODE;

    // --- Overlap mode: env var override or auto-detect ---
    ctx->overlapMode = MODE_SEQUENTIAL;  // safe default
    ctx->overlapOverride = 0;

    const char* modeEnv = getenv("NCCL_OVERLAP_MODE");
    if (modeEnv) {
        ctx->overlapMode = (atoi(modeEnv) == 1) ? MODE_OVERLAP : MODE_SEQUENTIAL;
        ctx->overlapOverride = 1;
    }

    // --- Verbose logging ---
    const char* logEnv = getenv("NCCL_TUNER_LOG");
    ctx->verboseLog = (logEnv && atoi(logEnv) == 1) ? 1 : 0;

    if (logFunction) {
        logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                    "TUNER-V2: init | nodes=%zu ranks=%zu topo=%s "
                    "overlap=%s%s",
                    nNodes, nRanks, topoName(ctx->topoMode),
                    modeName(ctx->overlapMode),
                    ctx->overlapOverride ? " (env override)" : " (auto-detecting)");

        if (ctx->topoMode == TOPO_MULTI_NODE) {
            logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                        "TUNER-V2: multi-node detected — Tree+Simple policy active "
                        "(up to 57%% improvement over AUTO at 64MB)");
        }
    }

    *context = ctx;
    return ncclSuccess;
}

// =======================================================================
// Plugin API: getCollInfo
// =======================================================================

__hidden ncclResult_t pluginGetCollInfo(void* context, ncclFunc_t collType,
                                        size_t nBytes, int numPipeOps,
                                        float** collCostTable,
                                        int numAlgo, int numProto,
                                        int regBuff, int* nChannels) {
    (void)regBuff;

    TunerContext* ctx = (TunerContext*)context;
    if (!ctx) return ncclInternalError;

    ctx->callCount++;

    // --- Overlap auto-detection: track timing ---
    if (!ctx->overlapOverride) {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);

        if (ctx->hasLastCall) {
            double gap = timespec_diff_ms(&ctx->lastCallTime, &now);
            // Only track reasonable gaps (ignore long pauses like init)
            if (gap > 0.01 && gap < 5000.0) {
                ctx->gapHistory[ctx->gapIdx % TIMING_WINDOW] = gap;
                ctx->gapIdx++;
                if (ctx->gapCount < TIMING_WINDOW) ctx->gapCount++;
            }
        }
        ctx->lastCallTime = now;
        ctx->hasLastCall = 1;

        // Update detection periodically
        if (ctx->gapCount >= TIMING_WINDOW) {
            updateOverlapDetection(ctx);
        }
    }

    // --- Only tune AllReduce (our experimental data) ---
    // For other collectives, let NCCL decide. Future work: AllGather,
    // ReduceScatter have different profiles.
    if (collType != ncclFuncAllReduce) {
        return ncclSuccess;
    }

    // --- Look up policy ---
    int topo = ctx->topoMode;
    int mode = ctx->overlapMode;
    int band = sizeBand(nBytes);

    const PolicyWithFallback* pwf;
    if (topo == TOPO_MULTI_NODE) {
        pwf = &multiNodePolicy[mode][band];
    } else {
        pwf = &singleNodePolicy[mode][band];
    }

    // --- Try preferred, then fallbacks ---
    const PolicyEntry* applied = NULL;
    int tier = 0;

    if (tryApplyEntry(&pwf->preferred, collCostTable, numAlgo, numProto, nChannels)) {
        applied = &pwf->preferred;
        tier = 1;
    } else if (tryApplyEntry(&pwf->fallback1, collCostTable, numAlgo, numProto, nChannels)) {
        applied = &pwf->fallback1;
        tier = 2;
        ctx->fallbackCount++;
    } else if (tryApplyEntry(&pwf->fallback2, collCostTable, numAlgo, numProto, nChannels)) {
        applied = &pwf->fallback2;
        tier = 3;
        ctx->fallbackCount++;
    }

    // --- Logging ---
    if (applied) {
        ctx->overrideCount++;
        if (ctx->verboseLog && ctx->logFn) {
            ctx->logFn(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                       "TUNER-V2: %s/%s band=%d nBytes=%zu -> %s+%s nCh=%d (tier %d)",
                       topoName(topo), modeName(mode), band, nBytes,
                       algoName(applied->algo), protoName(applied->proto),
                       applied->nChannels, tier);
        }
    } else {
        ctx->autoCount++;
        if (ctx->verboseLog && ctx->logFn) {
            ctx->logFn(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                       "TUNER-V2: %s/%s band=%d nBytes=%zu -> AUTO",
                       topoName(topo), modeName(mode), band, nBytes);
        }
    }

    return ncclSuccess;
}

// =======================================================================
// Plugin API: finalize
// =======================================================================

__hidden ncclResult_t pluginFinalize(void* context) {
    TunerContext* ctx = (TunerContext*)context;
    if (ctx) {
        if (ctx->logFn) {
            ctx->logFn(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                       "TUNER-V2: finalize | topo=%s overlap=%s "
                       "overrides=%llu fallbacks=%llu auto=%llu calls=%d",
                       topoName(ctx->topoMode), modeName(ctx->overlapMode),
                       (unsigned long long)ctx->overrideCount,
                       (unsigned long long)ctx->fallbackCount,
                       (unsigned long long)ctx->autoCount,
                       ctx->callCount);
        }
        free(ctx);
    }
    return ncclSuccess;
}

// =======================================================================
// Plugin export symbol
// =======================================================================

#define PLUGIN_NAME "WorkloadAware-v2"

const ncclTuner_v5_t ncclTunerPlugin_v5 = {
    .name = PLUGIN_NAME,
    .init = pluginInit,
    .getCollInfo = pluginGetCollInfo,
    .finalize = pluginFinalize
};
