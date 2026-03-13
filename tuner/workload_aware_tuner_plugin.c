// Workload-Aware NCCL Tuner Plugin
//
// A static lookup-table tuner derived from empirical measurements on 8x A100
// (NVLink, single-node). The key insight: NCCL's default AUTO selection is
// calibrated for isolated collectives, but when collectives execute concurrently
// with compute (the overlap regime typical of real training), the optimal
// (algorithm, protocol, channel count) changes.
//
// This plugin reads NCCL_OVERLAP_MODE to determine which policy branch to use:
//   0 (default) = sequential (compute and communication do not overlap)
//   1           = overlap (compute and communication run on concurrent streams)
//
// The lookup table maps (overlap_mode, message_size_band) -> (algo, proto, nChannels).
//
// Key findings encoded in the table:
//   - Under overlap, Ring+Simple dominates for messages >= 2MB
//   - Under sequential, the landscape is more varied (Tree+LL128, Ring+LL128)
//   - CTA count (nChannels) optimal point is size-dependent and differs
//     between sequential and overlap modes
//   - 6 out of 9 tested message sizes show a "winner flip" between modes
//
// Environment variables:
//   NCCL_OVERLAP_MODE: 0 (sequential) or 1 (overlap). Default: 0.
//
// Build:
//   gcc -shared -fPIC -o libnccl_workload_tuner.so workload_aware_tuner_plugin.c -I.
//
// Usage:
//   NCCL_TUNER_PLUGIN=libnccl_workload_tuner.so NCCL_OVERLAP_MODE=1 \
//     python -m torch.distributed.run --nproc_per_node=8 my_training.py
//

#include "tuner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define __hidden __attribute__ ((visibility("hidden")))

// -----------------------------------------------------------------------
// Policy table: derived from experimental data on 8x A100 NVLink
// -----------------------------------------------------------------------

#define NUM_SIZE_BANDS 6
#define NUM_MODES      2  // 0=sequential, 1=overlap

// Size banding (matches NCCL internal bands)
//   Band 0: < 1 KB          (tiny — LL protocol territory)
//   Band 1: 1 KB - 16 KB    (small — transition zone)
//   Band 2: 16 KB - 256 KB  (medium-small — LL128 territory)
//   Band 3: 256 KB - 1 MB   (medium — protocol transitions)
//   Band 4: 1 MB - 8 MB     (medium-large — Simple territory)
//   Band 5: >= 8 MB          (large — bandwidth-bound)

typedef struct {
    int algo;       // NCCL algo index, or -1 to let NCCL decide
    int proto;      // NCCL proto index, or -1 to let NCCL decide
    int nChannels;  // recommended channel count, or 0 to let NCCL decide
} PolicyEntry;

// Experimentally-derived lookup table.
// policy_table[overlap_mode][size_band]
//
// Data sources:
//   - Experiment 2: Sequential vs overlap sweep across 9 message sizes
//     (50 iters, 10 warmup, 5 algo/proto configs, compute_mul=4096)
//   - Experiment 3: CTA count sweep (1-32 CTAs × 4 sizes × 5 configs)
//     under both overlap and sequential modes
//
// For each (mode, band), the entry encodes the empirically best
// (algo, proto) combo, plus optimal CTA count when data is available.
//
static const PolicyEntry policy_table[NUM_MODES][NUM_SIZE_BANDS] = {
    // ---------------------------------------------------------------
    // Mode 0: SEQUENTIAL (no compute-communication overlap)
    //
    // Under sequential execution, NCCL AUTO is near-optimal for most
    // sizes. The biggest gains come from:
    //   - Band 3 (256KB): Ring+Simple saves 3.0% over AUTO
    //   - Band 5 (>=8MB): Ring+Simple saves 2-3% over AUTO
    //   - CTA tuning: 24 for 16MB (8.4% gain), 16 for 64MB (6.4% gain)
    //
    // Complete data: Exp 2 (9 sizes) + Exp 3 Block B (4 sizes, 8 CTA counts)
    // ---------------------------------------------------------------
    {
        { -1, -1,  0 },  // band 0 (<1KB): AUTO (no data, tiny messages)
        { -1, -1,  0 },  // band 1 (1-16KB): AUTO (marginal differences)
        {  NCCL_ALGO_TREE, NCCL_PROTO_LL128,  0 },  // band 2 (16-256KB): Tree+LL128 (1.5% gain)
        {  NCCL_ALGO_RING, NCCL_PROTO_SIMPLE,  0 },  // band 3 (256KB-1MB): Ring+Simple (3.0% gain)
        {  NCCL_ALGO_RING, NCCL_PROTO_LL128, 24 },  // band 4 (1-8MB): Ring+LL128 + 24 CTAs (3.5% gain)
        {  NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 16 },  // band 5 (>=8MB): Ring+Simple + 16 CTAs (6.4% gain at 64MB)
    },
    // ---------------------------------------------------------------
    // Mode 1: OVERLAP (compute and communication on concurrent streams)
    //
    // Under overlap, the landscape shifts significantly:
    //   - Ring+Simple dominates for medium-to-large messages (>=2MB)
    //   - AUTO gap is larger than sequential (up to 3.3% at 2MB)
    //   - Optimal CTA count varies: low CTAs for small messages (less
    //     SM contention), higher CTAs for large messages (need bandwidth)
    //   - tree_ll128 is best at 1MB (low SM footprint helps under overlap)
    // ---------------------------------------------------------------
    {
        { -1, -1,  0 },  // band 0 (<1KB): AUTO (compute-bound)
        { -1, -1,  0 },  // band 1 (1-16KB): AUTO (compute-bound)
        { -1, -1,  0 },  // band 2 (16-256KB): AUTO (gap <0.1%)
        { -1, -1,  0 },  // band 3 (256KB-1MB): AUTO (gap <0.1%)
        {  NCCL_ALGO_RING, NCCL_PROTO_SIMPLE,  4 },  // band 4 (1-8MB): Ring+Simple + 4 CTAs (3.3% gain)
        {  NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 24 },  // band 5 (>=8MB): Ring+Simple + 24 CTAs (3.0% gain)
    },
};

// -----------------------------------------------------------------------
// Tuner context
// -----------------------------------------------------------------------

typedef struct {
    int overlapMode;   // 0=sequential, 1=overlap
    size_t nRanks;
    size_t nNodes;
    ncclDebugLogger_t logFunction;
} WorkloadTunerContext;

// -----------------------------------------------------------------------
// Size band computation
// -----------------------------------------------------------------------

static int sizeBandFromBytes(size_t nBytes) {
    if (nBytes < 1024ULL)             return 0;  // < 1 KB
    if (nBytes < 16ULL * 1024)        return 1;  // 1 KB - 16 KB
    if (nBytes < 256ULL * 1024)       return 2;  // 16 KB - 256 KB
    if (nBytes < 1024ULL * 1024)      return 3;  // 256 KB - 1 MB
    if (nBytes < 8ULL * 1024 * 1024)  return 4;  // 1 MB - 8 MB
    return 5;                                     // >= 8 MB
}

static const char* algoName(int algo) {
    switch (algo) {
        case NCCL_ALGO_TREE: return "Tree";
        case NCCL_ALGO_RING: return "Ring";
        default:             return "AUTO";
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

// -----------------------------------------------------------------------
// Plugin API
// -----------------------------------------------------------------------

__hidden ncclResult_t pluginInit(void** context, uint64_t commId, size_t nRanks, size_t nNodes,
                                 ncclDebugLogger_t logFunction,
                                 ncclNvlDomainInfo_v5_t* nvlDomainInfo,
                                 ncclTunerConstants_v5_t* constants) {
    (void)commId;
    (void)nvlDomainInfo;
    (void)constants;

    WorkloadTunerContext* ctx = (WorkloadTunerContext*)malloc(sizeof(WorkloadTunerContext));
    if (!ctx) return ncclSystemError;

    memset(ctx, 0, sizeof(WorkloadTunerContext));
    ctx->nRanks = nRanks;
    ctx->nNodes = nNodes;
    ctx->logFunction = logFunction;

    // Read overlap mode from environment
    ctx->overlapMode = 0;
    const char* modeEnv = getenv("NCCL_OVERLAP_MODE");
    if (modeEnv) {
        int val = atoi(modeEnv);
        if (val == 1) ctx->overlapMode = 1;
    }

    if (logFunction) {
        logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                    "WORKLOAD-TUNER: init for %zu nodes, %zu ranks, overlap_mode=%d",
                    nNodes, nRanks, ctx->overlapMode);
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

    WorkloadTunerContext* ctx = (WorkloadTunerContext*)context;
    if (!ctx) return ncclInternalError;

    // Only tune AllReduce for now (our experiments focused on AllReduce)
    if (collType != ncclFuncAllReduce) {
        return ncclSuccess;  // let NCCL decide for other collectives
    }

    // Only tune single-node for now
    if (ctx->nNodes > 1) {
        return ncclSuccess;  // multi-node policy TBD
    }

    int band = sizeBandFromBytes(nBytes);
    int mode = ctx->overlapMode;

    if (mode < 0 || mode >= NUM_MODES || band < 0 || band >= NUM_SIZE_BANDS) {
        return ncclSuccess;
    }

    const PolicyEntry* entry = &policy_table[mode][band];

    // If policy says -1 (AUTO), let NCCL decide
    if (entry->algo < 0 || entry->proto < 0) {
        if (ctx->logFunction) {
            ctx->logFunction(NCCL_LOG_TRACE, NCCL_TUNING, __FILE__, __LINE__,
                             "WORKLOAD-TUNER: nBytes=%zu band=%d mode=%d -> AUTO (no override)",
                             nBytes, band, mode);
        }
        return ncclSuccess;
    }

    // Validate algo/proto against NCCL's tables
    if (entry->algo >= numAlgo || entry->proto >= numProto) {
        if (ctx->logFunction) {
            ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                             "WORKLOAD-TUNER: algo=%d or proto=%d out of bounds (numAlgo=%d, numProto=%d)",
                             entry->algo, entry->proto, numAlgo, numProto);
        }
        return ncclSuccess;
    }

    // Check if this algo/proto combo is available (not IGNORE)
    if (collCostTable[entry->algo][entry->proto] == NCCL_ALGO_PROTO_IGNORE) {
        if (ctx->logFunction) {
            ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                             "WORKLOAD-TUNER: preferred %s+%s is IGNORE for nBytes=%zu; falling back to AUTO",
                             algoName(entry->algo), protoName(entry->proto), nBytes);
        }
        return ncclSuccess;
    }

    // Set our preferred (algo, proto) as lowest cost
    collCostTable[entry->algo][entry->proto] = 0.0f;

    // Set channel count if we have a recommendation
    if (entry->nChannels > 0) {
        *nChannels = entry->nChannels;
    }

    if (ctx->logFunction) {
        ctx->logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                         "WORKLOAD-TUNER: nBytes=%zu band=%d mode=%s -> %s+%s nCh=%d",
                         nBytes, band,
                         mode == 0 ? "SEQ" : "OVL",
                         algoName(entry->algo), protoName(entry->proto),
                         entry->nChannels);
    }

    return ncclSuccess;
}

__hidden ncclResult_t pluginFinalize(void* context) {
    if (context) {
        free(context);
    }
    return ncclSuccess;
}

#define PLUGIN_NAME "WorkloadAware"

const ncclTuner_v5_t ncclTunerPlugin_v5 = {
    .name = PLUGIN_NAME,
    .init = pluginInit,
    .getCollInfo = pluginGetCollInfo,
    .finalize = pluginFinalize
};
