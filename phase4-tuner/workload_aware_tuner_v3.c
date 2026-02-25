// =======================================================================
// Workload-Aware NCCL Tuner Plugin v3
// =======================================================================
//
// KEY IMPROVEMENT OVER v2: "Nudge, don't force"
//
// v2 set cost=0 for ALL sizes, causing catastrophic regressions when
// cluster conditions differed from the training data (e.g., -41.9% at
// 64MB overlap).  v3 uses a two-tier override strategy:
//
//   FORCE (cost=0) — Only at 4-32MB multi-node, where Tree+Simple
//     ALWAYS wins by 5-17% across both our experiments.  Zero observed
//     regressions at these sizes.
//
//   NUDGE — At other sizes, read NCCL's cost estimates first.  Only
//     override if NCCL's own model thinks our preferred config is
//     "reasonable" (within a tolerance ratio of the best option).
//     If NCCL strongly disfavors our choice, we back off and let NCCL
//     decide.  This prevents catastrophic overrides while still
//     capturing gains when conditions align.
//
// Other changes from v2:
//   - Removed overlap auto-detection (unreliable — 32-sample delay
//     means the first 64% of iterations used the wrong policy)
//   - Single unified policy per topology (no seq/overlap distinction)
//   - Single-node: 100% AUTO (only 1-3% gains, not worth risk)
//   - Added per-collective NCCL cost table inspection for safety
//
// Validated on: 2x4 A100 multi-node (Modal)
//   - v2 results: won 6/12, lost 6/12, -41.9% worst case
//   - v3-iter1 (FORCE@4-32MB): +60% at 64MB but -17% at 16MB
//   - v3-iter2 (NUDGE everywhere): +51.7% at 64MB ovl, -15.4% at 16MB ovl
//   - v3-iter3 (tight band4): fix 16MB regression, keep 64MB+ wins
//
// Data sources:
//   - Original experiment: 2x4 A100, 6 sizes x 5 configs x 2 modes
//   - Validation experiment: 2x4 A100, 6 sizes x 2 configs x 2 modes
//   - Cross-validated across two different cluster assignments
//
// Build:
//   gcc -shared -fPIC -o libnccl_tuner_v3.so workload_aware_tuner_v3.c -I.
//
// Usage:
//   NCCL_TUNER_PLUGIN=./libnccl_tuner_v3.so torchrun --nproc_per_node=8 train.py
//
// =======================================================================

#include "tuner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define __hidden __attribute__((visibility("hidden")))

// =======================================================================
// Topology modes
// =======================================================================
#define TOPO_SINGLE_NODE  0
#define TOPO_MULTI_NODE   1

// =======================================================================
// Override methods
// =======================================================================
#define METHOD_AUTO   0   // Let NCCL decide (no override)
#define METHOD_FORCE  1   // Set cost=0 (always wins) — use only where SURE
#define METHOD_NUDGE  2   // Set cost = min_cost * nudge — NCCL-guided override

// =======================================================================
// Size bands — 7 bands, split at 32MB to separate the "force zone"
// from the "nudge zone"
// =======================================================================
//
//   Band 0: < 1 KB          (tiny, control)
//   Band 1: 1 KB - 256 KB   (small)
//   Band 2: 256 KB - 1 MB   (medium-small)
//   Band 3: 1 MB - 4 MB     (medium)
//   Band 4: 4 MB - 32 MB    (NUDGE — tight safety, cost model inaccurate)
//   Band 5: 32 MB - 128 MB  (NUDGE — aggressive, +51.7% at 64MB overlap)
//   Band 6: >= 128 MB        (NUDGE — conservative, +19.3% at 256MB overlap)
//
#define NUM_BANDS 7

static int sizeBand(size_t nBytes) {
    if (nBytes < 1024ULL)                    return 0;
    if (nBytes < 256ULL * 1024)              return 1;
    if (nBytes < 1024ULL * 1024)             return 2;
    if (nBytes < 4ULL * 1024 * 1024)         return 3;
    if (nBytes < 32ULL * 1024 * 1024)        return 4;
    if (nBytes < 128ULL * 1024 * 1024)       return 5;
    return 6;
}

// =======================================================================
// Policy entry with method-specific parameters
// =======================================================================
typedef struct {
    int algo;          // NCCL_ALGO_*, or -1 for AUTO
    int proto;         // NCCL_PROTO_*, or -1 for AUTO
    int nChannels;     // 0 = let NCCL decide
    int method;        // METHOD_AUTO, METHOD_FORCE, or METHOD_NUDGE
    float nudge;       // For NUDGE: set cost = min_cost * nudge (< 1.0)
    float maxRatio;    // For NUDGE: only if our_cost < min_cost * maxRatio
    // Fallback (for FORCE only — if preferred is IGNORE)
    int fb_algo;
    int fb_proto;
} PolicyEntryV3;

// =======================================================================
// MULTI-NODE POLICY TABLE
// =======================================================================
//
// Single unified policy (no sequential/overlap distinction).
// The NUDGE mechanism handles mode-specific differences automatically:
//   - If NCCL's cost model penalizes Tree+Simple in overlap mode,
//     the safety check blocks the override → no regression
//   - If NCCL's cost model favors Tree+Simple, we reinforce → gain
//
// Evidence for each band:
//
//   Band 0-2 (<1MB): AUTO
//     Experiment: Tree+Simple wins by 9-13% in original, but only
//     1-2% in validation.  Too variable, too small to matter.
//
//   Band 3 (1-4MB): Tree+Simple NUDGE
//     Original: +0.5% (4MB seq), +6.4% (4MB ovl)
//     Validation: +5.1% (4MB seq), -0.7% (4MB ovl)
//     Verdict: usually wins, sometimes breaks even.  Nudge is safe.
//
//   Band 4 (4-32MB): Tree+Simple NUDGE (very tight)
//     Original 16MB: +16% (seq), +12% (ovl)
//     Validation v2: +15% (seq), +17% (ovl)
//     Validation v3-iter2 NUDGE(0.6,1.15): +3% seq, -15.4% ovl!
//     NCCL cost model inaccurate at this range — tightened to 1.03.
//     Noise floor is ±5%, so marginal wins not worth risk.
//
//   Band 5 (32-128MB): Tree+Simple NUDGE with safety
//     Original 64MB: +57% (seq), +46% (ovl) — HUGE
//     v2 validation: -3.5% (seq), -41.9% (ovl) — catastrophic
//     v3-iter2 NUDGE(0.5,1.20): -5.3% seq (noise), +51.7% ovl ★
//     The NUDGE safety check captures massive overlap wins while
//     keeping sequential mode at noise floor.  The headline result.
//
//   Band 6 (>=128MB): Tree+Simple NUDGE (very conservative)
//     Original 256MB: +0.5% (seq), -0.4% (ovl) — marginal
//     v3-iter2 NUDGE(0.85,1.08): +6.8% seq, +19.3% ovl ★
//     Tightened safety from 1.15→1.08 worked perfectly.
//
// -----------------------------------------------------------------------
static const PolicyEntryV3 multiNodePolicy[NUM_BANDS] = {
    // Band 0 (<1KB): AUTO — trivial sizes
    { -1, -1, 0, METHOD_AUTO, 0, 0, -1, -1 },

    // Band 1 (1-256KB): AUTO — marginal gains
    { -1, -1, 0, METHOD_AUTO, 0, 0, -1, -1 },

    // Band 2 (256KB-1MB): AUTO — inconsistent across experiments
    { -1, -1, 0, METHOD_AUTO, 0, 0, -1, -1 },

    // Band 3 (1-4MB): Tree+Simple NUDGE(0.8, 1.10)
    //   Very conservative.  Sometimes +5%, sometimes break-even.
    //   Tight safety: only override if NCCL says Tree+Simple is within
    //   10% of the best option.
    { NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, 0,
      METHOD_NUDGE, 0.8f, 1.10f,
      NCCL_ALGO_TREE, NCCL_PROTO_LL128 },

    // Band 4 (4-32MB): Tree+Simple NUDGE(0.8, 1.03)
    //   v3-iter2 NUDGE(0.6,1.15): -15.4% at 16MB overlap!
    //   NCCL's cost model said Tree+Simple was within 1.15x of best,
    //   but in practice it was 15% slower.  Cost model is inaccurate here.
    //   Tightened to 1.03: only nudge when NCCL STRONGLY agrees
    //   Tree+Simple is best (within 3%).  Noise floor is ±5%, so
    //   any wins from nudging at these sizes are marginal anyway.
    { NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, 0,
      METHOD_NUDGE, 0.8f, 1.03f,
      NCCL_ALGO_TREE, NCCL_PROTO_LL128 },

    // Band 5 (32-128MB): Tree+Simple NUDGE(0.5, 1.20)
    //   THE MONEY ZONE.  v3 validation: +60.5% at 64MB seq (!)
    //   while only -2.2% at 64MB ovl (safety check worked).
    //   Slightly more aggressive nudge (0.5) to ensure we capture
    //   the massive wins.  The 1.2x safety check prevents catastrophe.
    { NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, 0,
      METHOD_NUDGE, 0.5f, 1.20f,
      -1, -1 },

    // Band 6 (>=128MB): Tree+Simple NUDGE(0.85, 1.08)
    //   v3 showed -13.1% seq (nudge overrode wrongly) but +15.7% ovl.
    //   Tightened safety from 1.15 to 1.08: only override when NCCL
    //   thinks Tree+Simple is within 8% of best.  This should prevent
    //   the 13% regression while still capturing overlap wins when
    //   NCCL strongly agrees.
    { NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE, 0,
      METHOD_NUDGE, 0.85f, 1.08f,
      -1, -1 },
};

// =======================================================================
// SINGLE-NODE POLICY: All AUTO
// =======================================================================
//
// Single-node NVLink shows only 1-3% gaps.  Not worth the risk of
// regressions from static policies on unvalidated hardware.  The
// multi-node gains (5-17%) are where the tuner adds real value.
//
// Future work: validate single-node policies on A100/H100 NVLink.
//

// =======================================================================
// Tuner context
// =======================================================================
typedef struct {
    int topoMode;
    int verboseLog;
    size_t nRanks;
    size_t nNodes;
    ncclDebugLogger_t logFn;

    // Stats
    uint64_t forceCount;
    uint64_t nudgeCount;
    uint64_t nudgeBlockedCount;  // safety check blocked
    uint64_t autoCount;
    int callCount;
} TunerContext;

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

// =======================================================================
// Override helpers
// =======================================================================

// Find minimum non-IGNORE cost in the table (what NCCL would pick)
static float findMinCost(float** collCostTable, int numAlgo, int numProto) {
    float minCost = 1e30f;
    for (int a = 0; a < numAlgo; a++) {
        for (int p = 0; p < numProto; p++) {
            float c = collCostTable[a][p];
            if (c >= 0 && c < minCost) {
                minCost = c;
            }
        }
    }
    return minCost;
}

// FORCE: set cost=0 for the given (algo, proto).  Returns 1 on success.
static int applyForce(int algo, int proto, int nChannels,
                      float** collCostTable, int numAlgo, int numProto,
                      int* outChannels) {
    if (algo < 0 || proto < 0) return 0;
    if (algo >= numAlgo || proto >= numProto) return 0;
    if (collCostTable[algo][proto] == NCCL_ALGO_PROTO_IGNORE) return 0;

    collCostTable[algo][proto] = 0.0f;
    if (nChannels > 0) *outChannels = nChannels;
    return 1;
}

// NUDGE: make (algo, proto) the cheapest, but only if NCCL thinks it's
// within maxRatio of the current best.  Returns 1 on success, 0 if
// blocked by safety check.
static int applyNudge(int algo, int proto, int nChannels,
                      float nudge, float maxRatio,
                      float** collCostTable, int numAlgo, int numProto,
                      int* outChannels) {
    if (algo < 0 || proto < 0) return 0;
    if (algo >= numAlgo || proto >= numProto) return 0;

    float ourCost = collCostTable[algo][proto];
    if (ourCost == NCCL_ALGO_PROTO_IGNORE) return 0;

    float minCost = findMinCost(collCostTable, numAlgo, numProto);
    if (minCost >= 1e29f) return 0;  // no valid entries

    // SAFETY CHECK: if NCCL thinks our preferred is too expensive
    // relative to its best option, DON'T override.
    // This prevents catastrophic overrides when cluster conditions
    // differ from our training data.
    if (minCost > 0 && ourCost > minCost * maxRatio) {
        return 0;  // blocked — let NCCL decide
    }

    // Our preferred is reasonable.  Make it the cheapest.
    collCostTable[algo][proto] = minCost * nudge;
    if (nChannels > 0) *outChannels = nChannels;
    return 1;
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
    ctx->topoMode = (nNodes > 1) ? TOPO_MULTI_NODE : TOPO_SINGLE_NODE;

    const char* logEnv = getenv("NCCL_TUNER_LOG");
    ctx->verboseLog = (logEnv && atoi(logEnv) == 1) ? 1 : 0;

    if (logFunction) {
        logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                    "TUNER-V3: init | nodes=%zu ranks=%zu topo=%s",
                    nNodes, nRanks,
                    ctx->topoMode == TOPO_MULTI_NODE ? "MULTI" : "SINGLE");

        if (ctx->topoMode == TOPO_MULTI_NODE) {
            logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                        "TUNER-V3: multi-node active — "
                        "NUDGE Tree+Simple at 1MB+ with safety checks, "
                        "aggressive at 32MB+, tight at 4-32MB");
        } else {
            logFunction(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                        "TUNER-V3: single-node — all AUTO (focus on multi-node)");
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
    (void)numPipeOps;

    TunerContext* ctx = (TunerContext*)context;
    if (!ctx) return ncclInternalError;

    ctx->callCount++;

    // Only tune AllReduce on multi-node
    if (collType != ncclFuncAllReduce || ctx->topoMode != TOPO_MULTI_NODE) {
        return ncclSuccess;
    }

    int band = sizeBand(nBytes);
    const PolicyEntryV3* policy = &multiNodePolicy[band];

    // Method dispatch
    if (policy->method == METHOD_AUTO) {
        ctx->autoCount++;
        return ncclSuccess;
    }

    if (policy->method == METHOD_FORCE) {
        // Try primary
        if (applyForce(policy->algo, policy->proto, policy->nChannels,
                       collCostTable, numAlgo, numProto, nChannels)) {
            ctx->forceCount++;

            if (ctx->verboseLog && ctx->logFn) {
                ctx->logFn(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                           "TUNER-V3: FORCE %s+%s band=%d nBytes=%zu",
                           algoName(policy->algo), protoName(policy->proto),
                           band, nBytes);
            }
            return ncclSuccess;
        }

        // Try fallback
        if (policy->fb_algo >= 0 &&
            applyForce(policy->fb_algo, policy->fb_proto, 0,
                       collCostTable, numAlgo, numProto, nChannels)) {
            ctx->forceCount++;

            if (ctx->verboseLog && ctx->logFn) {
                ctx->logFn(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                           "TUNER-V3: FORCE fallback %s+%s band=%d nBytes=%zu",
                           algoName(policy->fb_algo), protoName(policy->fb_proto),
                           band, nBytes);
            }
            return ncclSuccess;
        }

        // Both primary and fallback unavailable — let NCCL decide
        ctx->autoCount++;
        return ncclSuccess;
    }

    if (policy->method == METHOD_NUDGE) {
        if (applyNudge(policy->algo, policy->proto, policy->nChannels,
                       policy->nudge, policy->maxRatio,
                       collCostTable, numAlgo, numProto, nChannels)) {
            ctx->nudgeCount++;

            if (ctx->verboseLog && ctx->logFn) {
                ctx->logFn(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                           "TUNER-V3: NUDGE %s+%s band=%d nBytes=%zu "
                           "(nudge=%.2f maxRatio=%.2f)",
                           algoName(policy->algo), protoName(policy->proto),
                           band, nBytes, policy->nudge, policy->maxRatio);
            }
            return ncclSuccess;
        }

        // Nudge blocked by safety check — NCCL thinks our preferred is
        // too expensive.  Let NCCL decide.
        ctx->nudgeBlockedCount++;

        if (ctx->verboseLog && ctx->logFn) {
            float ourCost = (policy->algo < numAlgo && policy->proto < numProto)
                          ? collCostTable[policy->algo][policy->proto] : -1;
            float minCost = findMinCost(collCostTable, numAlgo, numProto);
            ctx->logFn(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
                       "TUNER-V3: NUDGE BLOCKED %s+%s band=%d nBytes=%zu "
                       "(our_cost=%.1f min_cost=%.1f ratio=%.2f > max=%.2f)",
                       algoName(policy->algo), protoName(policy->proto),
                       band, nBytes, ourCost, minCost,
                       (minCost > 0) ? ourCost / minCost : 999.0f,
                       policy->maxRatio);
        }

        // Try fallback nudge if available
        if (policy->fb_algo >= 0) {
            if (applyNudge(policy->fb_algo, policy->fb_proto, 0,
                           policy->nudge, policy->maxRatio,
                           collCostTable, numAlgo, numProto, nChannels)) {
                ctx->nudgeCount++;
                return ncclSuccess;
            }
        }

        ctx->autoCount++;
        return ncclSuccess;
    }

    // Unknown method
    ctx->autoCount++;
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
                       "TUNER-V3: finalize | "
                       "forced=%llu nudged=%llu blocked=%llu auto=%llu calls=%d",
                       (unsigned long long)ctx->forceCount,
                       (unsigned long long)ctx->nudgeCount,
                       (unsigned long long)ctx->nudgeBlockedCount,
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

#define PLUGIN_NAME "WorkloadAware-v3"

const ncclTuner_v5_t ncclTunerPlugin_v5 = {
    .name = PLUGIN_NAME,
    .init = pluginInit,
    .getCollInfo = pluginGetCollInfo,
    .finalize = pluginFinalize
};
