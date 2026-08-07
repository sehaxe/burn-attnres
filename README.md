# burn-attnres - Attention Residuals for Burn

[![CI](https://github.com/sehaxe/burn-attnres/actions/workflows/ci.yml/badge.svg)](https://github.com/sehaxe/burn-attnres/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/burn-attnres)](https://crates.io/crates/burn-attnres)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.21-orange.svg)](https://burn.dev)

Learned depth-wise attention over layer outputs. Drop-in replacement for
fixed residual accumulation. Mitigates PreNorm dilution: output magnitudes
stay bounded, gradients distribute uniformly.

> Paper: [Attention Residuals](https://arxiv.org/abs/2603.15031) (Moonshot/Kimi, 2026).
> Key results: GPQA +7.5, HumanEval +3.1, MMLU +1.1 on Kimi Linear 48B.

## Install

```bash
cargo add burn-attnres
```

## Quick start

```rust
use burn_attnres::{AttnRes, BlockAttnRes, depth_attend};

// Full: query attends over all previous hidden states
let attn = AttnRes::new(512, &device);
let out = attn.forward(&history); // [h0, h1, ..., hN] -> [B,T,D]

// Block: groups into blocks for efficiency at scale
let attn = BlockAttnRes::new(512, 8, &device);
let out = attn.forward(&history);

// Raw depth attention (no parameters)
let out = depth_attend(&history, query);
```

## API

| Export | What |
|--------|------|
| `AttnRes` | Full: attend over ALL previous states |
| `BlockAttnRes` | Block: attend over block summaries |
| `depth_attend` | Core: softmax over depth via learned query |

## License

AGPL-3.0. See [LICENSE](LICENSE).

## Performance (RTX 5060 Ti, CUDA, burn 0.22)

The Full AttnRes hot path (`depth_attend`) is 8 tensor passes over
`[L, B, T, D]` on the naive path (square, sum, sqrt, div, mul, sum, softmax,
weighted-sum). The fused CUDA path runs it in three kernel groups:

1. per-layer score launches (RMS norm + `q·h` in one read pass, shared-memory
   tree reductions) that also fill the `[L, B, T, D]` stack as a by-product —
   burn's `Tensor::cat` measured ~23 ms for that copy;
2. a tiny softmax over the depth axis (one thread per `(b, t)`);
3. one weighted-sum launch that reads the stack once and writes `out` once.

| Config | Tensor path | Fused | Speedup |
|--------|-------------|-------|---------|
| L=24, b=1, t=2048, d=4096 | 127 ms | **24 ms** | **5.3×** |
| L=8, b=2, t=2048, d=5120 | 60 ms | **20 ms** | **3.0×** |

The streaming `BlockAttnRes` step collapses ~15 launches per layer output to
~4: `source_score` (norm+dot) in one launch and the online-softmax merge in
one launch (state update + attended output). Both verified == tensor path
(<1e-4 max abs).
