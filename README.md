# burn-attnres - Attention Residuals for Burn

[![CI](https://github.com/sehaxe/burn-attnres/actions/workflows/ci.yml/badge.svg)](https://github.com/sehaxe/burn-attnres/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/burn-attnres)](https://crates.io/crates/burn-attnres)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![Burn](https://img.shields.io/badge/Burn-0.22-orange.svg)](https://burn.dev)

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


## Performance (RTX 3090, CUDA, burn 0.22)

The Full AttnRes hot path (`depth_attend`) is 8 tensor passes over
`[L, B, T, D]` on the naive path. The fused CUDA path processes the history in
chunks of G=8 layers (online softmax, running state aliased to the output), so
peak memory is `(G+1)·B·T·D` — ~2× less than the `(L+1)·B·T·D` stack — with
exact full-depth weights.

| Op | Config | Tensor path | Fused | Speedup |
|----|--------|-------------|-------|---------|
| forward | L=24, b=1, t=2048, d=4096 | 80.3 ms | **7.6 ms** | **10.6×** |
| forward | L=8, b=2, t=2048, d=5120 | 16.1 ms | 19.1 ms | 0.8× (per-layer launch overhead on small L) |
| backward | L=24, b=1, t=2048, d=4096 | 129.4 ms | **46.7 ms** | **2.8×** |

## Training

The fused `depth_attend` runs as a single tracked node under `Autodiff<Cuda>`
with a fused backward (per (b,t) cube: RMS scores over L via tree reductions,
softmax, then d_h_l and d_q in one launch; the history is stacked with a fast
copy kernel). Verified fused backward == tensor-path backward (dh/dq < 1e-2).
Parent count is capped at 64 history layers + query (const-generic op).

## Inference

Forward-only builds use the bare CUDA fused chunked `depth_attend` directly
(no graph overhead). The streaming `BlockAttnRes` path collapses ~15 launches
per layer output to ~4 (`source_score` norm+dot and the online-softmax merge,
each one launch).

## License

AGPL-3.0. See [LICENSE](LICENSE).

