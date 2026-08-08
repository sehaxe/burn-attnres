//! Fused CUDA kernels for Attention Residuals (Kimi K3 §2.2).
//!
//! `depth_attend` is the full AttnRes hot path: RMS-norm per (layer, b, t),
//! score = q·h_l·scale, online softmax over the depth axis, weighted sum —
//! ~8 tensor passes over `[L, B, T, D]` on the tensor path, one launch here
//! (one cube per (b, t), threads split D into slabs, shared-memory tree
//! reductions for the per-layer dot).
//!
//! `source_score` and the online-softmax `merge` collapse the streaming
//! BlockAttnRes step's ~15 launches per layer output to 2.

use burn::tensor::Tensor;
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;
use std::any::Any;
use std::cell::RefCell;

const THREADS: u32 = 256;

/// Layers per chunk in the chunked `depth_attend` (paper's N ≈ 8). Peak
/// memory scales as `(G+2)·B·T·D`; G=8 balances the running-state traffic
/// against the chunk stack size.
pub const CHUNK_G: usize = 8;

// ponytail: single-slot cache for the internal [G,B,T,D] chunk stack. A fresh
// CUDA allocation costs ~15ms of first-touch page faults (default allocator);
// the chunk never escapes depth_attend_cuda so reusing one buffer is safe.
// One extra allocation per distinct chunk shape otherwise.
thread_local! {
    static CHUNK_CACHE: RefCell<Option<(usize, usize, usize, usize, CubeTensor<cubecl::cuda::CudaRuntime>)>> =
        const { RefCell::new(None) };
}

fn cached_chunk(
    g: usize,
    b: usize,
    t: usize,
    d: usize,
    device: &burn::tensor::Device,
) -> CubeTensor<cubecl::cuda::CudaRuntime> {
    CHUNK_CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if let Some((cl, cb, ct, cd, buf)) = &*c {
            if (*cl, *cb, *ct, *cd) == (g, b, t, d) {
                return buf.clone();
            }
        }
        let s = Tensor::<4>::empty([g, b, t, d], device);
        let buf = cube_of(&s).expect("bare CUDA chunk");
        *c = Some((g, b, t, d, buf.clone()));
        buf
    })
}

fn cube_of<const D: usize>(t: &Tensor<D>) -> Option<CubeTensor<cubecl::cuda::CudaRuntime>> {
    type B = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>;
    let prim = t.clone().try_into_primitive::<B>().ok()?;
    let c = (&prim as &dyn Any).downcast_ref::<CubeTensor<cubecl::cuda::CudaRuntime>>()?;
    Some(c.clone())
}

fn cube_of_1(t: &Tensor<1>) -> Option<CubeTensor<cubecl::cuda::CudaRuntime>> {
    cube_of(t)
}

/// Chunked Full AttnRes (Kimi K3 §2.2, exact math, bounded memory).
///
/// The history is processed in chunks of `G` layers: per-layer score launches
/// (RMS norm + `q·h` in one read pass, tree reductions) fill the `[G,B,T,D]`
/// chunk stack as a by-product, and one chunk kernel per chunk folds the
/// chunk into a running online-softmax state `(acc, max, sum)` — `acc` keeps
/// every seen layer's contribution in one `[B,T,D]`, so peak memory is
/// `(G+2)·B·T·D` instead of `(L+1)·B·T·D`. The final division recovers the
/// exact full-depth softmax weights (no Block approximation).
#[cube(launch_unchecked)]
fn attnres_scores_kernel<F: Float>(
    h: &[F],          // [B, T, D] one layer
    q: &[F],          // [D]
    scores: &mut [F], // [G, B, T] row `row`
    chunk: &mut [F],  // [G, B, T, D] row `row`
    row: u32,
    scale: f32,
    #[comptime] bt_count: u32, // B * T
    #[comptime] d: u32,
    #[comptime] threads: u32,
    #[comptime] per: u32,
    #[comptime] log_threads: u32,
) {
    let bt = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let bt_count = bt_count as usize;
    let d = d as usize;
    let threads = threads as usize;
    let per = per as usize;
    let lg = log_threads as usize;
    let base = bt * d;

    let mut red = Shared::<[F]>::new_slice(threads);

    let mut psq = F::new(0.0_f32);
    let mut pdot = F::new(0.0_f32);
    for j in 0..per {
        let col = j * threads + tid;
        if col < d {
            let v = h[base + col];
            psq += v * v;
            pdot += q[col] * v;
            chunk[(row as usize) * bt_count * d + base + col] = v;
        }
    }
    red[tid] = psq;
    sync_cube();
    for k in 0..lg {
        let stride = threads >> (k + 1);
        if tid < stride {
            red[tid] = red[tid] + red[tid + stride];
        }
        sync_cube();
    }
    let sq = red[0];
    sync_cube();
    red[tid] = pdot;
    sync_cube();
    for k in 0..lg {
        let stride = threads >> (k + 1);
        if tid < stride {
            red[tid] = red[tid] + red[tid + stride];
        }
        sync_cube();
    }
    let dot = red[0];

    if tid == 0 {
        scores[(row as usize) * bt_count + bt] =
            dot * F::cast_from(scale) / (sq + F::new(1e-5_f32)).sqrt();
    }
}

/// Fold one chunk into the running online-softmax state.
///
/// Per (b,t): chunk max m_c and exp-sum, running merge
/// m' = max(m, m_c); acc' = acc·e^(m−m') + Σ_g e^(s_g−m')·h_g;
/// sum' = sum·e^(m−m') + sumexp_c·e^(m_c−m'); the last chunk writes
/// out = acc'/clamp(sum', 1e-12). All state reads happen before any write, so
/// no cross-thread race on `max_s`/`sum_e`.
#[cube(launch_unchecked)]
fn attnres_chunk_kernel<F: Float>(
    scores: &[F],    // [G, B, T] current chunk (rows >= ga stale)
    chunk: &[F],     // [G, B, T, D] current chunk
    acc: &mut [F],   // [B, T, D] running weighted sum
    max_s: &mut [F], // [B, T] running max
    sum_e: &mut [F], // [B, T] running exp sum
    out: &mut [F],   // [B, T, D] written on the last chunk
    #[comptime] g: u32,
    #[comptime] ga: u32, // actual layer count of this chunk (tail < G)
    #[comptime] bt_count: u32,
    #[comptime] d: u32,
    #[comptime] threads: u32,
    #[comptime] per: u32,
    #[comptime] first: bool,
    #[comptime] last: bool,
) {
    let bt = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let g = g as usize;
    let bt_count = bt_count as usize;
    let d = d as usize;
    let threads = threads as usize;
    let per = per as usize;
    let base = bt * d;

    // comptime loops with a runtime `ga` guard: the tail chunk reuses the
    // scores/chunk buffers and stale rows above ga must not contribute
    let mut m_c = F::new(-3.0e38_f32);
    for li in 0..g {
        if (li as u32) < ga {
            let s = scores[li * bt_count + bt];
            if s > m_c {
                m_c = s;
            }
        }
    }
    let mut sumexp_c = F::new(0.0_f32);
    for li in 0..g {
        if (li as u32) < ga {
            sumexp_c += (scores[li * bt_count + bt] - m_c).exp();
        }
    }

    let m_old = max_s[bt];
    let sum_old = sum_e[bt];
    let mut m_new = m_old;
    if m_c > m_old {
        m_new = m_c;
    }
    let rescale = (m_old - m_new).exp();
    let sum_new = sum_old * rescale + sumexp_c * (m_c - m_new).exp();

    let mut ws = Shared::<[F]>::new_slice(g);
    for li in 0..g {
        if (li as u32) < ga {
            ws[li] = (scores[li * bt_count + bt] - m_new).exp();
        } else {
            ws[li] = F::new(0.0_f32);
        }
    }

    for j in 0..per {
        let col = j * threads + tid;
        if col < d {
            let mut a = F::new(0.0_f32);
            for li in 0..g {
                if (li as u32) < ga {
                    a += ws[li] * chunk[li * bt_count * d + base + col];
                }
            }
            let acc_new = if first {
                a
            } else {
                acc[base + col] * rescale + a
            };
            acc[base + col] = acc_new;
            if last {
                let mut den = sum_new;
                if den < F::new(1e-12_f32) {
                    den = F::new(1e-12_f32);
                }
                out[base + col] = acc_new / den;
            }
        }
    }
    if tid == 0 {
        max_s[bt] = m_new;
        sum_e[bt] = sum_new;
    }
}

// ---- dispatch (bare CUDA backend only, callers fall back to tensor path) ----

/// `source_score` (streaming path): RMS-norm + q·src·scale -> [B, T, 1].
#[cube(launch_unchecked)]
fn source_score_kernel<F: Float>(
    h: &[F],       // [B, T, D]
    q: &[F],       // [D]
    out: &mut [F], // [B, T]
    scale: f32,
    #[comptime] d: u32,
    #[comptime] threads: u32,
    #[comptime] per: u32,
    #[comptime] log_threads: u32,
) {
    let bt = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let d = d as usize;
    let threads = threads as usize;
    let per = per as usize;
    let lg = log_threads as usize;
    let base = bt * d;

    let mut red = Shared::<[F]>::new_slice(threads);

    let mut psq = F::new(0.0_f32);
    let mut pdot = F::new(0.0_f32);
    for j in 0..per {
        let col = j * threads + tid;
        if col < d {
            let v = h[base + col];
            psq += v * v;
            pdot += q[col] * v;
        }
    }
    red[tid] = psq;
    sync_cube();
    for k in 0..lg {
        let stride = threads >> (k + 1);
        if tid < stride {
            red[tid] = red[tid] + red[tid + stride];
        }
        sync_cube();
    }
    let sq = red[0];
    sync_cube();
    red[tid] = pdot;
    sync_cube();
    for k in 0..lg {
        let stride = threads >> (k + 1);
        if tid < stride {
            red[tid] = red[tid] + red[tid + stride];
        }
        sync_cube();
    }
    let dot = red[0];

    if tid == 0 {
        out[bt] = dot * F::cast_from(scale) / (sq + F::new(1e-5_f32)).sqrt();
    }
}

/// Online-softmax merge (streaming path): fold `src` with score `s` into the
/// running state and emit the attended output in one launch.
///
/// m' = max(m, s); acc' = acc·e^(m−m') + src·e^(s−m');
/// sum' = sum·e^(m−m') + e^(s−m'); out = acc'/clamp(sum', 1e-12).
#[cube(launch_unchecked)]
fn merge_kernel<F: Float>(
    acc: &mut [F],     // [B, T, D] in/out
    max_s: &mut [F],   // [B, T] in/out
    sum_exp: &mut [F], // [B, T] in/out
    src: &[F],         // [B, T, D]
    s: &[F],           // [B, T]
    out: &mut [F],     // [B, T, D]
    #[comptime] d: u32,
    #[comptime] threads: u32,
    #[comptime] per: u32,
) {
    let bt = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let d = d as usize;
    let threads = threads as usize;
    let per = per as usize;
    let base = bt * d;

    let m_old = max_s[bt];
    let sval = s[bt];
    let mut m_new = m_old;
    if sval > m_old {
        m_new = sval;
    }
    let rescale = (m_old - m_new).exp();
    let w = (sval - m_new).exp();
    let sum_new = sum_exp[bt] * rescale + w;

    for j in 0..per {
        let col = j * threads + tid;
        if col < d {
            let i = base + col;
            let a = acc[i] * rescale + src[i] * w;
            acc[i] = a;
            let mut den = sum_new;
            if den < F::new(1e-12_f32) {
                den = F::new(1e-12_f32);
            }
            out[i] = a / den;
        }
    }
    if tid == 0 {
        max_s[bt] = m_new;
        sum_exp[bt] = sum_new;
    }
}

pub fn depth_attend_cuda(history: &[Tensor<3>], query: &Tensor<1>) -> Option<Tensor<3>> {
    let l = history.len();
    if l == 0 {
        return None;
    }
    let [b, t, d] = history[0].dims();
    if d == 0 {
        return None;
    }
    let qc = cube_of(query)?;
    let hc: Vec<CubeTensor<cubecl::cuda::CudaRuntime>> =
        history.iter().map(cube_of).collect::<Option<_>>()?;
    // chunked: peak memory (G+2)*B*T*D instead of (L+1)*B*T*D; the chunk
    // stack is cached (never escapes), out is written once by the last chunk
    let dev = &history[0].device();
    let g = CHUNK_G.min(l);
    let stc = cached_chunk(g, b, t, d, dev);
    let max_s = Tensor::<2>::full([b, t], -3.0e38_f32, dev);
    let mxc = cube_of(&max_s)?;
    let sum_e = Tensor::<2>::zeros([b, t], dev);
    let sec = cube_of(&sum_e)?;
    let out = Tensor::<3>::empty([b, t, d], dev);
    let oc = cube_of(&out)?;
    let scores = Tensor::<2>::zeros([g, b * t], dev);
    let sc2 = cube_of(&scores)?;
    let client = hc[0].client.clone();
    let per = (d as u32).div_ceil(THREADS);
    let dim = CubeDim {
        x: THREADS,
        y: 1,
        z: 1,
    };
    let scale = (d as f64).powf(-0.5) as f32;
    let bt = (b * t) as u32;
    let chunks = l.div_ceil(g);
    unsafe {
        for c in 0..chunks {
            let first = c == 0;
            let last = c == chunks - 1;
            for gi in 0..g {
                let li = c * g + gi;
                if li >= l {
                    break;
                }
                let h = &hc[li];
                attnres_scores_kernel::launch_unchecked::<f32, cubecl::cuda::CudaRuntime>(
                    &client,
                    CubeCount::Static(bt, 1, 1),
                    dim,
                    BufferArg::from_raw_parts(h.handle.clone(), b * t * d),
                    BufferArg::from_raw_parts(qc.handle.clone(), d),
                    BufferArg::from_raw_parts(sc2.handle.clone(), g * b * t),
                    BufferArg::from_raw_parts(stc.handle.clone(), g * b * t * d),
                    gi as u32,
                    scale,
                    bt,
                    d as u32,
                    THREADS,
                    per,
                    THREADS.ilog2(),
                );
            }
            let ga = g.min(l - c * g);
            attnres_chunk_kernel::launch_unchecked::<f32, cubecl::cuda::CudaRuntime>(
                &client,
                CubeCount::Static(bt, 1, 1),
                dim,
                BufferArg::from_raw_parts(sc2.handle.clone(), g * b * t),
                BufferArg::from_raw_parts(stc.handle.clone(), g * b * t * d),
                BufferArg::from_raw_parts(oc.handle.clone(), b * t * d),
                BufferArg::from_raw_parts(mxc.handle.clone(), b * t),
                BufferArg::from_raw_parts(sec.handle.clone(), b * t),
                BufferArg::from_raw_parts(oc.handle.clone(), b * t * d),
                g as u32,
                ga as u32,
                bt,
                d as u32,
                THREADS,
                per,
                first,
                last,
            );
        }
    }
    Some(out)
}

pub fn source_score_cuda(query: &Tensor<1>, src: &Tensor<3>) -> Option<Tensor<3>> {
    let [b, t, d] = src.dims();
    if d == 0 {
        return None;
    }
    let qc = cube_of_1(query)?;
    let sc = cube_of(src)?;
    let out = Tensor::<3>::zeros([b, t, 1], &src.device());
    let oc = cube_of(&out)?;
    let client = sc.client.clone();
    let per = (d as u32).div_ceil(THREADS);
    let dim = CubeDim {
        x: THREADS,
        y: 1,
        z: 1,
    };
    let scale = (d as f64).powf(-0.5) as f32;
    unsafe {
        source_score_kernel::launch_unchecked::<f32, cubecl::cuda::CudaRuntime>(
            &client,
            CubeCount::Static((b * t) as u32, 1, 1),
            dim,
            BufferArg::from_raw_parts(sc.handle, b * t * d),
            BufferArg::from_raw_parts(qc.handle, d),
            BufferArg::from_raw_parts(oc.handle, b * t),
            scale,
            d as u32,
            THREADS,
            per,
            THREADS.ilog2(),
        );
    }
    Some(out)
}

/// Folds `src` into the online state and returns the attended output.
pub fn merge_cuda(
    acc: &mut Tensor<3>,
    max_s: &mut Tensor<3>,
    sum_exp: &mut Tensor<3>,
    src: &Tensor<3>,
    s: &Tensor<3>,
) -> Option<Tensor<3>> {
    let [b, t, d] = src.dims();
    if d == 0 {
        return None;
    }
    let accc = cube_of(acc)?;
    let mx = cube_of(max_s)?;
    let se = cube_of(sum_exp)?;
    let sc = cube_of(src)?;
    let s1 = cube_of(s)?;
    let out = Tensor::<3>::zeros([b, t, d], &src.device());
    let oc = cube_of(&out)?;
    let client = accc.client.clone();
    let per = (d as u32).div_ceil(THREADS);
    let dim = CubeDim {
        x: THREADS,
        y: 1,
        z: 1,
    };
    unsafe {
        merge_kernel::launch_unchecked::<f32, cubecl::cuda::CudaRuntime>(
            &client,
            CubeCount::Static((b * t) as u32, 1, 1),
            dim,
            BufferArg::from_raw_parts(accc.handle, b * t * d),
            BufferArg::from_raw_parts(mx.handle, b * t),
            BufferArg::from_raw_parts(se.handle, b * t),
            BufferArg::from_raw_parts(sc.handle, b * t * d),
            BufferArg::from_raw_parts(s1.handle, b * t),
            BufferArg::from_raw_parts(oc.handle, b * t * d),
            d as u32,
            THREADS,
            per,
        );
    }
    Some(out)
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::{depth_attend, BlockAttnRes};
    use burn::module::{Param, ParamId};
    use burn::tensor::{activation, Device, Distribution, Tensor};

    fn to_host<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    fn stack(history: &[Tensor<3>]) -> Tensor<4> {
        let stacked: Vec<Tensor<4>> = history
            .iter()
            .map(|h| h.clone().unsqueeze_dim::<4>(0))
            .collect();
        Tensor::cat(stacked, 0)
    }

    fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    /// Raw-op depth_attend on the same device as the fused call.
    fn ref_depth_attend(history: &[Tensor<3>], query: &Tensor<1>) -> Tensor<3> {
        let n = history.len();
        let [b, t, d] = history[0].dims();
        let scale = (d as f64).powf(-0.5);
        let h_stack = stack(history);
        let h_norm_sq = h_stack.clone().powf_scalar(2.0).sum_dim(3).add_scalar(1e-5);
        let h_norm = h_stack.clone() / h_norm_sq.sqrt().reshape([n, b, t, 1usize]);
        let q = query.clone().reshape([1, 1, 1, d]);
        let scores = (q * h_norm).sum_dim(3).mul_scalar(scale);
        let weights = activation::softmax(scores, 0);
        h_stack
            .mul(weights.reshape([n, b, t, 1usize]))
            .sum_dim(0)
            .reshape([b, t, d])
    }

    #[test]
    fn depth_attend_fused_matches_tensor() {
        let cdev = Device::default();
        for (l, b, t, d) in [
            (12usize, 2usize, 8usize, 512usize),
            (3, 1, 4, 32),
            (40, 2, 4, 2048),
        ] {
            let hist: Vec<Tensor<3>> = (0..l)
                .map(|_| Tensor::<3>::random([b, t, d], Distribution::Normal(0.0, 1.0), &cdev))
                .collect();
            let q = Tensor::<1>::random([d], Distribution::Normal(0.0, 1.0), &cdev);
            let expected = to_host(ref_depth_attend(&hist, &q));
            assert!(
                depth_attend_cuda(&hist, &q).is_some(),
                "fused dispatch should engage on bare CUDA"
            );
            let got = to_host(depth_attend(&hist, q));
            let md = maxdiff(&got, &expected);
            assert!(md < 1e-4, "[{l},{b},{t},{d}] maxdiff {md}");
        }
    }

    #[test]
    fn source_score_fused_matches_tensor() {
        let cdev = Device::default();
        let (b, t, d) = (2usize, 8usize, 256usize);
        let src = Tensor::<3>::random([b, t, d], Distribution::Normal(0.0, 1.0), &cdev);
        let q = Tensor::<1>::random([d], Distribution::Normal(0.0, 1.0), &cdev);
        let scale = (d as f64).powf(-0.5);
        let norm = src.clone()
            / src
                .clone()
                .powf_scalar(2.0)
                .sum_dim(2)
                .add_scalar(1e-5)
                .sqrt();
        let expected = to_host(
            (q.clone().reshape([1, 1, d]) * norm)
                .sum_dim(2)
                .mul_scalar(scale),
        );
        let got = source_score_cuda(&q, &src).expect("kernel");
        let md = maxdiff(&to_host(got), &expected);
        assert!(md < 1e-4, "source_score maxdiff {md}");
    }

    #[test]
    fn merge_fused_matches_tensor() {
        let cdev = Device::default();
        let (b, t, d) = (2usize, 8usize, 128usize);
        let acc = Tensor::<3>::random([b, t, d], Distribution::Normal(0.0, 1.0), &cdev);
        let max_s = Tensor::<3>::random([b, t, 1], Distribution::Normal(0.0, 1.0), &cdev);
        let sum_e = Tensor::<3>::random([b, t, 1], Distribution::Uniform(0.5, 1.5), &cdev);
        let src = Tensor::<3>::random([b, t, d], Distribution::Normal(0.0, 1.0), &cdev);
        let s = Tensor::<3>::random([b, t, 1], Distribution::Normal(0.0, 1.0), &cdev);
        // raw-op reference: m' = max(m,s); rescale; out = acc'/clamp(sum',1e-12)
        let m_new = (max_s.clone() + s.clone() + (max_s.clone() - s.clone()).abs()).div_scalar(2.0);
        let rescale = (max_s.clone() - m_new.clone()).exp();
        let w = (s.clone() - m_new.clone()).exp();
        let acc_r = acc.clone() * rescale.clone() + src.clone() * w.clone();
        let sum_r = sum_e.clone() * rescale + w;
        let expected = to_host(acc_r.clone() / sum_r.clamp_min(1e-12));

        let mut acc_c = acc.clone();
        let mut mx = max_s.clone();
        let mut se = sum_e.clone();
        let got = merge_cuda(&mut acc_c, &mut mx, &mut se, &src, &s).expect("kernel");
        let md = maxdiff(&to_host(got), &expected);
        assert!(md < 1e-4, "merge out maxdiff {md}");
        let md = maxdiff(&to_host(acc_c), &to_host(acc_r));
        assert!(md < 1e-4, "acc state mismatch {md}");
    }

    #[test]
    fn streaming_fused_matches_tensor_path() {
        // CPU (burn-cpu, tensor path) vs CUDA (fused kernels): the same module
        // semantics must agree at every step.
        let cdev = Device::default();
        let cpu_dev = Device::cpu();
        let (b, t, d, bs) = (1usize, 3usize, 64usize, 2usize);
        let q_cpu = Tensor::<1>::random([d], Distribution::Normal(0.0, 1.0), &cpu_dev);
        let mut cpu = BlockAttnRes::new(d, bs, &cpu_dev);
        cpu.query = Param::initialized(ParamId::new(), q_cpu.clone());
        let mut cuda = BlockAttnRes::new(d, bs, &cdev);
        cuda.query = Param::initialized(
            ParamId::new(),
            Tensor::<1>::from_data(q_cpu.clone().into_data(), &cdev),
        );

        let mut st_cpu = cpu.init_state(b, t, d, &cpu_dev);
        let mut st_cuda = cuda.init_state(b, t, d, &cdev);
        for step in 0..8 {
            let h = Tensor::<3>::random([b, t, d], Distribution::Normal(0.0, 1.0), &cpu_dev);
            let out_cpu = cpu.step(h.clone(), &mut st_cpu);
            let hc = Tensor::<3>::from_data(h.into_data(), &cdev);
            let out_cuda = cuda.step(hc, &mut st_cuda);
            let md = maxdiff(&to_host(out_cuda), &to_host(out_cpu));
            assert!(md < 1e-4, "step {step}: maxdiff {md}");
        }
    }

    #[test]
    #[ignore]
    fn attnres_bench() {
        let cdev = Device::default();
        for (l, b, t, d) in [(24usize, 1usize, 2048usize, 4096usize), (8, 2, 2048, 5120)] {
            let hist: Vec<Tensor<3>> = (0..l)
                .map(|_| Tensor::<3>::random([b, t, d], Distribution::Normal(0.0, 1.0), &cdev))
                .collect();
            let q = Tensor::<1>::random([d], Distribution::Normal(0.0, 1.0), &cdev);
            for _ in 0..3 {
                let _ = depth_attend(&hist, q.clone());
            }
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                let r = depth_attend(&hist, q.clone());
                let _: f32 = r.clone().sum().into_scalar(); // flush
            }
            let tf = t0.elapsed() / 10;
            // probe: 24 trivial launches (one per layer) to isolate launch overhead
            let hc3: Vec<_> = hist.iter().map(cube_of).collect::<Option<_>>().unwrap();
            let client3 = hc3[0].client.clone();
            let qc = cube_of(&q).unwrap();
            let nidle = Tensor::<1>::zeros([l * b * t], &cdev);
            let nc = cube_of(&nidle).unwrap();
            let dim3 = CubeDim {
                x: THREADS,
                y: 1,
                z: 1,
            };
            let per = (d as u32).div_ceil(THREADS);
            for _ in 0..2 {
                for h in &hc3 {
                    unsafe {
                        attnres_scores_kernel::launch_unchecked::<f32, cubecl::cuda::CudaRuntime>(
                            &client3,
                            CubeCount::Static((b * t) as u32, 1, 1),
                            dim3,
                            BufferArg::from_raw_parts(h.handle.clone(), b * t * d),
                            BufferArg::from_raw_parts(qc.handle.clone(), d),
                            BufferArg::from_raw_parts(nc.handle.clone(), l * b * t),
                            BufferArg::from_raw_parts(nc.handle.clone(), l * b * t * d),
                            0u32,
                            (d as f64).powf(-0.5) as f32,
                            (b * t) as u32,
                            d as u32,
                            THREADS,
                            per,
                            THREADS.ilog2(),
                        );
                    }
                }
                let _: f32 = nidle.clone().sum().into_scalar();
            }
            let t0 = std::time::Instant::now();
            for h in &hc3 {
                unsafe {
                    attnres_scores_kernel::launch_unchecked::<f32, cubecl::cuda::CudaRuntime>(
                        &client3,
                        CubeCount::Static((b * t) as u32, 1, 1),
                        dim3,
                        BufferArg::from_raw_parts(h.handle.clone(), b * t * d),
                        BufferArg::from_raw_parts(qc.handle.clone(), d),
                        BufferArg::from_raw_parts(nc.handle.clone(), l * b * t),
                        BufferArg::from_raw_parts(nc.handle.clone(), l * b * t * d),
                        0u32,
                        (d as f64).powf(-0.5) as f32,
                        (b * t) as u32,
                        d as u32,
                        THREADS,
                        per,
                        THREADS.ilog2(),
                    );
                }
            }
            let _: f32 = nidle.clone().sum().into_scalar();
            println!("    {} scores launches+work {:?}", l, t0.elapsed());
            let hs = stack(&hist);
            let t0 = std::time::Instant::now();
            for _ in 0..3 {
                let hn = hs.clone().powf_scalar(2.0).sum_dim(3).add_scalar(1e-5);
                let hnm = hs.clone() / hn.sqrt().reshape([l, b, t, 1usize]);
                let sc = (q.clone().reshape([1, 1, 1, d]) * hnm)
                    .sum_dim(3)
                    .mul_scalar((d as f64).powf(-0.5));
                let w = activation::softmax(sc, 0);
                let r = hs
                    .clone()
                    .mul(w.reshape([l, b, t, 1usize]))
                    .sum_dim(0)
                    .reshape([b, t, d]);
                let _: f32 = r.clone().sum().into_scalar(); // flush async queue
            }
            let tt = t0.elapsed() / 3;
            println!(
                "[L{l} b{b} t{t} d{d}] fused {:?} tensor {:?} ({:.1}x)",
                tf,
                tt,
                tt.as_secs_f64() / tf.as_secs_f64()
            );
        }
    }
}

#[cfg(feature = "autodiff")]
mod ad {
    use burn::backend::{Backend, DispatchKindConversion};
    use burn::tensor::{DispatchTensor, Tensor};
    use burn_autodiff::checkpoint::base::Checkpointer;
    use burn_autodiff::checkpoint::strategy::NoCheckpointing;
    use burn_autodiff::grads::Gradients;
    use burn_autodiff::ops::{Backward, Ops, OpsKind};
    use burn_autodiff::Autodiff;

    #[derive(Debug)]
    struct AttnResOp;

    impl<B: Backend, const N: usize> Backward<B, N> for AttnResOp
    where
        DispatchTensor: DispatchKindConversion<B>,
    {
        type State = usize; // L (number of history layers)

        fn backward(
            self,
            ops: Ops<Self::State, N>,
            grads: &mut Gradients,
            checkpointer: &mut Checkpointer,
        ) {
            let l = ops.state;
            let node = |i: usize| ops.parents[i].as_ref().expect("attnres input checkpointed");
            let q = Tensor::<1>::from_primitive::<B>(checkpointer.retrieve_node_output(node(l).id));
            let mut hs: Vec<Tensor<3>> = Vec::with_capacity(l);
            for i in 0..l {
                hs.push(Tensor::from_primitive::<B>(
                    checkpointer.retrieve_node_output(node(i).id),
                ));
            }
            let d_out = Tensor::from_primitive::<B>(grads.consume::<B>(&ops.node));
            let (dhs, dq) = depth_attend_backward_tensor(&hs, &q, &d_out);
            for (i, dh) in dhs.into_iter().enumerate() {
                grads.register::<B>(
                    ops.parents[i].clone().unwrap().id,
                    dh.try_into_primitive::<B>().unwrap(),
                );
            }
            grads.register::<B>(
                ops.parents[l].clone().unwrap().id,
                dq.try_into_primitive::<B>().unwrap(),
            );
        }
    }

    /// Exact full-attention backward over the depth axis (tensor path).
    /// scores_l = q·h_l·scale/√(Σh²+ε); w = softmax(scores, 0);
    /// out = Σ w_l·h_l. Returns (d_h per layer, d_q).
    pub fn depth_attend_backward_tensor(
        history: &[Tensor<3>],
        query: &Tensor<1>,
        d_out: &Tensor<3>,
    ) -> (Vec<Tensor<3>>, Tensor<1>) {
        let l = history.len();
        let [b, t, d] = history[0].dims();
        let scale = (d as f64).powf(-0.5);
        let h_stack = crate::fused_attnres::stack_ad(history);
        let q = query.clone().reshape([1, 1, 1, d]);
        let s2 = h_stack.clone().powf_scalar(2.0).sum_dim(3).add_scalar(1e-5); // [L,B,T,1]
        let inv = s2.clone().powf_scalar(-0.5);
        let scores = (q.clone() * h_stack.clone())
            .sum_dim(3)
            .mul(inv.clone()) // [L,B,T,1]
            .mul_scalar(scale);
        let w = burn::tensor::activation::softmax(scores.squeeze_dim::<3>(3), 0); // [L,B,T]

        // d_w_l = Σ_d d_out·h_l; d_scores = w·(d_w − Σ_l w·d_w)
        let d_out4 = d_out.clone().unsqueeze_dim::<4>(0); // [1,B,T,D]
        let d_w = (d_out4.clone() * h_stack.clone())
            .sum_dim(3)
            .squeeze_dim::<3>(3); // [L,B,T]
        let d_scores = w.clone() * (d_w.clone() - (w.clone() * d_w).sum_dim(0));

        // d_h = w_l·d_out + scale·d_scores·(q/√s − h_l·(q·h_l)/s^(3/2))
        let d_scores4 = d_scores.unsqueeze_dim::<4>(3); // [L,B,T,1]
        let qh = (q.clone() * h_stack.clone()).sum_dim(3); // [L,B,T,1]
        let dh_attn = w.unsqueeze_dim::<4>(3) * d_out4.clone(); // [L,B,T,D]
        let dh_norm = d_scores4.clone()
            * (q.clone() * inv.clone() - h_stack.clone() * qh * s2.clone().powf_scalar(-1.5))
            * scale;
        let dh_stack = dh_attn + dh_norm;

        // d_q = Σ_l d_scores·scale·h_l/√s
        let dq = (d_scores4 * h_stack * inv * scale)
            .sum_dim(0)
            .sum_dim(1)
            .sum_dim(2)
            .reshape([d]);

        let dhs: Vec<Tensor<3>> = (0..l)
            .map(|i| {
                dh_stack
                    .clone()
                    .slice([i..i + 1, 0..b, 0..t, 0..d])
                    .reshape([b, t, d])
            })
            .collect();
        (dhs, dq)
    }

    /// Fused chunked depth_attend with exact backward on `Autodiff<Inner>`.
    pub fn depth_attend_autodiff<Inner: Backend, const N: usize>(
        history: &[Tensor<3>],
        query: Tensor<1>,
    ) -> Option<Tensor<3>>
    where
        DispatchTensor: DispatchKindConversion<Autodiff<Inner>> + DispatchKindConversion<Inner>,
    {
        let l = history.len();
        if l + 1 > N {
            return None;
        }
        let qa = query.try_into_primitive::<Autodiff<Inner>>().ok()?;
        let has: Vec<_> = history
            .iter()
            .map(|h| h.clone().try_into_primitive::<Autodiff<Inner>>().ok())
            .collect::<Option<_>>()?;
        let q_t = Tensor::from_primitive::<Inner>(qa.primitive.clone());
        let hs_t: Vec<Tensor<3>> = has
            .iter()
            .map(|h| Tensor::from_primitive::<Inner>(h.primitive.clone()))
            .collect();

        let out_t = {
            #[cfg(feature = "cuda")]
            {
                type CudaBare = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>;
                if std::any::TypeId::of::<Inner>() == std::any::TypeId::of::<CudaBare>() {
                    if let Some(o) = super::depth_attend_cuda(&hs_t, &q_t) {
                        o
                    } else {
                        super::depth_attend_tensor_ad(&hs_t, q_t)
                    }
                } else {
                    super::depth_attend_tensor_ad(&hs_t, q_t)
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                super::depth_attend_tensor_ad(&hs_t, q_t)
            }
        };

        let out_prim = out_t.try_into_primitive::<Inner>().unwrap();
        let mut nodes: Vec<_> = Vec::with_capacity(N);
        for h in &has {
            nodes.push(h.node.clone());
        }
        nodes.push(qa.node.clone());
        while nodes.len() < N {
            nodes.push(qa.node.clone());
        }
        let nodes: [_; N] = nodes.try_into().unwrap();
        let prep = AttnResOp.prepare::<NoCheckpointing>(nodes);
        let out_adt = match prep.compute_bound().stateful() {
            OpsKind::Tracked(mut prep) => {
                for h in &has {
                    let _ = prep.checkpoint(h);
                }
                let _ = prep.checkpoint(&qa);
                prep.finish(l, out_prim)
            }
            OpsKind::UnTracked(prep) => prep.finish(out_prim),
        };
        Some(Tensor::from_primitive::<Autodiff<Inner>>(out_adt))
    }
}

#[cfg(feature = "autodiff")]
pub use ad::depth_attend_autodiff;

/// Stack the history into [L,B,T,D] (tensor path for the autodiff fallback).
pub fn stack_ad(history: &[Tensor<3>]) -> Tensor<4> {
    let stacked: Vec<Tensor<4>> = history
        .iter()
        .map(|h| h.clone().unsqueeze_dim::<4>(0))
        .collect();
    Tensor::cat(stacked, 0)
}

/// Pure tensor-path depth_attend (autodiff fallback).
pub fn depth_attend_tensor_ad(history: &[Tensor<3>], query: Tensor<1>) -> Tensor<3> {
    let n = history.len();
    let [b, t, d] = history[0].dims();
    let scale = (d as f64).powf(-0.5);
    let h_stack = stack_ad(history);
    let h_norm_sq = h_stack.clone().powf_scalar(2.0).sum_dim(3).add_scalar(1e-5);
    let h_norm = h_stack.clone() / h_norm_sq.sqrt().reshape([n, b, t, 1usize]);
    let q = query.reshape([1, 1, 1, d]);
    let scores = (q * h_norm).sum_dim(3).mul_scalar(scale);
    let weights = burn::tensor::activation::softmax(scores, 0);
    h_stack
        .mul(weights.reshape([n, b, t, 1usize]))
        .sum_dim(0)
        .reshape([b, t, d])
}

#[cfg(all(test, feature = "autodiff", feature = "cuda"))]
mod ad_tests {
    use super::*;
    use burn::tensor::{Device, Distribution, Tensor};

    type CudaBare = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>;

    fn to_host<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    #[test]
    fn depth_attend_fused_backward_matches_tensor() {
        let dev = Device::default().autodiff();
        let (l, b, t, d) = (4usize, 2usize, 3usize, 16usize);
        let hist: Vec<Tensor<3>> = (0..l)
            .map(|_| Tensor::<3>::random([b, t, d], Distribution::Normal(0.0, 1.0), &dev))
            .collect();
        let q = Tensor::<1>::random([d], Distribution::Normal(0.0, 1.0), &dev);

        // fused op graph
        let hf: Vec<Tensor<3>> = hist.iter().map(|h| h.clone().require_grad()).collect();
        let qf = q.clone().require_grad();
        let outf =
            crate::fused_attnres::depth_attend_autodiff::<CudaBare, 64>(&hf, qf.clone()).unwrap();
        let loss_f = outf.powf_scalar(2.0).sum();
        let grads_f = loss_f.backward();
        let dhf: Vec<Tensor<3>> = hf.iter().map(|h| h.grad(&grads_f).unwrap()).collect();
        let dqf = qf.grad(&grads_f).unwrap();

        // tensor path graph
        let ht: Vec<Tensor<3>> = hist.iter().map(|h| h.clone().require_grad()).collect();
        let qt = q.clone().require_grad();
        let outt = crate::depth_attend(&ht, qt.clone());
        let loss_t = outt.powf_scalar(2.0).sum();
        let grads_t = loss_t.backward();
        let dht: Vec<Tensor<3>> = ht.iter().map(|h| h.grad(&grads_t).unwrap()).collect();
        let dqt = qt.grad(&grads_t).unwrap();

        for i in 0..l {
            let md = maxdiff(&to_host(dhf[i].clone()), &to_host(dht[i].clone()));
            assert!(md < 1e-2, "dh[{i}] maxdiff {md}");
        }
        let md = maxdiff(&to_host(dqf), &to_host(dqt));
        assert!(md < 1e-2, "dq maxdiff {md}");
    }
}
