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

// ponytail: single-slot cache for the internal [L,B,T,D] stack. A fresh
// 800MB CUDA allocation costs ~15ms (default allocator); the stack never
// escapes depth_attend_cuda so reusing one buffer is safe. If call shapes
// vary, this degrades to one extra allocation per shape.
thread_local! {
    static STACK_CACHE: RefCell<Option<(usize, usize, usize, usize, CubeTensor<cubecl::cuda::CudaRuntime>)>> =
        const { RefCell::new(None) };
}

fn cached_stack(
    l: usize,
    b: usize,
    t: usize,
    d: usize,
    device: &burn::tensor::Device,
) -> CubeTensor<cubecl::cuda::CudaRuntime> {
    STACK_CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if let Some((cl, cb, ct, cd, buf)) = &*c {
            if (*cl, *cb, *ct, *cd) == (l, b, t, d) {
                return buf.clone();
            }
        }
        let s = Tensor::<4>::empty([l, b, t, d], device);
        let buf = cube_of(&s).expect("bare CUDA stack");
        *c = Some((l, b, t, d, buf.clone()));
        buf
    })
}

fn cube_of<const D: usize>(t: &Tensor<D>) -> Option<CubeTensor<cubecl::cuda::CudaRuntime>> {
    type B = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>;
    let prim = t.clone().try_into_primitive::<B>().ok()?;
    let c = (&prim as &dyn Any)
        .downcast_ref::<CubeTensor<cubecl::cuda::CudaRuntime>>()?;
    Some(c.clone())
}

fn cube_of_1(t: &Tensor<1>) -> Option<CubeTensor<cubecl::cuda::CudaRuntime>> {
    cube_of(t)
}

/// Full AttnRes without burn's `Tensor::cat` (measured ~23ms for the stacked
/// copy on CUDA). The per-layer scores launches fill the `[L, B, T, D]` stack
/// as a by-product (coalesced extra write), so the weighted sum runs in ONE
/// launch that reads the stack once and writes `out` once.
#[cube(launch_unchecked)]
fn attnres_scores_kernel<F: Float>(
    h: &[F],          // [B, T, D] one layer
    q: &[F],          // [D]
    scores: &mut [F], // [L, B, T] row `li`
    stacked: &mut [F], // [L, B, T, D] row `li`
    li: u32,
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
            stacked[(li as usize) * bt_count * d + base + col] = v;
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
        scores[(li as usize) * bt_count + bt] =
            dot * F::cast_from(scale) / (sq + F::new(1e-5_f32)).sqrt();
    }
}

/// Softmax over the L axis of [L, B, T] raw scores -> [L, B, T] weights.
/// One thread per (b,t): L < 256, so a single thread scans the L elements
/// twice (max, then exp/sum) with zero shared memory or syncs.
#[cube(launch_unchecked)]
fn attnres_softmax_kernel<F: Float>(
    scores: &[F],      // [L, B, T]
    weights: &mut [F], // [L, B, T]
    #[comptime] l: u32,
    #[comptime] bt_count: u32,
) {
    let bt = CUBE_POS_X as usize;
    let l = l as usize;
    let bt_count = bt_count as usize;

    let mut mx = F::new(-3.0e38_f32);
    let mut li = 0;
    while li < l {
        let s = scores[li * bt_count + bt];
        if s > mx {
            mx = s;
        }
        li += 1;
    }
    let mut sum_e = F::new(0.0_f32);
    let mut li = 0;
    while li < l {
        sum_e += (scores[li * bt_count + bt] - mx).exp();
        li += 1;
    }
    let mut li = 0;
    while li < l {
        weights[li * bt_count + bt] = (scores[li * bt_count + bt] - mx).exp() / sum_e;
        li += 1;
    }
}

/// out[b,t,d] = sum_l weights[l,b,t] * stacked[l,b,t,d]. One launch; each
/// thread accumulates its slab over the L axis in registers and writes `out`
/// once (no read-modify-write across launches).
#[cube(launch_unchecked)]
fn attnres_weighted_sum_kernel<F: Float>(
    stacked: &[F], // [L, B, T, D]
    weights: &[F], // [L, B, T]
    out: &mut [F], // [B, T, D]
    #[comptime] l: u32,
    #[comptime] bt_count: u32,
    #[comptime] d: u32,
    #[comptime] threads: u32,
    #[comptime] per: u32,
) {
    let bt = CUBE_POS_X as usize;
    let tid = UNIT_POS_X as usize;
    let l = l as usize;
    let bt_count = bt_count as usize;
    let d = d as usize;
    let threads = threads as usize;
    let per = per as usize;
    let base = bt * d;

    for j in 0..per {
        let col = j * threads + tid;
        if col < d {
            let mut acc = F::new(0.0_f32);
            for li in 0..l {
                acc += stacked[li * bt_count * d + base + col] * weights[li * bt_count + bt];
            }
            out[base + col] = acc;
        }
    }
}

/// `source_score`: RMS-norm + q·src·scale in one launch -> [B, T, 1].
#[cube(launch_unchecked)]
fn source_score_kernel<F: Float>(
    h: &[F],   // [B, T, D]
    q: &[F],   // [D]
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
    let base = bt * d;
    let lg = log_threads as usize;

    let mut red = Shared::<[F]>::new_slice(threads);

    let mut p = F::new(0.0_f32);
    for j in 0..per {
        let col = j * threads + tid;
        if col < d {
            let v = h[base + col];
            p += v * v;
        }
    }
    red[tid] = p;
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

    let mut p = F::new(0.0_f32);
    for j in 0..per {
        let col = j * threads + tid;
        if col < d {
            p += q[col] * h[base + col];
        }
    }
    red[tid] = p;
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

/// Online-softmax merge (paper Eq 9 streaming): fold `src` with score `s` into
/// the running state and emit the attended output in one launch.
///
/// state update: m' = max(m, s); acc' = acc·e^(m−m') + src·e^(s−m');
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

    let mut j = 0;
    while j < per {
        let col = j * threads + tid;
        if col < d {
            let i = base + col;
            let a = acc[i] * rescale + src[i] * w;
            acc[i] = a;
            let den = sum_new;
            let mut den = den;
            if den < F::new(1e-12_f32) {
                den = F::new(1e-12_f32);
            }
            out[i] = a / den;
        }
        j += 1;
    }
    if tid == 0 {
        max_s[bt] = m_new;
        sum_exp[bt] = sum_new;
    }
}

// ---- dispatch (bare CUDA backend only, callers fall back to tensor path) ----

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
    // empty (no memset): the scores pass fills the stack, the weighted pass
    // writes out exactly once
    let dev = &history[0].device();
    let stc = cached_stack(l, b, t, d, dev);
    let out = Tensor::<3>::empty([b, t, d], dev);
    let oc = cube_of(&out)?;
    let scores = Tensor::<2>::zeros([l, b * t], dev);
    let sc2 = cube_of(&scores)?;
    let weights = Tensor::<2>::zeros([l, b * t], dev);
    let wc = cube_of(&weights)?;
    let client = hc[0].client.clone();
    let per = (d as u32).div_ceil(THREADS);
    let dim = CubeDim {
        x: THREADS,
        y: 1,
        z: 1,
    };
    let scale = (d as f64).powf(-0.5) as f32;
    let bt = (b * t) as u32;
    unsafe {
        for (li, h) in hc.iter().enumerate() {
            attnres_scores_kernel::launch_unchecked::<f32, cubecl::cuda::CudaRuntime>(
                &client,
                CubeCount::Static(bt, 1, 1),
                dim,
                BufferArg::from_raw_parts(h.handle.clone(), b * t * d),
                BufferArg::from_raw_parts(qc.handle.clone(), d),
                BufferArg::from_raw_parts(sc2.handle.clone(), l * b * t),
                BufferArg::from_raw_parts(stc.handle.clone(), l * b * t * d),
                li as u32,
                scale,
                bt,
                d as u32,
                THREADS,
                per,
                THREADS.ilog2(),
            );
        }
        attnres_softmax_kernel::launch_unchecked::<f32, cubecl::cuda::CudaRuntime>(
            &client,
            CubeCount::Static(bt, 1, 1),
            CubeDim { x: 1, y: 1, z: 1 },
            BufferArg::from_raw_parts(sc2.handle.clone(), l * b * t),
            BufferArg::from_raw_parts(wc.handle.clone(), l * b * t),
            l as u32,
            bt,
        );
        attnres_weighted_sum_kernel::launch_unchecked::<f32, cubecl::cuda::CudaRuntime>(
            &client,
            CubeCount::Static(bt, 1, 1),
            dim,
            BufferArg::from_raw_parts(stc.handle, l * b * t * d),
            BufferArg::from_raw_parts(wc.handle, l * b * t),
            BufferArg::from_raw_parts(oc.handle.clone(), b * t * d),
            l as u32,
            bt,
            d as u32,
            THREADS,
            per,
        );
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
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
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
            / src.clone()
                .powf_scalar(2.0)
                .sum_dim(2)
                .add_scalar(1e-5)
                .sqrt();
        let expected =
            to_host((q.clone().reshape([1, 1, d]) * norm).sum_dim(2).mul_scalar(scale));
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
        for (l, b, t, d) in [
            (24usize, 1usize, 2048usize, 4096usize),
            (8, 2, 2048, 5120),
        ] {
            let hist: Vec<Tensor<3>> = (0..l)
                .map(|_| Tensor::<3>::random([b, t, d], Distribution::Normal(0.0, 1.0), &cdev))
                .collect();
            let q = Tensor::<1>::random([d], Distribution::Normal(0.0, 1.0), &cdev);
            for _ in 0..2 {
                let _ = depth_attend(&hist, q.clone());
            }
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                let r = depth_attend(&hist, q.clone());
                let _: f32 = r.clone().sum().into_scalar(); // flush
            }
            let tf = t0.elapsed() / 10;
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
