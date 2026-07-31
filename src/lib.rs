//! # burn-attnres - Attention Residuals for Burn
//!
//! | arXiv | Mode | What |
//! |-------|------|------|
//! | [2603.15031](https://arxiv.org/abs/2603.15031) | Full | Softmax over all previous layer outputs |
//! | [2603.15031](https://arxiv.org/abs/2603.15031) | Block | Softmax over block-level representations |
//!
//! Drop-in replacement for fixed residual accumulation. Mitigates PreNorm
//! dilution: learned query per layer attends over previous representations.
//!
//! Key results (Kimi Linear 48B): GPQA +7.5, HumanEval +3.1, MMLU +1.1.
use burn::module::{Module, Param};
use burn::nn::Initializer;
use burn::tensor::{activation, backend::Backend, Tensor};

/// Full AttnRes: learned pseudo-query attends over ALL previous hidden states.
///
/// ```text
/// scores[i] = query · norm(h_i) / sqrt(d)
/// weights = softmax(scores)
/// out = sum_i weights[i] · h_i
/// ```
#[derive(Module, Debug)]
pub struct AttnRes<B: Backend> {
    pub query: Param<Tensor<B, 1>>,
}

impl<B: Backend> AttnRes<B> {
    pub fn new(d_model: usize, device: &B::Device) -> Self {
        Self {
            // Paper §5: pseudo-queries MUST be initialized to zero (gives
            // exactly uniform alpha at init; prevents training volatility).
            query: Initializer::Zeros.init([d_model], device),
        }
    }

    /// Full AttnRes: attend over all previous hidden states.
    pub fn forward(&self, history: &[Tensor<B, 3>]) -> Tensor<B, 3> {
        depth_attend(history, self.query.val())
    }
}

/// Block AttnRes: partition into blocks, attend over block summaries.
///
/// Groups hidden states into `num_blocks` chunks. Within each chunk,
/// standard residual accumulation. Between chunks: AttnRes over chunk
/// summaries. Reduces memory from O(L·d) to O(N·d) where N ≪ L.
#[derive(Module, Debug)]
pub struct BlockAttnRes<B: Backend> {
    pub query: Param<Tensor<B, 1>>,
    #[module(skip)]
    pub block_size: usize,
}

impl<B: Backend> BlockAttnRes<B> {
    pub fn new(d_model: usize, block_size: usize, device: &B::Device) -> Self {
        Self {
            // Paper §5: pseudo-queries MUST be initialized to zero.
            query: Initializer::Zeros.init([d_model], device),
            block_size,
        }
    }

    /// Block AttnRes: accumulate within blocks, attend between blocks.
    ///
    /// `history`: all previous hidden states [h0, h1, ..., hL-1].
    /// Groups into `ceil(L / block_size)` blocks, applies depth attention.
    pub fn forward(&self, history: &[Tensor<B, 3>]) -> Tensor<B, 3> {
        let n = history.len();
        if n <= self.block_size {
            return depth_attend(history, self.query.val());
        }

        let num_blocks = n.div_ceil(self.block_size);
        let mut block_reps: Vec<Tensor<B, 3>> = Vec::with_capacity(num_blocks);

        // Within-block: standard residual accumulation. Between-block: AttnRes.
        // Paper Eq 5: block rep b_n = sum of layer outputs (plain sum, no mean).
        for b in 0..num_blocks {
            let start = b * self.block_size;
            let end = (start + self.block_size).min(n);
            let mut block_sum = history[start].clone();
            for h in &history[(start + 1)..end] {
                block_sum = block_sum + h.clone();
            }
            block_reps.push(block_sum);
        }

        depth_attend(&block_reps, self.query.val())
    }
}

/// Core depth-wise attention: softmax over history via learned query.
///
/// `history`: `[h0, ..., hN]` - each `[B, T, D]`
/// `query`: `[D]` - learned per-layer pseudo-query
///
/// Returns `[B, T, D]` - weighted sum via softmax attention over depth.
pub fn depth_attend<B: Backend>(history: &[Tensor<B, 3>], query: Tensor<B, 1>) -> Tensor<B, 3> {
    let n = history.len();
    if n == 1 {
        return history[0].clone();
    }
    let [b, t, d] = history[0].dims();
    let scale = (d as f64).powf(-0.5);

    let stacked: Vec<Tensor<B, 4>> = history
        .iter()
        .map(|h| h.clone().unsqueeze_dim::<4>(0))
        .collect();
    let h_stack = Tensor::cat(stacked, 0);

    // Per-element normalization (RMS-style, inline for perf)
    let h_norm_sq = h_stack.clone().powf_scalar(2.0).sum_dim(3).add_scalar(1e-5);
    let [nh, bh, th, _ns] = h_norm_sq.dims();
    let h_norm = h_stack.clone() / h_norm_sq.sqrt().reshape([nh, bh, th, 1usize]);

    let q = query.reshape([1, 1, 1, d]);
    let scores = (q * h_norm).sum_dim(3).mul_scalar(scale);
    let weights = activation::softmax(scores, 0);
    h_stack
        .mul(weights.reshape([nh, bh, th, 1usize]))
        .sum_dim(0)
        .reshape([b, t, d])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;
    use burn_ndarray::{NdArray, NdArrayDevice};
    type B = NdArray;
    fn dev() -> NdArrayDevice {
        NdArrayDevice::default()
    }

    fn random_h(b: usize, t: usize, d: usize) -> Tensor<B, 3> {
        Tensor::<B, 3>::random([b, t, d], Distribution::Default, &dev())
    }

    #[test]
    fn depth_attend_shape() {
        let h = vec![random_h(1, 4, 32), random_h(1, 4, 32), random_h(1, 4, 32)];
        let q = Tensor::<B, 1>::random([32], Distribution::Default, &dev());
        assert_eq!(depth_attend(&h, q).dims(), [1, 4, 32]);
    }
    #[test]
    fn depth_attend_single() {
        let h = random_h(2, 8, 16);
        let q = Tensor::<B, 1>::random([16], Distribution::Default, &dev());
        let out = depth_attend(std::slice::from_ref(&h), q);
        let d: Vec<f32> = (out - h)
            .into_data()
            .bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert!(
            d.iter().all(|x| x.abs() < 1e-4),
            "single item should be identity"
        );
    }
    #[test]
    fn full_attnres_module() {
        let a = AttnRes::new(64, &dev());
        let h = vec![random_h(1, 4, 64), random_h(1, 4, 64), random_h(1, 4, 64)];
        assert_eq!(a.forward(&h).dims(), [1, 4, 64]);
    }
    #[test]
    fn block_attnres_module() {
        let a = BlockAttnRes::new(64, 2, &dev());
        let h = vec![random_h(1, 4, 64), random_h(1, 4, 64), random_h(1, 4, 64)];
        assert_eq!(a.forward(&h).dims(), [1, 4, 64]);
    }
    #[test]
    fn block_attnres_small() {
        let a = BlockAttnRes::new(32, 4, &dev());
        let h = vec![random_h(1, 4, 32), random_h(1, 4, 32)]; // fewer than block_size
        assert_eq!(a.forward(&h).dims(), [1, 4, 32]);
    }
}
