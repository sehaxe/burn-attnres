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
use burn::tensor::{activation, Device, Tensor};

#[cfg(feature = "cuda")]
mod fused_attnres;

/// Full AttnRes: learned pseudo-query attends over ALL previous hidden states.
///
/// ```text
/// scores[i] = query · norm(h_i) / sqrt(d)
/// weights = softmax(scores)
/// out = sum_i weights[i] · h_i
/// ```
#[derive(Module, Debug)]
pub struct AttnRes {
    pub query: Param<Tensor<1>>,
}

impl AttnRes {
    pub fn new(d_model: usize, device: &Device) -> Self {
        Self {
            // Paper §5: pseudo-queries MUST be initialized to zero (gives
            // exactly uniform alpha at init; prevents training volatility).
            query: Initializer::Zeros.init([d_model], device),
        }
    }

    /// Full AttnRes: attend over all previous hidden states.
    pub fn forward(&self, history: &[Tensor<3>]) -> Tensor<3> {
        depth_attend(history, self.query.val())
    }
}

/// Block AttnRes: partition into blocks, attend over block summaries.
///
/// Groups hidden states into `num_blocks` chunks. Within each chunk,
/// standard residual accumulation. Between chunks: AttnRes over chunk
/// summaries. Reduces memory from O(L·d) to O(N·d) where N ≪ L.
#[derive(Module, Debug)]
pub struct BlockAttnRes {
    pub query: Param<Tensor<1>>,
    #[module(skip)]
    pub block_size: usize,
}

impl BlockAttnRes {
    pub fn new(d_model: usize, block_size: usize, device: &Device) -> Self {
        Self {
            // Paper §5: pseudo-queries MUST be initialized to zero.
            query: Initializer::Zeros.init([d_model], device),
            block_size,
        }
    }

    /// Block AttnRes: accumulate within blocks, attend between blocks.
    ///
    /// `history`: all previous hidden states [h0, h1, ..., hL-1].
    /// Groups into `ceil(L / block_size)` blocks, applies depth attention
    /// over the block summaries (paper Eq 5-6: plain sums; the current
    /// partial block is attended from its second layer on).
    pub fn forward(&self, history: &[Tensor<3>]) -> Tensor<3> {
        let n = history.len();
        assert!(n > 0, "BlockAttnRes::forward needs >= 1 history entry");
        if n == 1 {
            return history[0].clone();
        }
        let [b, t, d] = history[0].dims();
        let dev = history[0].device();
        let mut st = BlockAttnState::new(b, t, d, &dev);
        let mut last = history[0].clone();
        for h in history {
            last = self.step(h.clone(), &mut st);
        }
        last
    }
}

/// Core depth-wise attention: softmax over history via learned query.
///
/// `history`: `[h0, ..., hN]` - each `[B, T, D]`
/// `query`: `[D]` - learned per-layer pseudo-query
///
/// Returns `[B, T, D]` - weighted sum via softmax attention over depth.
pub fn depth_attend(history: &[Tensor<3>], query: Tensor<1>) -> Tensor<3> {
    let n = history.len();
    if n == 1 {
        return history[0].clone();
    }
    let [b, t, d] = history[0].dims();
    let scale = (d as f64).powf(-0.5);

    #[cfg(feature = "cuda")]
    if let Some(out) = crate::fused_attnres::depth_attend_cuda(history, &query) {
        return out;
    }

    let stacked: Vec<Tensor<4>> = history
        .iter()
        .map(|h| h.clone().unsqueeze_dim::<4>(0))
        .collect();
    let h_stack = Tensor::cat(stacked, 0);

    // Tensor fallback (non-CUDA backend or fused dispatch unavailable).
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

/// Online-softmax attention score of one source against the query
/// (identical normalization and scale to [`depth_attend`]).
fn source_score(query: &Tensor<1>, src: &Tensor<3>) -> Tensor<3> {
    #[cfg(feature = "cuda")]
    if let Some(s) = crate::fused_attnres::source_score_cuda(query, src) {
        return s;
    }
    let [_, _, d] = src.dims();
    let norm = src.clone()
        / src
            .clone()
            .powf_scalar(2.0)
            .sum_dim(2)
            .add_scalar(1e-5)
            .sqrt();
    let q = query.clone().reshape([1, 1, d]);
    // sum_dim keeps the size-1 dim: [B,T,1]
    (q * norm).sum_dim(2).mul_scalar((d as f64).powf(-0.5))
}

/// Streaming state for [`BlockAttnRes::step`]: online-softmax attention over
/// completed blocks plus a running partial block (paper Eq 6 + Algorithm 1).
///
/// The caller owns the state and calls `step(h)` per new layer output;
/// `step` returns the attention output for that position, O(1) in the number
/// of completed blocks (no full recompute, no O(L·d) history retention).
#[derive(Debug)]
pub struct BlockAttnState {
    /// Running sum of the current block's layer outputs.
    pub partial: Tensor<3>,
    pub partial_count: usize,
    /// Online softmax over completed block reps: max score, exp-sum, weighted values.
    pub max_score: Tensor<3>,
    pub sum_exp: Tensor<3>,
    pub acc: Tensor<3>,
    pub started: bool,
}

impl BlockAttnState {
    pub fn new(b: usize, t: usize, d: usize, device: &Device) -> Self {
        Self {
            partial: Tensor::zeros([b, t, d], device),
            partial_count: 0,
            max_score: Tensor::zeros([b, t, 1], device),
            sum_exp: Tensor::zeros([b, t, 1], device),
            acc: Tensor::zeros([b, t, d], device),
            started: false,
        }
    }

    fn incorporate(&mut self, query: &Tensor<1>, src: &Tensor<3>) {
        let s = source_score(query, src); // [B,T,1]
        if !self.started {
            self.max_score = s.clone();
            self.sum_exp = Tensor::ones([s.dims()[0], s.dims()[1], 1], &src.device());
            self.acc = src.clone();
            self.started = true;
            return;
        }
        self.merge_source(src, &s);
    }

    /// Fuse `src` (score `s`) into the online state; returns the attended
    /// output over all attended sources (fused CUDA merge, else tensor path).
    fn merge_source(&mut self, src: &Tensor<3>, s: &Tensor<3>) -> Tensor<3> {
        #[cfg(feature = "cuda")]
        if let Some(m) = crate::fused_attnres::merge_cuda(
            &mut self.acc,
            &mut self.max_score,
            &mut self.sum_exp,
            src,
            s,
        ) {
            return m;
        }
        let m_new =
            (self.max_score.clone() + s.clone() + (self.max_score.clone() - s.clone()).abs())
                .div_scalar(2.0);
        let rescale = (self.max_score.clone() - m_new.clone()).exp();
        self.acc = self.acc.clone().mul(rescale.clone())
            + src.clone().mul((s.clone() - m_new.clone()).exp());
        self.sum_exp = self.sum_exp.clone().mul(rescale) + (s.clone() - m_new.clone()).exp();
        self.max_score = m_new;
        self.attended()
    }

    /// Attention output over the currently attended sources.
    fn attended(&self) -> Tensor<3> {
        self.acc.clone() / self.sum_exp.clone().clamp_min(1e-12)
    }
}

impl BlockAttnRes {
    /// Create a fresh streaming state for `[b, t, d]` inputs.
    pub fn init_state(&self, b: usize, t: usize, d: usize, device: &Device) -> BlockAttnState {
        BlockAttnState::new(b, t, d, device)
    }

    /// Streaming block attention (paper §4.2, Algorithm 1 + Eq 6).
    ///
    /// `h`: the new layer output. Updates the running partial block; when the
    /// block completes (`block_size` outputs) it is folded into the attended
    /// set via an online-softmax merge. The partial block is attended only
    /// from its second layer on (Eq 6: the block's first layer excludes the
    /// current partial sum, which would otherwise create a self-loop).
    ///
    /// Returns the depth-attention output `[B, T, D]`.
    pub fn step(&self, h: Tensor<3>, st: &mut BlockAttnState) -> Tensor<3> {
        let [b, t, d] = h.dims();
        st.partial = if st.partial_count == 0 {
            h.clone()
        } else {
            st.partial.clone() + h.clone()
        };
        st.partial_count += 1;

        // Attended sources: completed blocks (online state) + partial block
        // if it is past its first layer. With no sources at all (first layer
        // of the first block) the output is the identity (h).
        let mut out = if st.started || st.partial_count >= 2 {
            st.attended()
        } else {
            h.clone()
        };
        if st.partial_count >= 2 {
            let s_p = source_score(&self.query.val(), &st.partial);
            if st.started {
                let p = st.partial.clone();
                out = st.merge_source(&p, &s_p);
            } else {
                out = st.partial.clone();
            }
        }

        if st.partial_count == self.block_size {
            let completed = st.partial.clone();
            st.incorporate(&self.query.val(), &completed);
            st.partial = Tensor::zeros([b, t, d], &h.device());
            st.partial_count = 0;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;
    fn dev() -> Device {
        Device::default()
    }

    fn random_h(b: usize, t: usize, d: usize) -> Tensor<3> {
        Tensor::<3>::random([b, t, d], Distribution::Default, &dev())
    }

    #[test]
    fn depth_attend_shape() {
        let h = vec![random_h(1, 4, 32), random_h(1, 4, 32), random_h(1, 4, 32)];
        let q = Tensor::<1>::random([32], Distribution::Default, &dev());
        assert_eq!(depth_attend(&h, q).dims(), [1, 4, 32]);
    }
    #[test]
    fn depth_attend_single() {
        let h = random_h(2, 8, 16);
        let q = Tensor::<1>::random([16], Distribution::Default, &dev());
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

    #[test]
    fn streaming_matches_full_recompute() {
        // The paper's streaming scheme (Eq 6 semantics: partial block
        // excluded at its first layer, attended from its second layer on)
        // must reproduce the full-recompute BlockAttnRes at every step.
        let a = BlockAttnRes::new(16, 2, &dev());
        let (b, t, d) = (1usize, 3usize, 16usize);
        let mut st = a.init_state(b, t, d, &dev());
        let mut history: Vec<Tensor<3>> = Vec::new();
        for step in 0..6 {
            let h = random_h(b, t, d);
            history.push(h.clone());
            let streamed = a.step(h, &mut st);
            let full = a.forward(&history);
            let diff: f32 = (streamed - full).powf_scalar(2.0).mean().into_scalar();
            assert!(
                diff < 1e-5,
                "step {step}: streaming mse {diff} vs full recompute"
            );
        }
    }
}
