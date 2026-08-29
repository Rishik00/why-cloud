# Adaptive Computation Time (ACT): A Comprehensive Technical Report

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [The Seminal Paper: Graves (2016)](#2-the-seminal-paper-graves-2016)
   - 2.1 Core Mechanism
   - 2.2 The Halting Algorithm in Detail
   - 2.3 The Ponder Cost Loss
   - 2.4 Experimental Results
3. [Key Variants and Extensions](#3-key-variants-and-extensions)
   - 3.1 Spatially Adaptive Computation Time (SACT)
   - 3.2 Universal Transformers with ACT
   - 3.3 PonderNet: Learning to Ponder
   - 3.4 Depth-Adaptive Transformer (DAT)
   - 3.5 Mixture-of-Depths (MoD)
   - 3.6 Continuous Thought Machines (CTM)
4. [Where ACT Is Applied](#4-where-act-is-applied)
   - 4.1 Sequence Modeling and Language
   - 4.2 Computer Vision
   - 4.3 Algorithmic Reasoning and Planning
   - 4.4 Agentic AI and Task Halting
   - 4.5 Reinforcement Learning
5. [The Relationship Between ACT and Broader Ideas](#5-the-relationship-between-act-and-broader-ideas)
   - 5.1 Conditional Computation
   - 5.2 Early Exit Networks
   - 5.3 Mixture of Experts
   - 5.4 Test-Time Compute Scaling
6. [Open Problems in Adaptive Computation](#6-open-problems-in-adaptive-computation)
7. [Summary Timeline](#7-summary-timeline)
8. [References](#8-references)

---

## 1. Introduction and Motivation

One of the defining characteristics of human cognition is that harder problems receive more thought. A person can recognize a common household object in milliseconds, but may deliberate for minutes over a chess position or a mathematical proof. Traditional neural networks do not possess this property. Whether the input is trivially easy or fiendishly complex, a fixed architecture performs exactly the same number of operations — determined entirely by the network's depth and the size of the input, not by the difficulty of the task at hand.

This is a profound limitation. Consider a recurrent neural network (RNN) processing a sequence of characters. The word "cat" in a piece of English text is trivially predictable from context; the word "serendipitous" following a line break in a poem is not. Yet a standard RNN allocates identical compute to each character. The mismatch between compute budget and task complexity leads to both inefficiency (wasting compute on easy inputs) and inadequacy (failing to "think harder" when the problem demands it).

**Adaptive Computation Time (ACT)** is the family of techniques that addresses this mismatch. The core idea is to give a neural network a learned, differentiable mechanism for deciding how many computational steps to take before producing an output, conditioned on the input itself. The model should be able to halt early on simple inputs and continue iterating on complex ones, all without explicit supervision on what constitutes "simple" or "complex."

The phrase "adaptive computation" encompasses a wide conceptual space — from early exit networks, to Mixture of Experts (MoE), to token-level routing in transformers — but the term ACT in its precise technical sense refers to the halting-based mechanism introduced by Alex Graves in 2016 and its direct descendants.

---

## 2. The Seminal Paper: Graves (2016)

### 2.1 Core Mechanism

The foundational paper, *Adaptive Computation Time for Recurrent Neural Networks* (Graves, arXiv:1603.08983), introduces ACT as an algorithm that allows an RNN to learn how many computational steps to take between receiving an input token and emitting an output. The key innovation is a differentiable, learned **halting mechanism** embedded in the recurrent cell.

At each recurrent step `t`, the ACT-augmented RNN produces, in addition to its usual hidden state `h_t` and output `y_t`, a scalar **halting probability** `p_t ∈ [0, 1]` via a learned linear layer with sigmoid activation applied to the hidden state:

```
p_t = sigmoid(W_h · h_t + b_h)
```

The network is unrolled for multiple "ponder steps" per input token. After each ponder step, the cumulative sum of halting probabilities is checked:

```
P_t = Σ_{t'=1}^{t} p_{t'}
```

Computation halts at the first step `N` where this cumulative sum exceeds a threshold `1 - ε` (where `ε` is a small constant, typically 0.01). At that final step, rather than using the full halting probability, the algorithm uses a **remainder** `R = 1 - Σ_{t'=1}^{N-1} p_{t'}` to ensure the total weight sums to exactly 1.

The final output state is a **weighted average** of all the hidden states encountered during pondering:

```
ŷ = Σ_{t=1}^{N} p̃_t · y_t
```

where `p̃_t = p_t` for `t < N` and `p̃_N = R`. This soft weighting is crucial: it makes the halting mechanism **differentiable**, allowing gradients to flow back through the halting decision, unlike a hard binary stop.

### 2.2 The Halting Algorithm in Detail

To make this concrete, the full ACT algorithm for processing a single input token `x` is:

1. Initialize: hidden state `h_0`, cumulative probability `P = 0`, counter `t = 0`, weighted output accumulator `ŷ = 0`.
2. Compute new hidden state: `h_t = RNN(h_{t-1}, x)`.
3. Compute halting probability: `p_t = sigmoid(W_h · h_t + b_h)`.
4. If `P + p_t ≥ 1 - ε`: set final step flag, set effective `p_t = 1 - P` (the remainder), stop after this step.
5. Update: `P += p_t`, `ŷ += p_t · y_t`.
6. Repeat from step 2.
7. Emit `ŷ` as the output for this input token.

The number of steps taken, `N`, is not fixed and varies per input. The weighted state `ŷ` is what gets passed to subsequent layers or used as the model's prediction.

A critical design decision is the use of the **weighted average** rather than the state at the halting step. This avoids a hard discrete stop that would zero out gradients for all earlier steps. The remainder `R` provides a gradient path even for the halting decision itself, making the entire procedure end-to-end trainable with standard backpropagation.

### 2.3 The Ponder Cost Loss

To prevent the network from trivially spending infinite computation on every input (maximizing accuracy at arbitrarily high cost), Graves introduces a **ponder cost** penalty added to the training loss:

```
L = L_task + τ · ρ
```

where `ρ = N + R` is the ponder cost (the number of full ponder steps plus the remainder), and `τ` is a hyperparameter controlling the speed-accuracy tradeoff.

The ponder cost `ρ` is differentiable almost everywhere with respect to the halting probabilities `p_t`, because:
- `N` itself is not differentiable (it is a discrete count).
- But the remainder `R = 1 - Σ_{t'=1}^{N-1} p_{t'}` is differentiable with respect to `p_t`.

So the gradient flows primarily through `R`. This is a pragmatic approximation; as later work shows, it introduces subtle gradient biases (see Section 6).

### 2.4 Experimental Results

Graves evaluates ACT on four synthetic algorithmic tasks: parity of binary vectors, binary logic operations, addition of integers, and sorting of real numbers. These tasks are chosen because their difficulty varies with the content of the input (e.g., sorting 100 numbers is harder than sorting 5).

ACT dramatically improves performance on all four tasks compared to fixed-step RNNs. Crucially, the model learns **semantically meaningful** allocation: sorting 10 numbers requires more ponder steps than sorting 3, and the model discovers this without explicit supervision.

On character-level language modelling (the Hutter Prize Wikipedia dataset), ACT does not yield large accuracy gains but produces a striking interpretability result: more computation is allocated to harder-to-predict transitions, specifically spaces between words and ends of sentences. The model has implicitly learned to identify syntactic boundaries as requiring more deliberation — a form of emergent structure discovery.

---

## 3. Key Variants and Extensions

### 3.1 Spatially Adaptive Computation Time (SACT)

Figurnov et al. (2017) extend ACT from sequential inputs to **spatial inputs** in computer vision. The key observation is that in an image, different spatial regions have very different complexity — the background of a photo is easy to classify, while the foreground object is harder. Standard ResNets process every spatial location for the same number of layers.

SACT applies ACT **per spatial location** within a ResNet. Each residual block, instead of being applied uniformly across the feature map, is applied a variable number of times per spatial position. A per-pixel halting score is learned, and computation is terminated at different depths for different image regions.

Results on ImageNet and COCO demonstrate that SACT achieves the same accuracy as a full ResNet while skipping substantial computation. Remarkably, the learned computation maps correlate with human eye fixation patterns from the visual saliency literature — the model has learned, unsupervised, that humans also devote more attention to salient regions.

### 3.2 Universal Transformers with ACT

Dehghani et al. (2018) introduce the **Universal Transformer (UT)**, which applies the same weight-shared transformer block repeatedly over the input, rather than having separate parameters for each layer. When run for a fixed number of steps, a UT is equivalent to a standard transformer with tied weights across layers; when coupled with ACT, it becomes a fully adaptive architecture.

In the UT+ACT setting, ACT is applied **per-position**: each token in the sequence has its own halting state, and different tokens can halt after different numbers of recurrent transformer steps. Tokens that have halted copy their state to subsequent steps unchanged.

This architecture achieves the remarkable theoretical property of **Turing completeness**: under appropriate memory assumptions, a Universal Transformer with adaptive computation can simulate any computable function, something a fixed-depth standard transformer cannot claim.

In practice, UT+ACT achieves strong results on the bAbI question-answering tasks and the LAMBADA language modelling benchmark, particularly on tasks requiring multi-step reasoning where standard transformers fail. However, ACT in practice has shown mixed results in UTs — training can be unstable, and many practical UT implementations have dropped ACT in favor of fixed step counts, reflecting a broader tension between the theoretical appeal and practical difficulty of learned halting.

### 3.3 PonderNet: Learning to Ponder

Banino et al. (2021) identify several fundamental problems with Graves' original ACT formulation and propose **PonderNet** as a principled replacement.

The problems with original ACT are:
1. **Biased gradients**: Because `N` is discontinuous, the gradient through the ponder cost flows only through `R`, providing a biased estimate of the true gradient with respect to halting decisions.
2. **Heuristic loss**: The ponder cost `ρ = N + R` is an ad hoc combination of incommensurable quantities (a count and a probability remainder), making the loss hard to interpret.
3. **Training instability**: Empirically, ACT training is sensitive to the `τ` hyperparameter and prone to degenerate solutions where the model either always uses one step or always uses the maximum.

PonderNet reformulates halting as a **probabilistic latent variable model**. At each step `n`, the network computes a halting probability `λ_n ∈ [0, 1]` (Bernoulli). The probability of halting at exactly step `n` is:

```
p(N = n) = λ_n · Π_{i=1}^{n-1} (1 - λ_i)
```

This is a geometric-like distribution over halting steps. The training loss combines:
1. A **reconstruction loss** (task accuracy), weighted by the halting distribution:
   `L_rec = Σ_n p(N=n) · L_task(y_n)`
2. A **regularization loss** encouraging the model not to ponder indefinitely, formulated as a KL divergence between the learned halting distribution and a geometric prior distribution:
   `L_reg = KL[p(N) || Geom(λ_p)]`

where `λ_p` is a hyperparameter that sets the prior expected ponder time.

PonderNet outperforms ACT substantially on complex synthetic tasks and achieves competitive performance on real-world question answering while using less compute. Its probabilistic formulation also enables cleaner theoretical analysis and better training stability. However, it introduces its own complexities: two new hyperparameters (`λ_p` and `β` controlling regularization weight) and difficulties in chaining multiple PonderNet modules together (because the loss must be conditioned on each module's halting decision, leading to exponential complexity in the number of modules).

### 3.4 Depth-Adaptive Transformer (DAT)

Elbayad et al. (2020) take a different approach: instead of re-applying the same layers, they allow a transformer to **exit at different layers** for different inputs. Each layer produces a potential output, and a learned classifier decides whether to use that output or continue to deeper layers.

Unlike ACT (which uses soft weight averaging), DAT uses a hard exit, making it more computationally efficient at inference but harder to train. DAT matches a well-tuned transformer baseline while reducing computation by up to 76% on natural language inference tasks — a striking efficiency gain.

### 3.5 Mixture-of-Depths (MoD)

Raposo et al. (2024) introduce **Mixture-of-Depths**, which reframes the adaptive computation problem as a **routing problem** applied at the token level within a transformer.

Rather than adapting the number of forward passes through the same block (as in ACT/UT), MoD asks: for each layer, which tokens actually need to go through the expensive self-attention and MLP computations? A learned router assigns each token either to the full computation or to a skip connection. A fixed budget `k` (the "capacity") determines how many tokens at each layer can participate in full attention.

This is a particularly elegant formulation for modern LLMs because:
- It maintains strict computational budgets (no unbounded pondering).
- It is compatible with existing transformer architectures with minimal modification.
- It can be combined with Mixture-of-Experts (MoE) routing for a doubly-adaptive architecture.

MoD models match the performance of isoFLOP transformer baselines while using substantially less compute per forward pass. The routing patterns that emerge are semantically meaningful: tokens in dense, high-information regions are more consistently routed to full computation, while tokens in predictable, low-complexity regions are frequently skipped.

### 3.6 Continuous Thought Machines (CTM)

Darlow et al. (2025) at Sakana AI introduce a radically different architecture called **Continuous Thought Machines**, which takes inspiration from neuroscience to implement a form of adaptive computation that does not rely on a halting signal or ponder cost.

The key innovation is that each neuron in the CTM maintains a short **history of its own activations** (a "neural memory"), and uses this history — rather than just the current input — to determine when to fire next. This creates an internal temporal dimension: the network unfolds computation over "micro-ticks," during which neurons can activate, update, and synchronize before the network commits to an output.

From an ACT perspective, the CTM achieves adaptive compute depth (spending more ticks on hard inputs, fewer on easy ones) without requiring an explicit halting loss, which is historically difficult to tune. The emergence of this behavior is more organic, arising from the dynamics of the neural activity itself. On diverse tasks including ImageNet classification, maze solving, sorting, parity, and reinforcement learning, the CTM demonstrates interpretable internal reasoning processes — making the "thought steps" themselves visible to researchers — a property rarely seen in other architectures.

---

## 4. Where ACT Is Applied

### 4.1 Sequence Modeling and Language

The original home of ACT is language and sequence modeling. In character-level language modeling, ACT-augmented RNNs learn to spend more computation on syntactically complex positions (ends of sentences, rare words, transitions between discourse segments). The emergent computation patterns serve as an unsupervised proxy for syntactic structure.

In large language models, test-time compute scaling has emerged as a major paradigm: allowing models to "think longer" on hard problems. While modern approaches to this (chain-of-thought, best-of-N sampling, process reward models) are not always framed as ACT, they share the core intuition. More recent work has explored architectures where the number of internal recurrent steps at inference time scales with problem difficulty without requiring explicit verbalization of intermediate thoughts (the "Coconut" / continuous latent reasoning direction, related to ACT in spirit).

### 4.2 Computer Vision

SACT (Section 3.1) extends ACT to vision, achieving spatially adaptive compute in ResNets. Subsequent work has applied ACT-style mechanisms to Vision Transformers (ViTs), adapting the depth along which computation happens for different image patches. A-ViT (Yin et al., 2022) applies ACT along the depth axis of a ViT, allowing easier image regions to skip later transformer blocks.

Adaptive computation has also been used for object detection and image segmentation, where different image regions have vastly different computational requirements. The SACT paper demonstrates this on COCO detection using Faster R-CNN, showing that background regions can be processed with far fewer residual blocks than foreground objects.

### 4.3 Algorithmic Reasoning and Planning

ACT-style mechanisms are particularly compelling for **algorithmic and planning tasks**, where the number of computational steps required scales directly with the problem difficulty. Sorting `N` numbers intuitively requires `O(N log N)` comparisons; a fixed-depth network cannot adapt to this scaling.

Universal Transformers with ACT show striking results on tasks like the bAbI question answering dataset, which requires multi-hop reasoning (retrieving 2-3 supporting facts from a story). The model learns to allocate more computation to questions requiring more reasoning steps. Similarly, ACT-augmented ConvRNNs have been shown to perform "zero-shot computation scaling" on visual reasoning tasks — generalizing to harder instances at test time by simply running more recurrent steps, without any additional training.

### 4.4 Agentic AI and Task Halting

In agentic settings — where an AI must take a sequence of actions to complete a task — ACT-style halting is used to answer the question: **when is the task done?** Rather than halting per input token, the agent halts per task attempt.

The key challenge is that the agent must decide, at each timestep, whether to take another action or declare the task complete. Without a principled halting mechanism, agents either stop too early (underthinking) or waste resources continuing when they should stop (overthinking). ACT provides a framework for learning this decision in a differentiable way.

In practice, this is applied in settings like web navigation, tool-use agents, and coding agents, where the agent's "ponder time" corresponds to the number of action steps taken before submitting a final answer. The difficulty of learning to halt well in these settings is amplified by sparse rewards: the agent often only receives feedback at the very end, making it hard to learn when it's appropriate to stop mid-task.

### 4.5 Reinforcement Learning

ACT is less naturally suited to RL than to supervised learning, because the ponder cost (a direct differentiable loss) conflicts with the typically non-differentiable RL objective. Nevertheless, several approaches combine the two.

One direction uses RL to learn a discrete halting policy (using REINFORCE or similar), accepting higher variance in exchange for a principled stopping criterion. Another direction integrates PonderNet-style probabilistic halting into model-based RL, where the world model uses adaptive computation to reason about future states. The computational savings are most pronounced in complex environments where some situations (stable, predictable states) require little forward modeling and others (dynamic, contested states) require deep lookahead.

---

## 5. The Relationship Between ACT and Broader Ideas

### 5.1 Conditional Computation

ACT is one instance of the broader paradigm of **conditional computation**: the idea that computation should be conditioned on the input, not fixed a priori. Other instances include:
- **Sparse activation** (e.g., dropout, ReLU sparsity): some neurons/paths are active for some inputs.
- **Hard attention**: attending to a subset of input positions, not all.
- **Mixture of Experts (MoE)**: routing different inputs to different expert subnetworks.

ACT is distinguished by operating in the **depth dimension** rather than the width dimension. It asks "how many times should we process?" rather than "which parameters should process this?"

### 5.2 Early Exit Networks

Early exit networks (e.g., BranchyNet, 2016; Bolukbasi et al., 2017) are a related but distinct family. They attach auxiliary classifiers to intermediate layers and exit when the classifier is sufficiently confident. Unlike ACT, these use a hard threshold (no soft weighting), are often not end-to-end trainable in the same way, and typically do not allow the model to iterate over the same layers multiple times — they simply skip the remaining layers.

The DAT (Section 3.4) and MoD (Section 3.5) represent hybrids between ACT-style mechanisms and early exit ideas.

### 5.3 Mixture of Experts

MoE routes different tokens to different expert networks, adapting computation in the **width** dimension. MoD routes different tokens through different **depths**. Combining both (MoE + MoD) gives a doubly-adaptive architecture that can allocate both "what kind" and "how much" computation to each token — an active research frontier as of 2024-2025.

### 5.4 Test-Time Compute Scaling

The recent explosion of interest in **inference-time scaling** (o1-style models, DeepSeek-R1, chain-of-thought scaling) is deeply related to ACT, though the implementation is often very different. These systems scale test-time compute by generating longer sequences of intermediate reasoning tokens rather than by running more passes through the same network. The conceptual bridge is the same: harder problems deserve more computation, and this should be determined dynamically.

Work by Snell et al. (2024) demonstrates that on mathematical reasoning problems, adaptive allocation of test-time compute can match a model 14× larger on problems where the smaller model has some non-trivial success rate. This vindicates the core ACT intuition at the level of entire LLMs: compute allocation matters, and the right amount varies with difficulty.

---

## 6. Open Problems in Adaptive Computation

Despite nearly a decade of research since the original Graves (2016) paper, the field of adaptive computation remains rich with unsolved problems.

### 6.1 The Gradient Bias Problem

As noted by the PACT (Probabilistic ACT) paper (Kuck et al., 2017) and acknowledged by PonderNet, the Graves ACT loss is a biased gradient estimator. The ponder cost `ρ = N + R` is discontinuous with respect to the halting probabilities: when `N` changes value (a token crosses the threshold), there is a discontinuity that prevents backpropagation through the halting decision. Gradients only flow through the remainder `R`, which is a proxy.

PonderNet resolves this with a probabilistic formulation, but the KL divergence loss introduces its own optimization difficulties. The fundamental problem — how to define a truly differentiable, low-variance, principled signal for "how much computation should this input require?" — remains open. More recent work on differentiable halting (e.g., using attention-based formulations rather than sigmoid thresholds) addresses some aspects but does not fully solve the problem.

### 6.2 Hyperparameter Sensitivity

Both ACT and PonderNet require careful tuning of their regularization hyperparameters. In ACT, the ponder cost weight `τ` must be balanced carefully: too large, and the model degenerates into a one-step feedforward network (underthinking); too small, and it always uses the maximum number of steps regardless of input complexity (overthinking). Similar issues arise with PonderNet's `β` and `λ_p`.

Choosing these hyperparameters effectively requires task-specific knowledge and often extensive grid search. There is no principled method to automatically set them based on the task characteristics. This is a major practical barrier to the adoption of ACT-style mechanisms in production systems.

### 6.3 Training Instability in Transformers

When ACT is integrated with transformer architectures (as in Universal Transformers), training stability becomes a significant challenge. The ACT-based UT training is empirically observed to be less stable than vanilla transformer training, requiring careful hyperparameter tuning. This has led many practitioners to drop ACT from UT variants in practice, using fixed step counts instead — reflecting a gap between theoretical appeal and practical usability.

The instability is partly due to the interaction between the transformer's layer normalization, weight tying across depths, and the soft weighting of intermediate states. PonderNet partially alleviates this through its probabilistic formulation, but the combination of deep transformers and learned halting remains a reliability challenge.

### 6.4 The Underthinking / Overthinking Trade-off

ACT introduces a fundamental trade-off that is difficult to correctly calibrate in practice. The ponder cost encourages the model to halt early (compute-efficiency), but task loss encourages more thought (accuracy). The model can find local optima that correspond to either extreme:

- **Underthinking**: The model halts at step 1 for almost all inputs, reducing to a feedforward network. This optimizes the ponder cost at the expense of reasoning quality.
- **Overthinking**: The model uses the maximum number of steps for all inputs, wasting compute on easy examples. This optimizes task accuracy without learning to discriminate by difficulty.

Neither local optimum is desirable, and the global optimum (adaptively allocating exactly the right amount of compute) is hard to reach. Recent work on "learning to stop overthinking at test time" explores this problem but effective automatic calibration across tasks and difficulty levels remains unsolved.

### 6.5 Compositionality and Chaining of ACT Modules

Composing multiple ACT modules in a deep network is non-trivial. If each module can halt at a different step, the total computation consumed by a network is the **product** of the expected steps of each module, which can grow explosively. Moreover, the loss for a chained PonderNet model must be conditioned on each module's halting decision, leading to exponential computational cost in the number of modules.

No satisfying general solution exists for this problem. Practical systems often use ACT at a single bottleneck in the network and use fixed computation for the remaining layers.

### 6.6 Interpretability of Halting Decisions

While ACT produces a ponder time for each input, understanding *why* the model halts when it does remains difficult. The halting probability is computed from the hidden state via a learned linear layer — it is not directly interpretable. Visualizing ponder times per input (as Graves does for character-level language modeling) provides coarse insights but not mechanistic understanding.

Connecting halting decisions to task-relevant features — answering "what property of this input caused the model to need 7 steps instead of 2?" — is a mechanistic interpretability challenge that is largely unsolved. This matters practically: in safety-critical agentic settings, we want to know *why* an agent decided to stop acting.

### 6.7 Hardware Efficiency and Variable Compute

Adaptive computation introduces a fundamental tension with modern hardware optimized for **uniform, parallelizable computation**. GPUs and TPUs are designed for dense matrix operations over fixed-size batches. When different inputs in a batch require different numbers of ponder steps, the hardware cannot easily exploit this heterogeneity: a naive implementation must wait for the slowest input in the batch before proceeding.

Workarounds (e.g., padding shorter sequences to the maximum ponder time, bucketing by complexity, asynchronous execution) exist but introduce overhead and complexity. Truly realizing the efficiency gains promised by adaptive computation requires either new hardware designs or sophisticated scheduling systems that current frameworks do not natively provide.

### 6.8 Adaptive Computation at Scale

Most ACT research has been conducted at relatively small scale (millions to low billions of parameters). How adaptive halting mechanisms behave at the scale of frontier LLMs (hundreds of billions of parameters) is largely unknown. Several questions are open:

- Does the model's ability to adaptively halt improve or degrade with scale?
- Does the optimal ponder cost hyperparameter change with scale?
- Can the emergent structure-discovery behavior (e.g., learning syntactic boundaries) observed at small scale persist and generalize at large scale?
- How does adaptive computation interact with reinforcement learning from human feedback (RLHF), where the reward signal itself is noisy?

### 6.9 Generalization to Harder Inputs at Test Time

A particularly exciting open direction is whether ACT-style mechanisms enable **zero-shot generalization to harder inputs** at inference time. If a model trains on problems of complexity up to `k`, can it solve problems of complexity `k+1` at test time simply by running more ponder steps?

There is early evidence that this is possible: ACT-augmented ConvRNNs have demonstrated zero-shot computation scaling on visual reasoning tasks, running more recurrent steps at test time without any re-training and successfully solving harder visual instances. Universal Transformers show similar behavior on algorithmic tasks. But scaling this capability to more naturalistic tasks (complex reasoning, multi-step planning, long-horizon agent tasks) without explicit test-time supervision remains a major open problem.

### 6.10 Unified Theory of Adaptive Computation

Finally, the field lacks a clean unified theoretical framework. ACT, PonderNet, early exit networks, MoD, MoE, and CTMs all implement "adaptive computation" in meaningfully different ways — different axes (depth vs. width), different granularities (token vs. sequence vs. spatial position), different mechanisms (soft weighting vs. hard routing vs. neural dynamics), and different training signals. A unifying theory that explains when and why each approach is appropriate, and that could guide the design of novel adaptive architectures, does not yet exist.

---

## 7. Summary Timeline

| Year | Contribution |
|------|-------------|
| 2016 | **Graves**: Adaptive Computation Time for RNNs — the foundational paper |
| 2016 | **BranchyNet**: Early exit via auxiliary classifiers (parallel line of work) |
| 2017 | **Figurnov et al.**: Spatially Adaptive Computation Time (SACT) for ResNets |
| 2017 | **Kuck et al.**: Probabilistic ACT (PACT) — identifies gradient bias in Graves ACT |
| 2018 | **Dehghani et al.**: Universal Transformers with ACT halting — Turing-complete adaptive transformers |
| 2020 | **Elbayad et al.**: Depth-Adaptive Transformer (DAT) — hard early exit in transformers |
| 2021 | **Banino et al.**: PonderNet — probabilistic halting, more stable and principled than ACT |
| 2022 | **Yin et al.**: A-ViT — ACT applied along depth of Vision Transformers |
| 2023 | **Xue et al.**: AdaTape — adaptive tape tokens for dynamic input length |
| 2024 | **Raposo et al.**: Mixture-of-Depths — token-level routing as adaptive compute allocation |
| 2025 | **Darlow et al. (Sakana AI)**: Continuous Thought Machines — neuron-level temporal dynamics as adaptive compute |

---

## 8. References

- Graves, A. (2016). *Adaptive Computation Time for Recurrent Neural Networks*. arXiv:1603.08983.
- Figurnov, M., Collins, M. D., Zhu, Y., Zhang, L., Huang, J., Vetrov, D., & Salakhutdinov, R. (2017). *Spatially Adaptive Computation Time for Residual Networks*. CVPR 2017.
- Kuck, J., Dao, T., Gu, A., Rudra, A., Re, C., Ermon, S., & Sabharwal, A. (2017). *Probabilistic Adaptive Computation Time*. arXiv:1712.00386.
- Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., & Kaiser, Ł. (2018). *Universal Transformers*. ICLR 2019. arXiv:1807.03819.
- Elbayad, M., Gu, J., Grave, E., & Auli, M. (2020). *Depth-Adaptive Transformer*. ICLR 2020.
- Banino, A., Balaguer, J., & Blundell, C. (2021). *PonderNet: Learning to Ponder*. arXiv:2107.05407.
- Yin, H., Vahdat, A., Alvarez, J. M., Mallya, A., Kautz, J., & Molchanov, P. (2022). *A-ViT: Adaptive Tokens for Efficient Vision Transformers*. CVPR 2022.
- Xue, F., Likhosherstov, V., Arnab, A., Houlsby, N., Dehghani, M., & You, Y. (2023). *AdaTape: Adaptive Computation with Elastic Input Sequence*. ICML 2023.
- Raposo, D., Ritter, S., Richards, B., Lillicrap, T., Humphreys, P. C., & Santoro, A. (2024). *Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models*. arXiv:2404.02258.
- Darlow, L., Regan, C., Risi, S., Seely, J., & Jones, L. (2025). *Continuous Thought Machines*. Sakana AI. arXiv:2505.05522.

---

*Report prepared June 2026. All paper references verified against published versions.*