# On-Policy Distillation for Language Models
**A conceptual and practical introduction to learning from a student's own generations**

---

## Introduction

Knowledge distillation is usually introduced with a simple picture: a capable, expensive *teacher* produces a richer learning signal, and a smaller *student* learns to imitate it. In classification, that signal is often the teacher's probability distribution over labels. In modern language models, it may be a full response, a token-level probability distribution, a rationale, a preference, or an evaluation of a student's attempt. The aim is the same in every case: transfer useful behavior into a model that is cheaper, faster, or easier to deploy.

The phrase **on-policy distillation** names a particular choice about where the training sequences come from. The student generates a response using its current policy; the teacher then evaluates, labels, or supplies token-level guidance on that response. The student therefore learns on states it actually visits. By contrast, **off-policy distillation** usually trains the student on a fixed corpus of teacher demonstrations or other externally generated sequences. This distinction looks modest, but it changes the optimization problem, the data pipeline, and the failure modes.

The user request that motivated this document used the phrase *non-policy distillation*. That is not a standard term in the literature. Because the requested contrasts and applications explicitly concern on-policy and off-policy methods, this document treats **on-policy distillation** as the intended subject.

The central intuition is straightforward. An autoregressive model makes one token prediction, appends the sampled token to its context, and repeats. A small error can therefore move the model into a context that never appeared in its demonstration data. Future predictions are then made from an unfamiliar state, and errors can compound. Off-policy training teaches the student what the teacher does in teacher-like contexts. On-policy training asks a more operational question: *given the contexts this student creates, what would the teacher want it to do next?*

This document develops that idea from its roots in knowledge distillation, exposure bias, and interactive imitation learning. It separates two choices that are often conflated---the distribution used to collect trajectories and the divergence used to compare teacher and student---then surveys the main algorithmic families, modern uses, practical recipes, and unresolved problems. The emphasis is conceptual: enough mathematics to make the distinctions exact, but always tied back to what a training system must actually do.

## The basic object: a policy over text

For a prompt $x$, an autoregressive language model with parameters $\theta$ defines a distribution over a response $y=(y_1,\ldots,y_T)$:

$$
p_\theta(y\mid x)=\prod_{t=1}^{T}p_\theta(y_t\mid x,y_{<t}).
$$

Calling this distribution a *policy* borrows the language of reinforcement learning, but does not by itself imply that reinforcement learning is being used. A policy is simply a rule that maps the current state---here, the prompt and generated prefix---to a distribution over actions, here the next token. Let $p_T$ denote the teacher and $p_S$ the student.

A token-level distillation loss compares their next-token distributions at a context $s_t=(x,y_{<t})$. One common choice is forward Kullback--Leibler divergence:

$$
D_{\mathrm{KL}}(p_T\Vert p_S)
=\sum_v p_T(v\mid s_t)\log\frac{p_T(v\mid s_t)}{p_S(v\mid s_t)}.
$$

The crucial question is not only which divergence appears inside the loss. It is also **who generated the prefix $y_{<t}$**. A general objective can be written as

$$
\mathcal{L}(\theta)=
\mathbb{E}_{x\sim\mathcal{D},\ y\sim q(\cdot\mid x)}
\left[\sum_{t=1}^{T} d\!\left(p_T(\cdot\mid x,y_{<t}),
p_S(\cdot\mid x,y_{<t})\right)\right],
$$

where $q$ is the **rollout distribution** and $d$ is a chosen discrepancy. If $q=p_T$ or a fixed dataset, training is off-policy with respect to the current student. If $q=p_S$, it is on-policy. A mixture

$$
q_\alpha=\alpha p_T+(1-\alpha)p_S
$$

gives a continuum between the two and is often more useful than a rigid binary label.

This yields the first important rule for reading the literature:

> **Sampling policy and divergence direction are separate axes.** “On-policy” describes where contexts come from; “forward KL” or “reverse KL” describes how distributions are compared at those contexts.

An on-policy method can use forward KL, reverse KL, Jensen--Shannon divergence, a token reward, an outcome score, or even black-box textual feedback. Likewise, reverse KL does not automatically make a method on-policy.

## Where the idea came from

### Classical knowledge distillation

Modern distillation is commonly traced to Hinton, Vinyals, and Dean's formulation of training a compact student from a teacher's softened output distribution [2], building on earlier work on model compression [1]. Soft targets reveal relationships that hard labels hide: two wrong classes can receive very different probabilities, communicating which alternatives the teacher considers plausible. In a classifier there is no long generated prefix, so the training input distribution and the inference input distribution can often be treated as the same.

Sequence generation breaks that convenience. At training time, maximum likelihood normally conditions on the ground-truth prefix. At inference time, the model conditions on its own sampled tokens. Sequence-level knowledge distillation made teacher-generated translations into a simplified training corpus [6]. It was effective and operationally attractive, but the resulting student still trained on externally supplied sequences rather than on the prefixes it would produce itself.

### Exposure bias and compounding error

The mismatch between teacher-forced training and free-running generation is usually called **exposure bias**. Scheduled Sampling gradually replaced ground-truth previous tokens with model-generated tokens during training [3]. Professor Forcing instead tried to align the hidden-state dynamics of teacher-forced and free-running modes [4]. These methods were not LLM distillation in the modern sense, but they framed the same structural problem: a sequence model should learn to behave in the states induced by its own earlier decisions.

There is a subtle point here. Exposure bias is not a proof that every model must be trained entirely on-policy. Strong off-policy data, broad prompt coverage, and sufficiently capable students can work extremely well. On-policy collection is a targeted response when the student's own state distribution differs enough that static demonstrations cease to teach recovery.

### Interactive imitation learning

The clearest theoretical ancestor is DAgger---Dataset Aggregation---from interactive imitation learning [5]. A learner executes its current policy, an expert labels the states the learner visits, and those corrected states are added to the training set. Repeating this process turns an initially narrow supervised dataset into one that covers the learner-induced state distribution.

Autoregressive Knowledge Distillation through Imitation Learning, or ImitKD, brought this logic directly to translation and summarization [7]. The student explores its own generations and the teacher provides corrective distributions at encountered prefixes. Generalized Knowledge Distillation (GKD) later made the pattern explicit for large language models and demonstrated it across summarization, translation, arithmetic reasoning, and instruction tuning [9].

The lineage can be summarized as follows:

| Period | Problem being addressed | Representative idea |
|---|---|---|
| 2006--2015 | Compress a powerful predictor | Match teacher outputs or softened probabilities [1,2] |
| 2015--2016 | Train/test mismatch in sequence models | Expose the model to generated prefixes or align free-running dynamics [3,4] |
| 2011 onward | Errors change the learner's future state distribution | Query an expert on learner-visited states, as in DAgger [5] |
| 2016--2020 | Distill autoregressive generators | Teacher-generated sequences [6], then imitation learning on student prefixes [7] |
| 2023--2025 | Distill instruction following and reasoning in LLMs | GKD, MiniLLM, adaptive mixtures, and contrastive objectives [9--14] |
| 2025--2026 | Combine dense teacher feedback with reasoning rollouts | On-policy reasoning, hybrid post-training, self-play, and diffusion-model variants [17--25] |

## On-policy and off-policy distillation

### Off-policy distillation

In the common black-box form, a teacher API is prompted once to produce a dataset. The student is then fine-tuned on prompt--response pairs:

$$
\mathcal{L}_{\mathrm{SFT}}=-\mathbb{E}_{(x,y^T)\sim\mathcal{D}_T}
\sum_t\log p_S(y_t^T\mid x,y_{<t}^T).
$$

If teacher logits are available, the student can instead match the full teacher distribution on teacher or dataset prefixes. In both cases the expensive teacher interaction is decoupled from student optimization. The corpus can be filtered, audited, cached, reused across experiments, and trained on with an ordinary supervised stack.

Reasoning distillation commonly uses this design. A teacher generates answers and rationales; the student learns from those traces. Distilling Step-by-Step showed that rationales can provide richer supervision than labels alone [10]. DeepSeek-R1 reported that reasoning data generated by larger models can substantially improve smaller Qwen and Llama-family students [16]. These are important distillation results, but the data are teacher-generated and therefore off-policy with respect to the student being trained.

The weakness appears when the student deviates from those polished traces. A teacher demonstration shows how an expert proceeds from an expert prefix. It may never show how to recover from the student's characteristic false start, malformed tool call, premature answer, or locally plausible but globally inconsistent proof.

### On-policy distillation

An on-policy loop periodically samples outputs from the current student. With white-box access, the teacher returns logits for every student-generated prefix and the student minimizes a token-level divergence. With black-box access, the teacher may critique the trajectory, propose a correction, rank candidates, or assign an outcome score. The defining property is that feedback is attached to the student's current behavior distribution.

A minimal loop is:

1. Sample prompts from a prompt distribution.
2. Generate one or more trajectories with the current student.
3. Ask the teacher for feedback on those trajectories or their prefixes.
4. Update the student using that feedback.
5. Refresh trajectories often enough that the training distribution follows the changing student.

This costs more than pre-generating a corpus. Rollouts and teacher inference sit inside the training loop, while changing student weights make stale samples progressively less on-policy. The benefit is relevance: supervision is concentrated on errors the current student actually makes.

### The practical comparison

| Dimension | Off-policy distillation | On-policy distillation |
|---|---|---|
| Sequence source | Fixed data, human answers, or teacher generations | Current or recent student generations |
| Teacher timing | Usually before training | During training or in frequent refresh rounds |
| State coverage | Teacher-like and dataset-like contexts | Student-induced contexts, including mistakes |
| Infrastructure | Simple, cacheable supervised pipeline | Coupled rollout, scoring, and optimization pipeline |
| Teacher cost | Paid once per example; reusable | Repeated as the student changes |
| Stability | Generally high | Sensitive to rollout quality, lag, and estimator variance |
| Main strength | Efficient transfer of target behavior | Corrective feedback where the student needs it |
| Main weakness | Exposure mismatch and weak recovery training | Compute, memory, serving, and synchronization cost |
| Best early use | Cold start and broad capability transfer | Refinement after the student can produce meaningful attempts |

The table suggests a common recipe: **off-policy first, on-policy second**. A cold student produces low-quality trajectories that are expensive for a teacher to annotate and may carry little learning signal. Teacher demonstrations establish a useful support; on-policy rounds then concentrate on the student's residual errors. Recent empirical analysis likewise emphasizes compatibility between teacher and student behavior and a sufficient initial student capability [23].

## Objectives and feedback signals

### Forward and reverse KL

Forward KL, $D_{\mathrm{KL}}(p_T\Vert p_S)$, penalizes the student for assigning too little mass wherever the teacher assigns mass. It is often described as *mean-covering*. Reverse KL, $D_{\mathrm{KL}}(p_S\Vert p_T)$, weights discrepancies by student probability and is often described as *mode-seeking*. MiniLLM uses a reverse-KL objective with policy-gradient machinery and student-generated samples [8].

Those slogans are useful but incomplete. Language distributions are high-dimensional, autoregressive, and observed through finite samples. The behavior of a divergence depends on the teacher--student capacity gap, temperature, truncation, and which prefixes are evaluated. Work revisiting KL divergence in LLM distillation shows that the simple mean-covering/mode-seeking story can misdescribe different stages and regions of training [11]. The operational question is not “which KL is universally best?” but “which errors should receive weight for this student, on this rollout distribution?”

GKD makes the divergence configurable and also mixes teacher-generated and student-generated sequences [9]. DistiLLM introduces a skew-KL loss and an adaptive off-policy mechanism intended to retain benefits of student generations at lower cost [12]. DistiLLM-2 goes further by pairing different objectives with teacher and student data, increasing the likelihood of teacher responses while decreasing that of poor student responses [14]. These methods occupy the space between a clean taxonomy's endpoints.

### Token-level logit feedback

When teacher logits are available, every token position provides a dense learning signal. This is the closest analogue to classical distillation. The student need not wait for a complete answer to learn that its next-token distribution is drifting. For long reasoning sequences, however, storing or recomputing full-vocabulary teacher logits can dominate memory and compute. Vocabulary alignment is also required, or at least a principled mapping between tokenizations.

Token-level distillation is especially attractive when teacher and student share a tokenizer and architecture family. It is less accessible when the teacher is a proprietary API that reveals only text. In that case, top-$k$ log probabilities can be a partial substitute if the API exposes them, but the truncated distribution changes the objective.

### Sequence-level and outcome feedback

A teacher can score the completed response, select the better of two attempts, or return a critique. This works with black-box teachers and aligns naturally with tasks whose correctness is evaluated at the sequence level. The cost is sparse credit assignment: a scalar score does not identify which token caused failure.

Outcome-supervised on-policy distillation begins to resemble reinforcement learning. The difference is primarily the source and density of the learning signal. In RL, a reward may come from an environment, verifier, or reward model. In distillation, the supervisory target represents teacher behavior or judgment. In practice, hybrid systems blur the line: teacher logits provide dense shaping while a verifier supplies outcome correctness.

### Self-play and teacher-free variants

Recent surveys include self-distillation and self-play under a broader on-policy umbrella [22]. Here, a model or a stronger snapshot critiques, revises, or contrasts the current policy's outputs. This can reduce dependence on a permanently served external teacher, but it changes the guarantee: the system is no longer necessarily importing capability from a stronger model. Its progress depends on selection, verification, diversity, or asymmetry between roles.

It is therefore useful to record three labels for any method:

| Axis | Main possibilities | Question to ask |
|---|---|---|
| Rollout source | Teacher, dataset, student, mixture | Who generated the contexts? |
| Teacher access | Full logits, partial logits, text API, no external teacher | What feedback is actually observable? |
| Loss granularity | Token, span, trajectory, outcome, hybrid | Where is credit assigned? |

These labels are more informative than calling a system simply “distillation” or “RL.”

## Canonical algorithm families

### ImitKD and dataset aggregation

ImitKD follows the interactive imitation-learning template. It samples translations or summaries from the student, asks the teacher for token-level guidance on student-generated prefixes, and trains on the resulting state distribution [7]. Its conceptual importance exceeds its age: it demonstrates that the key object is not merely a teacher output but an **expert label at a learner-visited state**.

In a production implementation, examples need not be discarded after one update. A replay buffer can retain student states and teacher labels. This makes the method only approximately on-policy, but reduces repeated teacher computation. Sampling can favor recent data while keeping rare historical failures, much as experience replay balances freshness and coverage.

### Generalized Knowledge Distillation

GKD is the central modern reference [9]. It has two degrees of freedom. First, it can mix teacher-generated and student-generated sequences. Second, it can choose among divergences rather than committing to a single KL direction. This makes GKD a framework rather than a single fixed loss.

The practical lesson is that on-policy data are not an all-or-nothing commitment. Let $\lambda$ be the fraction of student-generated trajectories. Early training can use small $\lambda$ for stability, then increase it as the student improves. Alternatively, $\lambda$ can rise when an evaluation detects a train--inference gap and fall when rollouts collapse or teacher cost spikes.

GKD was evaluated on summarization, translation, arithmetic reasoning, and task-agnostic instruction tuning. It also connects naturally to RL fine-tuning because both consume model-generated trajectories. A single rollout service can attach teacher distributions, verifier rewards, and safety labels to the same sample.

### MiniLLM

MiniLLM starts from reverse KL and derives an optimization method suitable for generative LLMs [8]. Since the expectation in reverse KL is under the student, student-generated sequences arise naturally. The method uses policy-gradient techniques and introduces stabilization mechanisms for long sequences. Its results across models from 120M to 13B parameters helped establish on-policy distillation as a serious LLM-compression approach rather than only an imitation-learning curiosity.

MiniLLM also illustrates a terminology trap. Calling reverse KL “on-policy” skips a step. Reverse KL can be estimated from student samples, making an on-policy estimator natural, but the divergence and the sampling distribution remain logically distinct.

### Adaptive and contrastive hybrids

DistiLLM explicitly targets the cost of recent on-policy approaches. Its adaptive off-policy sampling reuses student-generated data and its skew-KL objective interpolates distributional behavior [12]. DistiLLM-2 uses teacher and student responses contrastively and reports applications beyond instruction following, including code, preference alignment, and vision--language settings [14]. Direct Preference Knowledge Distillation similarly reframes transfer through preferences rather than only token imitation [13].

These systems matter because “pure” on-policy training is an expensive limiting case. A useful engineering system usually mixes fresh rollouts, replayed student failures, high-quality teacher traces, and task data. The research problem becomes allocation: which source should receive the next unit of compute?

## Where on-policy distillation is used now

### Reasoning models

Reasoning is the most visible current application. Long chains of thought magnify distribution shift: one mistaken algebraic manipulation or unsupported premise changes every later context. A student trained only on clean teacher solutions may imitate their surface form without learning how to recover from its own mistakes. On-policy feedback can query the teacher precisely at the student's flawed prefixes.

This does not make off-policy reasoning distillation obsolete. DeepSeek-R1's distilled models are a prominent demonstration that teacher-generated reasoning traces can transfer substantial capability [16]. Qwen3 likewise describes distillation from flagship models as part of producing efficient smaller models [17]. These results provide a strong initialization and broad coverage. The on-policy opportunity is subsequent correction and refinement, particularly when the smaller model's reasoning style differs from the teacher's.

Thinking Machines Lab reported experiments framing on-policy distillation as combining the error relevance of RL with the dense signal of supervised learning, including math reasoning and an internal assistant [19]. That is a useful practitioner account, but it should be read alongside controlled papers: organizational workloads, teacher access, and serving infrastructure can make results difficult to reproduce exactly.

### Instruction following and assistants

Instruction tuning contains many superficially acceptable responses and relatively few uniquely correct token sequences. Full-distribution feedback can teach the student which alternatives the teacher considers reasonable without forcing exact imitation of one sampled answer. GKD and the DistiLLM family evaluate instruction-following settings [9,12,14]. On-policy collection further targets habits specific to the student: verbosity, refusal errors, formatting failures, or loss of constraints late in a response.

For an internal assistant, prompts also evolve with deployment. Fresh on-policy sampling can function as a moving diagnostic set. The system observes not only what users ask, but where the current student fails on those requests. Privacy and governance are central: production prompts should not automatically become teacher-labeled training data without consent, filtering, retention limits, and auditability.

### Code generation and agents

Code provides unusually strong outcome signals from compilers and tests. A teacher can still supply dense token guidance, while execution identifies whether a trajectory works. DistiLLM-2 reports code-generation experiments [14]. For tool-using agents, the “token” state expands to observations, tool calls, and environment transitions. The DAgger analogy becomes literal: a weak agent visits bad states, and an expert labels the recovery action.

Agent distillation remains less mature than token-level language distillation. The student's action can alter external state; unsafe exploratory actions cannot simply be allowed because they are on-policy. Sandboxed environments, reversible tools, action constraints, and counterfactual teacher labeling are necessary. Sequence length and branching also make naive per-token teacher inference prohibitive.

### Online speculative decoding

Speculative decoding uses a small draft model to propose tokens that a larger target model verifies. The draft model's value depends on matching the target specifically on contexts produced by the live draft--target system. Online Speculative Decoding updates the draft using deployment traffic and discusses forward- and reverse-KL choices [18]. This is a natural on-policy distillation setting: better agreement directly increases the number of accepted draft tokens and therefore serving speed.

It also reveals that the optimal divergence is workload-dependent. A public discussion by Hu notes that the best KL direction varied with workload and teacher--student capacity gap [20]. The post is useful commentary on an empirical paper, not a replacement for it.

### Diffusion language models and new frontiers

The 2026 literature extends on-policy ideas beyond left-to-right generation. OPTD studies on-policy transition distillation for few-step diffusion language models [24], while dOPSD explores on-policy self-distillation in the same broad model family [25]. These are frontier results, not yet settled recipes. Their importance is conceptual: once the student's inference process induces a state distribution different from the teacher's training paths, matching behavior on student-visited states becomes relevant even when generation is not autoregressive.

## A practical training recipe

An effective project should begin with the constraint that motivates distillation: latency, memory, throughput, privacy, or specialized behavior. “Make the student like the teacher” is too vague to guide data allocation.

### Phase 1: establish an off-policy base

Collect a broad, filtered set of teacher demonstrations and task data. Train the student with supervised learning or token-level KD if logits are available. Measure both task quality and behavioral compatibility: answer length, formatting, tool vocabulary, language coverage, and the probability that the student can produce a coherent attempt.

This phase is not merely a convenience. On-policy feedback is most useful when the student reaches states from which a correction is meaningful. If the student emits noise, teacher logits may train local token choices without conveying the long-range structure of a solution.

### Phase 2: instrument student rollouts

Sample from the exact decoding configurations used at inference, including temperature, tool constraints, and maximum length. Store the model version, prompt version, random seed, tokenization, and decoding parameters. Without this provenance, a rollout cannot be reliably reproduced or assigned to the correct policy.

Stratify prompts by domain and difficulty. Uniformly resampling a broad prompt pool can waste teacher compute on cases the student already solves. Uncertainty, disagreement, verifier failure, and novelty are useful prioritization signals, but each can bias the data. A balanced budget should preserve representative traffic as well as hard cases.

### Phase 3: attach the richest affordable feedback

Use full teacher logits when teacher and student are white-box, tokenizer-compatible, and the vocabulary cost is acceptable. Use top-$k$ logits when bandwidth dominates, while recording the residual mass. Use sequence critiques or pairwise preferences for black-box teachers. Add executable verifiers for math, code, structured output, and tool use whenever possible.

Teacher feedback is not automatically truth. Sample teacher reliability, detect contradictory judgments, and retain independent evaluations. The “teacher hacking” concern raised by Ramé---a student exploiting peculiarities of a teacher rather than acquiring the intended behavior---is a useful warning [21]. Diversity of teachers, prompts, and verifiers can reduce this risk, but does not eliminate it.

### Phase 4: mix data and control policy lag

Let the training batch contain a mixture of static teacher data, recent student rollouts, and replayed failures. Track **policy lag**: the number of optimizer steps or model versions between rollout generation and use. A sample from an old student is still valuable, but it is not strictly on-policy for the current model.

A practical controller can adjust the fresh-rollout fraction based on three measurements:

| Signal | Interpretation | Possible response |
|---|---|---|
| Rising teacher--student divergence on current rollouts | Student is visiting poorly modeled states | Increase fresh on-policy sampling |
| Low-quality or incoherent student trajectories | Feedback has weak long-horizon value | Return weight to teacher demonstrations |
| Stable task quality but high teacher cost | Marginal correction is expensive | Increase replay, reduce refresh cadence |
| Strong verifier score but declining diversity | Student may be collapsing to narrow modes | Adjust divergence, temperature, or data mixture |
| Good offline metrics but poor deployment traces | Evaluation misses the induced state distribution | Add representative on-policy traffic slices |

### Phase 5: evaluate the right counterfactuals

At minimum, compare: off-policy only; on-policy only from the same initialization; a hybrid; and a compute-matched baseline. Count teacher FLOPs or API cost, student rollout cost, and wall-clock pipeline overhead. Otherwise an apparent algorithmic gain may simply be more inference.

Evaluate on held-out prompts with the student's deployment decoder. Report correctness and preference metrics, but also calibration, response length, diversity, safety, and recovery after injected mistakes. For speculative decoding, measure accepted tokens and end-to-end latency. For agents, measure task success, unsafe actions, recovery rate, and environment cost.

## Failure modes and limits

### The teacher--student compatibility problem

If the teacher's preferred trajectories are far outside the student's representational or stylistic capacity, dense feedback can still be unhelpful. A small student may be unable to assign meaningful probability to the teacher's modes, while reverse-KL-like objectives can overconcentrate on a narrow subset. Recent work argues that on-policy gains depend on compatible reasoning patterns and teacher capability that is genuinely novel to the student [23].

### Confirmation and mode collapse

Student rollouts determine which states receive labels. A narrow student may never visit valuable alternatives, so on-policy training can reinforce its limited support. Teacher demonstrations, exploration, higher-temperature sampling, and explicit diversity terms counter this. The correct mixture is empirical.

### Cost and systems bottlenecks

White-box teacher logits over long trajectories are expensive to compute, transfer, and store. Generating sequences serially lowers accelerator utilization. Teacher and student may require different parallelism schemes, while synchronous coupling leaves one waiting for the other. Caching, top-$k$ logits, asynchronous queues, chunked generation, and replay all help, but each moves the algorithm away from a pristine on-policy ideal.

### Stale feedback

When student weights change rapidly, queued rollouts describe yesterday's policy. Importance weighting is theoretically tempting but difficult for long language-model trajectories because likelihood ratios can explode or vanish. In practice, bounded queues, version-aware sampling, and short refresh intervals are often more robust.

### Evaluation leakage and weak teachers

Repeatedly selecting prompts by benchmark failure can overfit the benchmark even if its labels are never directly trained on. A teacher can also be wrong, stylistically biased, or vulnerable to prompt injection embedded in student outputs. Treat teacher responses as untrusted model output: isolate instructions, validate structured feedback, and keep a genuinely untouched evaluation set.

## How to read claims in this area

On-policy distillation sits at the intersection of supervised learning, imitation learning, reinforcement learning, and systems optimization. Comparisons are easy to misread. A careful reader should ask:

1. Were trajectories generated by the current student, a lagged student, the teacher, or a fixed dataset?
2. Did the teacher expose full logits, sampled text, rankings, critiques, or rewards?
3. Was the baseline matched for teacher calls and total compute?
4. Did on-policy training start from the same off-policy checkpoint?
5. Were results measured with the actual deployment decoder?
6. Is the claimed gain task quality, latency, accepted speculative tokens, or some mixture?
7. Does “self-distillation” import new information through a verifier, search, or stronger snapshot?

The 2026 survey by Song and Zheng offers a useful taxonomy by feedback signal, teacher access, and loss granularity [22]. As a current survey, it maps a rapidly changing area; individual deployment claims should still be traced to their original reports.

## Open research questions

**Scaling laws for teacher feedback.** How should gains scale with teacher size, student size, rollout count, and feedback density? The relevant budget is not only training tokens but teacher inference and pipeline latency.

**Active trajectory selection.** The most useful student mistakes are neither trivial nor hopeless. Selecting prompts and prefixes by expected learning value could make on-policy distillation substantially cheaper, but naive uncertainty sampling can distort coverage.

**Cross-tokenizer and black-box distillation.** Many valuable teachers expose only text and use unknown tokenization. Sequence feedback is portable but sparse. Better span-level credit assignment and constrained counterfactual querying could close the gap.

**Agent-level distillation.** Distilling a complete policy over tool calls, memory, and environment interaction requires safe exploration and causal credit assignment. Static chat transcripts omit the counterfactual states where an agent most needs expert recovery behavior.

**Robustness to imperfect teachers.** A student should learn capabilities without inheriting every bias, exploit, and brittle preference. Multi-teacher disagreement, calibrated abstention, and verifier-grounded feedback are promising but underdeveloped.

**Unifying distillation and RL.** Dense teacher distributions and sparse environment rewards answer different questions. A principled hybrid could use the teacher for local behavioral guidance while preserving optimization toward verifiable outcomes. The challenge is preventing the teacher objective from suppressing discoveries that exceed the teacher.

## Conclusion

On-policy distillation is best understood as **expert feedback on the student's own state distribution**. Its roots run from classical soft-target distillation through exposure-bias research and interactive imitation learning. Its modern form is especially relevant to long autoregressive reasoning, instruction following, code, speculative decoding, and agents, where small early deviations create contexts absent from polished teacher demonstrations.

The distinction from off-policy distillation is not a contest with a universal winner. Off-policy data are cheap to reuse, easy to audit, and excellent for cold starts and broad transfer. On-policy data are expensive but targeted: they teach recovery from the current student's actual mistakes. The strongest practical systems are therefore likely to be hybrids that begin with teacher demonstrations, add fresh student trajectories, retain valuable failures in replay, and allocate teacher compute where it changes behavior most.

Finally, “on-policy” should never be used as shorthand for a particular KL direction or for reinforcement learning in general. A precise description states the rollout source, teacher access, feedback granularity, objective, and policy lag. Once those pieces are separated, the field becomes easier to reason about---and easier to engineer.

## References {.unnumbered}

```{=latex}
\markboth{References}{References}
```

1. Cristian Buciluǎ, Rich Caruana, and Alexandru Niculescu-Mizil. “Model Compression.” *KDD*, 2006. <https://doi.org/10.1145/1150402.1150464>
2. Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. “Distilling the Knowledge in a Neural Network.” *NIPS Deep Learning Workshop*, 2015. <https://arxiv.org/abs/1503.02531>
3. Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. “Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.” *NeurIPS*, 2015. <https://proceedings.neurips.cc/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html>
4. Alex M. Lamb et al. “Professor Forcing: A New Algorithm for Training Recurrent Networks.” *NeurIPS*, 2016. <https://proceedings.neurips.cc/paper/2016/hash/16026d60ff9b54410b3435b403afd226-Abstract.html>
5. Stéphane Ross, Geoffrey Gordon, and Drew Bagnell. “A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.” *AISTATS*, 2011. <https://proceedings.mlr.press/v15/ross11a.html>
6. Yoon Kim and Alexander M. Rush. “Sequence-Level Knowledge Distillation.” *EMNLP*, 2016. <https://aclanthology.org/D16-1139/>
7. Alexander Lin et al. “Autoregressive Knowledge Distillation through Imitation Learning.” *EMNLP*, 2020. <https://aclanthology.org/2020.emnlp-main.494/>
8. Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. “MiniLLM: Knowledge Distillation of Large Language Models.” *ICLR*, 2024. <https://arxiv.org/abs/2306.08543>
9. Rishabh Agarwal et al. “On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes.” *ICLR*, 2024. <https://arxiv.org/abs/2306.13649>
10. Cheng-Yu Hsieh et al. “Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes.” *Findings of ACL*, 2023. <https://aclanthology.org/2023.findings-acl.507/>
11. Taiqiang Wu et al. “Rethinking Kullback--Leibler Divergence in Knowledge Distillation for Large Language Models.” 2024. <https://arxiv.org/abs/2404.02657>
12. Jongwoo Ko, Sungnyun Kim, Tianyi Chen, and Se-Young Yun. “DistiLLM: Towards Streamlined Distillation for Large Language Models.” 2024. <https://arxiv.org/abs/2402.03898>
13. Yong Lin et al. “Direct Preference Knowledge Distillation for Large Language Models.” 2024. <https://arxiv.org/abs/2406.19774>
14. Jongwoo Ko et al. “DistiLLM-2: A Contrastive Approach Boosts the Distillation of LLMs.” 2025. <https://arxiv.org/abs/2503.07067>
15. Eric Zelikman et al. “STaR: Bootstrapping Reasoning With Reasoning.” *NeurIPS*, 2022. <https://arxiv.org/abs/2203.14465>
16. DeepSeek-AI et al. “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.” 2025. <https://arxiv.org/abs/2501.12948>
17. An Yang et al. “Qwen3 Technical Report.” 2025. <https://arxiv.org/abs/2505.09388>
18. Xiaoxuan Liu et al. “Online Speculative Decoding.” 2023. <https://arxiv.org/abs/2310.07177>
19. Thinking Machines Lab. “On-Policy Distillation.” 2025. <https://thinkingmachines.ai/blog/on-policy-distillation/>
20. Lanxiang Hu. Commentary on divergence choice in online speculative decoding and on-policy distillation. X, 27 October 2025. <https://x.com/Lanxiang_Hu/status/1982982868159869433>
21. Alexandre Ramé. Thread on teacher hacking and diversity in knowledge distillation. X, 7 February 2025. <https://x.com/ramealexandre/status/1887947781215846409>
22. Mingyang Song and Mao Zheng. “A Survey of On-Policy Distillation for Large Language Models.” 2026. <https://arxiv.org/abs/2604.00626>
23. Zhuofan Li et al. “Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mechanism, and Recipe.” 2026. <https://arxiv.org/abs/2604.13016>
24. “OPTD: On-Policy Transition Distillation for Few-Step Diffusion Language Models.” 2026. <https://arxiv.org/abs/2608.02942>
25. “dOPSD: On-Policy Self-Distillation for Diffusion Language Models.” 2026. <https://arxiv.org/abs/2607.04428>
26. Google Research. “Distilling Step-by-Step: Outperforming Larger Language Models with Less Training.” 2023. <https://research.google/blog/distilling-step-by-step-outperforming-larger-language-models-with-less-training-data-and-smaller-model-sizes/>
