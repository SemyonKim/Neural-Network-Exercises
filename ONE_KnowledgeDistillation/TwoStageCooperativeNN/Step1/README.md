# Step1 — Two-stage training of a cooperative neural network

Based on: Xu et al., *Knowledge Distillation On the Fly Native Ensemble (ONE)*, NeurIPS 2018.  
Original implementation: [Lan1991Xu/ONE_NeurIPS2018](https://github.com/Lan1991Xu/ONE_NeurIPS2018)

## Background
Deep neural networks achieve strong performance but require resource-intensive training. Knowledge distillation reduces training cost by transferring knowledge from a teacher to a student model. The ONE method eliminates the need for a pre-trained teacher by constructing auxiliary branches and forming an ensemble teacher on the fly.

## Objectives
1. Validate ONE results for RESNET32 on CIFAR100.  
2. Prove feasibility of scaling from 3 to 5 branches without loss of classification quality.  
3. Implement parameter transfer procedure from RESNET32 to extended network.  
4. Extend evaluation metrics (top‑1 and top‑5) to strengthen proof.

## Evolution of Experiments

### 🔹 Original Scripts
- **Baseline:** plain SGD, cross‑entropy, top‑1 accuracy only.  
- **ONE:** 3 branches + ensemble, cross‑entropy + KL distillation, top‑1 accuracy per branch.

### 🔹 First Modified Pair (“After LR”)
- Introduced **GradientRatioScheduler** and **geometric LR updates**.  
- Extended ONE architecture to **5 branches + ensemble**.  
- Loss extended to all branches.  
- Logging expanded with epoch, time, LR.  
- **Proof in practice:** ensemble accuracy ≥ smaller branch network, confirming p₂ ≥ p₁.

### 🔹 Last Modified Pair (“With top‑5 before LR”)
- Added **top‑5 accuracy tracking** per branch and ensemble.  
- Logging expanded to include both top‑1 and top‑5 metrics.  
- LR schedule adjusted (gamma = 0.333).  
- **Proof in practice:** ensemble top‑5 accuracy consistently matched/exceeded smaller branch networks, reinforcing corollaries.

## Assertion & Proof
- **Assertion:** A 5-branch network (S2) can be initialized from a 3-branch network (S1) such that accuracy p₂ ≥ p₁.  
- **Proof:** By contradiction, zeroing outputs of two branches in S2 reduces it to S1, contradicting p₂ < p₁.  
- **Corollaries:**  
  1. S2 classifies no worse than S1.  
  2. S2 can always be extended from S1 without degrading performance.

## Implementation
- `parameter_transfer.py` → prepares and transfers parameters from pre-trained model.  
- `cifar_baseline_original.py` / `cifar_one_original.py` → original scripts.  
- `cifar_baseline_afterLR.py` / `cifar_one_afterLR.py` → first modified pair.  
- `cifar_baseline_top5.py` / `cifar_one_top5.py` → last modified pair.  

## Practical Significance
Thanks to the two-stage learning method, as the number of branches increases, training can be faster and accuracy preserved or improved. Extended metrics (top‑5) further validate robustness of cooperative distillation.
