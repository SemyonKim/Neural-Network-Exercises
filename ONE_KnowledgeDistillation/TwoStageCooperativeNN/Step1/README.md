# Step1 — Two-stage training of a cooperative neural network
- **Reference Article:** **[Xu et al. *Knowledge Distillation On the Fly Native Ensemble* (ONE) NeurIPS 2018.](https://arxiv.org/pdf/1806.04606)**  
- **Original GitHub Repository:** [Lan1991Xu/ONE_NeurIPS2018](https://github.com/Lan1991Xu/ONE_NeurIPS2018)

---

## 📖 Background
Deep neural networks achieve strong performance but require resource-intensive training. Knowledge distillation reduces training cost by transferring knowledge from a teacher to a student model. The ONE method eliminates the need for a pre-trained teacher by constructing auxiliary branches and forming an ensemble teacher on the fly.

My theoretical contribution:
- **Assertion:** A larger cooperative network (e.g., 5 branches) can be initialized from a smaller one (e.g., 3 branches) such that accuracy $`𝑝_2 ≥ 𝑝_1`$.
- Corollaries:
  1. A 5‑branch network classifies no worse than a 3‑branch network.
  2. A larger network can always be extended from a smaller one without degrading performance.
 
---

## 🧩 Evolution of Scripts
- **Original Pair (Baseline + ONE)**
  - **Baseline** (cifar_baseline_original.py):
    - Standard CIFAR‑10/100 training loop.
    - Optimizer: plain SGD with fixed schedule.
    - Loss: cross‑entropy only.
    - Metrics: top‑1 accuracy only.
    - Logging: train/test loss + accuracy.
  - **ONE** (cifar_one_original.py):
    - Implements Xu et al. ONE method.
    - Architecture: 3 branches + ensemble teacher.
    - Loss: cross‑entropy per branch + KL distillation loss from ensemble teacher.
    - Metrics: top‑1 accuracy per branch + ensemble.
    - Logging: branch accuracies + ensemble accuracy.

- **First Modified Pair (“After LR”)**
  - **Baseline** (cifar_baseline_afterLR.py):
    - Added GradientRatioScheduler for parameter‑specific learning rates.
    - Introduced geometric LR updates inside training loop.
    - Added flags: --geo-lr, --deterministic, --save-checkpoint-model.
    - Expanded logging: epoch, time, LR.
    - Still tracked top‑1 accuracy only.
  - **ONE** (cifar_one_afterLR.py):
    - Extended architecture: 3 → 5 branches + ensemble teacher.
    - Loss: cross‑entropy + KL distillation across all 5 branches, ensemble = branch 6.
    - Optimizer replaced with GradientRatioScheduler.
    - Added geometric LR updates (“After LR”).
    - Logging expanded: per‑branch accuracies, ensemble accuracy, epoch, time, LR.

> **Integration with Theory:**  
This implements the scaling proof (S1 → S2, 3 → 5 branches). The custom LR scheduler stabilizes training of the larger cooperative network. Results confirmed ensemble accuracy ≥ smaller branch network, validating $`𝑝_2 ≥ 𝑝_1`$.

- **Last Modified Pair (“With Top‑5 Before LR”)**
  - **Baseline** (cifar_baseline_top5.py):
    - Added top‑5 accuracy tracking alongside top‑1.
    - Logging now includes both metrics before LR adjustments.
    - Retained standard optimizer (SGD) but gamma changed (0.333).
    - Extended evaluation metrics to validate broader classification performance.
  - **ONE** (cifar_one_top5.py):
    - Architecture: still 5 branches + ensemble.
    - Loss: cross‑entropy + KL distillation across all branches.
    - Metrics: top‑1 and top‑5 accuracy per branch + ensemble.
    - Logging: detailed per‑branch top‑1/top‑5, ensemble top‑1/top‑5.
    - LR adjustment remains standard schedule (not GradientRatioScheduler here).

> **Integration with Theory:**  
By tracking top‑5 accuracies, I validated that scaling branches preserves not only strict top‑1 accuracy but also broader classification quality. This strengthens the corollaries: larger cooperative networks classify no worse than smaller ones.


---

## 📊 Comparison Table
|Aspect |	Original Scripts (Baseline + ONE) |	First Modified Pair (“After LR”)|
| :--- | :---: | :--- |
| Optimizer	| Plain SGD with fixed schedule |	GradientRatioScheduler with parameter‑specific LR + geometric updates |
| Learning Rate Control | Static schedule (adjust_learning_rate) | Dynamic updates inside training loop (After LR) |
| Architecture | Baseline: single model ONE: 3 branches + ensemble | Extended to 5 branches + ensemble teacher |
| Loss Functions | Baseline: cross‑entropy only ONE: cross‑entropy + KL distillation (3 branches) | Cross‑entropy + KL distillation across 5 branches, ensemble = branch 6 |
| Metrics Tracked | Baseline: top‑1 only ONE: top‑1 per branch + ensemble | Top‑1 per branch + ensemble, epoch, time, LR |
| Logging | Train/test loss + accuracy | Expanded: per‑branch accuracies, ensemble, epoch, time, LR |
| Flags Added | None | --geo-lr, --deterministic, --save-checkpoint-model |
| Connection to Theory | Reproduces ONE baseline results | Implements scaling proof (S1 → S2, 3 → 5 branches) with LR stabilization |

---

## 🧠 Practical Proof of Theory
- **Assertion:** A 5‑branch network (S2) can be initialized from a 3‑branch network (S1) such that accuracy $`𝑝_2 ≥ 𝑝_1`$.
- **Proof in Practice:**
  - Extended scripts to 5 branches.
  - Implemented parameter transfer + cooperative distillation.
  - Logged both top‑1 and top‑5 accuracies across branches.
  - Results confirmed ensemble accuracy ≥ smaller branch network, validating theoretical claim.

---

## 📂 Folder Structure
- `parameter_transfer.py` → prepares and transfers parameters from pre-trained model.  
- `cifar_baseline_original.py` / `cifar_one_original.py` → original scripts.  
- `cifar_baseline_afterLR.py` / `cifar_one_afterLR.py` → first modified pair.  
- `cifar_baseline_top5.py` / `cifar_one_top5.py` → last modified pair.

---

## Practical Significance
Thanks to the two-stage learning method, as the number of branches increases, training can be faster and accuracy preserved or improved. Extended metrics (top‑5) further validate robustness of cooperative distillation.
