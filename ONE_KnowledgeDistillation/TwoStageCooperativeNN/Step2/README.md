# Step2 — Cooperative Neural Networks (ONE_NeurIPS2018 Adaptations)

This folder continues our exploration of cooperative neural networks, based on the framework introduced in:

- **Reference Article:** **[Xu et al. *Knowledge Distillation On the Fly Native Ensemble* (ONE) NeurIPS 2018.](https://arxiv.org/pdf/1806.04606)**  
- **Original GitHub Repository:** [Lan1991Xu/ONE_NeurIPS2018](https://github.com/Lan1991Xu/ONE_NeurIPS2018)

All adaptations here are refactored versions of the original code, extended for our experiments.  
We document the **theory of Step2** and the **Part1 notebook (5 subgroups)**, and now the Part2 notebook (4 subgroups).  
Step1’s README already covered the foundation, so we avoid duplication here.

---

## 📖 Theory of Step2

Step2 focuses on **multi-branch cooperative learning**.  
The central idea is to extend the baseline ResNet into multiple parallel branches that share early layers but diverge later, producing diverse predictions.  
A control vector dynamically weights branch outputs, enabling ensemble learning.  
This cooperative setup is expected to improve generalization, stability, and ensemble accuracy compared to single-branch models.

---

## 📂 Part1 — Multi-Branch Models (Completed)

Part1 contains **five subgroups**, each building on the cooperative framework.  
Below we describe **what we did, why we did it, and what outcomes we observed**.

---

### 1. Simple 1‑Branch Model (Baseline)
- **Goal:** Establish a baseline accuracy for ResNet32+ONE with 1 branch on CIFAR100.  
- **Changes:**  
  - Edited `resnet.py` to remove extra branches, leaving only `layer3_1` and `classifier3_1`.  
  - Edited `cifar_one.py` to adapt training loop for single branch outputs.  
- **Training setup:** 160 epochs, CIFAR100 dataset, ResNet32 depth.  
- **Result:** Final test accuracy ≈ **70.12%**.  
- **Why:** Provides a baseline for comparison with multi-branch cooperative models.

---

### 2. Simple 3‑Branch Model
- **Goal:** Evaluate cooperative learning with ResNet32+ONE using 3 branches on CIFAR100.  
- **Changes:**  
  - Edited `resnet.py` to define `layer3_1`, `layer3_2`, `layer3_3` and their classifiers.  
  - Edited `cifar_one.py` (refactored as `train_threebranch.py`) to adapt training/testing loops for three branch outputs plus ensemble.  
- **Training setup:** 10 epochs, CIFAR100 dataset, ResNet32 depth, consistency ramp‑up enabled, resuming from the 1‑branch baseline checkpoint.  
- **Result:**  
  - Branch accuracies: 73.36%, 72.43%, 72.05%  
  - Ensemble accuracy: **74.90%**  
- **Why:** Demonstrates how ensemble predictions provide a cooperative gain over individual branches, slightly boosting accuracy compared to the single‑branch baseline.

---

### 3. Simple 5‑Branch Model
- **Goal:** Evaluate cooperative learning with ResNet32+ONE using 5 branches on CIFAR10.  
- **Changes:**  
  - Edited `resnet.py` to define `layer3_1 … layer3_5` and their classifiers, with ensemble weighting.  
  - Edited `cifar_one.py` to adapt training/testing loops for five branch outputs plus ensemble.  
- **Training setup:** 300 epochs, CIFAR10 dataset, ResNet32 depth, consistency ramp‑up enabled, resuming from 1‑branch pretrained weights copied into all 5 branches.  
- **Result:** Ensemble accuracy improves over individual branches, showing stronger cooperative gains compared to 1‑branch and 3‑branch models.  
- **Why:** Tests scalability of cooperative learning to larger ensembles.

---

### 4. 5‑Branch Model with Voting
- **Goal:** Analyze cooperative learning dynamics with voting ensembles.  
- **Architecture:** Five parallel branches trained simultaneously, each producing logits.  
- **Special feature:** In addition to the standard ensemble (weighted sum of all branches), the model computes multiple voting combinations — subsets of branches (pairs, triples, quadruples, singles).  
- **Purpose:** To evaluate cooperative performance under different voting scenarios and test whether smaller ensembles can rival or complement the full 5‑branch ensemble.  
- **Outcome:** Provides deeper insight into cooperative learning dynamics, showing how ensemble diversity and voting strategies affect final performance compared to individual branches or the full ensemble.

---

### 5. 3‑Branch Model → 5‑Branch Model
- **Goal:** Test transfer learning from a smaller cooperative setup (3 branches) to a larger one (5 branches).  
- **Architecture:** ResNet‑32 with five parallel branches plus ensemble.  
- **Special feature:** Initialization copies weights from a previously trained 3‑branch model, expanding them into five branches.  
- **Two resume options:**  
  - `--resume` → loads the checkpoint from the 3‑branch model (source weights).  
  - `--resume0` → loads the checkpoint from the 5‑branch model (optimizer state).  
- **Purpose:** To test whether knowledge learned in a smaller cooperative setup can be effectively reused and scaled.  
- **Outcome:** Pretrained cooperative weights accelerate convergence and improve ensemble accuracy compared to training a 5‑branch model from scratch.

---

## 📂 Part2 — Cooperative Model Merging & Expansion (Completed)  
Part2 explores **how different pretrained models can be merged or expanded into cooperative architectures**.
We designed **four subgroups**, each testing a different merging or expansion strategy.

---

### 1. 3s3 → 3 : Combine Baseline+One‑Branch Models into One Cooperative Model
- **Goal:** Merge three pretrained models (each baseline+one branch) into a single cooperative 3‑branch ResNet.
- **Changes:**
	- Refactored resnet_threecomb.py to define three parallel branches with classifiers.
	- Refactored train_threecomb.py to load three separate checkpoints and combine them into one cooperative model.
- **Training setup:** 75 epochs, CIFAR100 dataset, ResNet32 depth, consistency ramp‑up enabled.
- **Result:** Cooperative ensemble accuracy exceeded individual branch accuracies, showing that merging multiple baseline+branch models into one cooperative framework improves performance.
- **Why:** Tests whether multiple partially cooperative models can be unified into a stronger ensemble.

---

### 2. 3s3 → 3 : Merge Three One‑Branch Models into One Cooperative Model
- **Goal:** Merge three independently trained one‑branch ResNet models into a single cooperative 3‑branch ResNet.
- **Changes:**
	- Refactored train_threes_into_three.py to load three one‑branch checkpoints (resume1, resume2, resume3) and merge them into one cooperative model (resume).
	- Branch IDs specify which branch slot each pretrained model occupies.
- **Training setup:** 75 epochs, CIFAR100 dataset, ResNet32 depth.
- **Result:** Cooperative ensemble accuracy improved compared to individual one‑branch models, demonstrating that independent single‑branch models can be merged into a cooperative ensemble.
- **Why:** Tests whether cooperative learning benefits can be achieved by merging separately trained one‑branch models.

---

### 3. 1 → 3 : Expand Single‑Branch Model into Cooperative 3‑Branch Model (Variant A)
- **Goal:** Expand a pretrained single‑branch ResNet into a cooperative 3‑branch ResNet.
- **Changes:**
	- Refactored train_one_into_three.py to load a single‑branch checkpoint (resume) and expand it into three branches inside the cooperative model (resume0).
	- Control box and classifiers duplicated/extended to distribute the single branch’s knowledge across all three branches.
	- Training setup: 120 epochs, CIFAR100 dataset, ResNet32 depth.
- **Result:** Ensemble accuracy improved compared to the single branch baseline, showing that expansion into cooperative branches provides ensemble benefits.
- **Why:** Tests whether a strong single‑branch model can be effectively expanded into a cooperative ensemble.

---

### 4. 1 → 3 : Single Branch Inside Cooperative Model (Variant B)
- **Goal:** Evaluate performance when only one branch is active inside the cooperative 3‑branch ResNet.
- **Changes:**
	- Refactored train_b_one_into_three.py to load a single‑branch checkpoint (resume) and run it inside the cooperative 3‑branch architecture (resume0), leaving other branches inactive.
	- Training setup: 200 epochs, CIFAR10 dataset, ResNet32 depth.
- **Result:** Accuracy matched the single‑branch baseline, confirming that without cooperative expansion, ensemble benefits are not realized.
- **Why:** Provides a control experiment to isolate the effect of cooperative expansion versus simply embedding a single branch inside the cooperative framework.

---

## ✅ Current Status
- **Part1 completed:** All five subgroups implemented and documented.  
- **Part2 completed:** All four subgroups implemented and documented.  

---

## 📝 Notes
- All code is refactored for clarity and reproducibility.  
- Checkpoints are organized under `checkpoints/cifar10/` and `checkpoints/cifar100/`.  
- Logger automatically writes headers and results — no manual header cells are needed.
> ⚠️ Important: This repository does not contain any pretrained weights.
Each viewer must train and generate their own checkpoints before running the cooperative experiments.  
