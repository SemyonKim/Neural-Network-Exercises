# Knowledge Distillation with ONE (On-the-Fly Native Ensemble)

This folder contains my experiments based on the paper:

- Xu et al., *Knowledge Distillation On the Fly Native Ensemble (ONE)*, NeurIPS 2018. [PDF](https://arxiv.org/pdf/1806.04606)  
- Original implementation: [Lan1991Xu/ONE_NeurIPS2018](https://github.com/Lan1991Xu/ONE_NeurIPS2018)

---

## 📖 Overview

The goal of this project is to explore **cooperative neural networks** using the ONE framework.  
We extend the baseline ResNet into multi‑branch cooperative architectures, test different merging and expansion strategies, and analyze how ensemble learning improves generalization and stability.

This folder contains two major stages:

- **Step1 — Foundations**  
  - Introduces the cooperative learning framework.  
  - Implements and documents five subgroups of experiments (baseline, 3‑branch, 5‑branch, voting, transfer 3→5).  
  - Establishes the theoretical basis for cooperative neural networks.

- **Step2 — Cooperative Adaptations**  
  - Builds on Step1 with more advanced experiments.  
  - Implements and documents four subgroups (merge baseline+branch models, merge three one‑branch models, expand single branch into 3, single branch inside cooperative model).  
  - Tests merging strategies and expansion from single‑branch to multi‑branch setups.

---

## 📂 Contents
- `TwoStageCooperativeNN/Step1/` → README and code for Step1 experiments
- `TwoStageCooperativeNN/Step2/` → README and code for Step2 experiments
  - `utils/` → helper functions (logging, accuracy, branch replacement, etc.)
  - `CooperativeNeuralNetworks.ipynb` → Colab notebook where experiments are executed

---

## 📝 Notes

- All code is refactored for clarity and reproducibility.  
- Checkpoints are organized under `checkpoints/cifar10/` and `checkpoints/cifar100/`.  
- Logger automatically writes headers and results — no manual header cells are needed.  
> ⚠️ **Important:** This repository does **not** contain any pretrained weights.  
  Each viewer must **train and generate their own checkpoints** before running cooperative experiments.

---

## 📚 References & Attribution

- **Original Authors:** Xu Lan, Xiatian Zhu and Shaogang Gong  
  *Knowledge Distillation On the Fly Native Ensemble (ONE).* NeurIPS 2018.  
  [Paper PDF](https://arxiv.org/pdf/1806.04606)  
  [Original GitHub Repository](https://github.com/Lan1991Xu/ONE_NeurIPS2018)

---

## License
This project is released under the MIT License. See [LICENSE](https://github.com/SemyonKim/Neural-Network-Exercises/blob/main/LICENSE) for details.

- The original codebase belongs to [Lan1991Xu](https://github.com/Lan1991Xu).  
- This folder contains **my own experiments and modifications**.  
- Please refer to the original repository for the full implementation details.
