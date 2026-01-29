# OptimisationNN

## 📖 Overview
The `OptimisationNN` folder contains a project focused on **synthetic license plate generation and recognition using neural networks**.  
The name *OptimisationNN* reflects the central idea: optimizing neural network accuracy by refining the dataset. Two related problems were solved here:

1. **Synthetic license plate generation**  
   - Plates were generated using the Russian alphabet.  
   - Two datasets were created:
     - A **full alphabet set** including all letters.  
     - A **specialized set** excluding paired letters (e.g., е–ё, и–й) and oversized letters (e.g., ж, ш).  

2. **Neural network training and comparison**  
   - The same CNN model was trained on both datasets.  
   - Results showed that **restricting the dataset to a more specialized alphabet improved recognition accuracy**, demonstrating optimization through dataset specification.

---

## 📂 Structure
- **Plate_Generator/**
  - `generator.py` → Generates synthetic license plate images using the Russian alphabet.  
  - `augmenter.py` → Applies image distortions (rotation, smudging, noise) to simulate real-world conditions.  

- **CNN_model/**
  - `train.py` → Defines and trains a CNN model with custom normalization layers.  
  - `test.py` → Evaluates the trained model and predicts license plate outputs.  

---

## 🎯 Objectives
- Create realistic synthetic license plate datasets for training.  
- Compare CNN performance on a **general dataset** vs. a **specialized dataset**.  
- Demonstrate that **dataset optimization improves neural network accuracy**.  

---

## ⚙️ Dependencies
- Python 3.x  
- Libraries:
  - Core: `numpy`, `pandas`, `matplotlib`, `opencv-python`, `PIL`
  - ML/NN: `torch`, `torchvision`, `scikit-learn`

Install missing packages with:
```bash
pip install -r requirements.txt
```
> Note: A viewer must create their own requirements.txt file.

---

## 🚀 Usage
1. Generate synthetic license plates with `generator.py`.
2. Apply distortions using `augmenter.py` to expand dataset variability.
3. Train the CNN model with `train.py`.
4. Evaluate recognition accuracy with `test.py`.
5. Compare results between the full alphabet dataset and the specialized dataset.

---

## 📌 Notes
- The project demonstrates how data specification can serve as a form of optimization in neural network training.
- Recognition accuracy improved when training on the specialized dataset, validating the approach.
- **Current state**: The combined code in this folder covers about *4/5 of the workable project*. This content is provided for **learning references**, but to run the code successfully the viewer must complete the residual part.

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](https://github.com/SemyonKim/Neural-Network-Exercises/blob/main/LICENSE) file for details.
