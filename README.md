# LSTM-MLP-Knowledge-Distillation-for-Indoor-Localization
## Overview
This repository implements a **dual-model framework** for Wi-Fi indoor localization that combines a **Long Short-Term Memory (LSTM)** network and a **Multilayer Perceptron (MLP)** through **knowledge distillation**.  
The method aims to achieve both **high accuracy** (via LSTM) and **low latency** (via MLP) by allowing the MLP to learn from the soft output distributions of the trained LSTM teacher.

<img width="2097" height="1532" alt="image" src="https://github.com/user-attachments/assets/994ee686-e236-4138-b2d5-e0f38c4659a3" />


The LSTM captures temporal dependencies in sequential signal data, while the distilled MLP approximates its predictive behavior using single-scan fingerprints.  
This design provides an adaptable system that performs well in both **sequence-based** and **snapshot-based** localization scenarios.

---

## Motivation
Real-world indoor localization systems often face two challenges:
1. **Temporal variability** – Wi-Fi signals fluctuate over time, requiring models that can capture temporal structure.  
2. **Real-time constraints** – In many applications, only one signal snapshot is available, and quick inference is necessary.

The proposed framework addresses these challenges by training a **temporal LSTM** as a high-accuracy teacher and a **compact MLP** as a distilled student.  
Through **knowledge distillation**, the student inherits the teacher’s understanding of spatial–temporal patterns while remaining lightweight enough for real-time deployment.

---

## Methodology

### 1. LSTM Teacher Model
- **Input:** Sequential RSS samples representing signal strength over time.  
- **Architecture:** One or more LSTM layers followed by dense layers for block-wise classification.  
- **Output:** Probability distribution over spatial blocks, later used as soft targets for the student model.  
- The teacher learns robust spatial–temporal features and serves as a reference model for distillation.

### 2. MLP Student Model
- **Input:** Single-scan RSS fingerprint.  
- **Training:** The MLP is trained with a combination of:
  - **Hard labels:** Ground-truth block IDs.
  - **Soft labels:** Probabilistic outputs (logits) from the LSTM teacher.
- **Loss Function:** Weighted combination of cross-entropy losses from hard and soft targets, controlled by α (weight) and T (temperature).
- **Goal:** Mimic the teacher’s decision boundaries while maintaining high-speed inference.

### 3. Training Process
1. Train the **LSTM teacher** on sequence data until convergence.  
2. Freeze the teacher’s parameters and use its soft outputs as targets.  
3. Train the **MLP student** using both soft and hard labels to balance accuracy and generalization.  
4. Evaluate both models using metrics such as mean localization error (MLE), accuracy, and runtime.

---

## Data and Preprocessing
The framework is evaluated on Wi-Fi fingerprint datasets such as:
- **UJIIndoorLoc**
- **UTSIndoorLoc**

Each fingerprint sample contains RSS values from multiple APs.  
- Missing values are replaced with –100 dBm.  
- Data are grouped into **blocks** to reformulate localization as a classification task.  
- For the LSTM, sequences are generated using sliding windows over consecutive samples within each block.  
- The MLP uses individual fingerprints with distilled supervision.
