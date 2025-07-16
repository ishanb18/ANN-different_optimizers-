# 🧠 ANN Optimizer Comparison on CIFAR-10

This project compares the performance of different optimization algorithms on a simple Artificial Neural Network (ANN) using the CIFAR-10 dataset.

---

## 📁 Files

- `ANN(Different optimizers).py` – Main script that:
  - Loads CIFAR-10 dataset
  - Trains the same ANN architecture using multiple optimizers
  - Visualizes validation loss and accuracy
  - Compares final test accuracy for each optimizer

---

## 🧪 Optimizers Used

- Gradient Descent (GD)
- Stochastic Gradient Descent (SGD)
- Momentum
- RMSprop
- Adam

---

## 📊 Evaluation

- Training and validation performed on 50% of CIFAR-10 dataset.
- Plots:
  - Validation Loss vs Epochs
  - Validation Accuracy vs Epochs
- Prints final test accuracy for each optimizer

---

## 📦 Requirements

Install necessary packages using:

```bash
pip install tensorflow matplotlib seaborn scikit-learn
```

---

## ▶️ How to Run

Simply run the script:

```bash
python "ANN(Different optimizers).py"
```

Make sure your environment has access to the internet to download the CIFAR-10 dataset if not cached locally.

---

## 🎯 Objective

To observe and compare:
- Convergence behavior
- Generalization performance
- Speed of different optimizers on a classification task

---


---

## 👤 Authors

**Ishan Bansal**  
*Deep Learning Explorer*
