# Linear Regression from Scratch using Gradient Descent

This project implements **Linear Regression** without using any machine learning libraries such as scikit-learn.  
The goal is to understand how models actually learn from data using **Gradient Descent**.

---

## Problem Statement

Given house sizes:

x = [1, 2, 3, 4, 5]  
y = [3, 5, 7, 9, 11]

The hidden relationship is:

y = 2x + 1

The model must discover this rule **by minimizing error**, not by hard-coding it.

---

## Learning Algorithm

The model uses:

- MSE (Mean Squared Error) as loss function  
- Batch Gradient Descent for optimization  

Update rules:

w = w - lr * dw  
b = b - lr * db

Where gradients are computed manually.

---

## What this project demonstrates

- How loss functions control learning behaviour  
- Why learning rate is critical for convergence  
- How convex loss enables stable optimization  
- Why normalization and scale matter  

---

## How to Run

```bash
python linear_regression.py

Author: Ishaan Sharma

