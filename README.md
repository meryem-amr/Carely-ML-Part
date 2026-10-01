# Carely 

**Machine-learning models for Carely, a smart baby monitoring and maternal wellness system.**

Carely helps new parents keep track of their baby's health and wellbeing, and
supports the mother during the postpartum period. It combines wearable sensors,
an IoT sound device, machine-learning models, and a mobile app in one connected system.

> This repository contains the **machine-learning part** of Carely: the
> **cry detection** and **sleep detection** models. 

---

## The Problem

New parents can't watch their baby around the clock. Crying and sleep patterns
are hard to track by hand, and they say a lot about a baby's needs and health.

## Our Solution

| Component | What it does |
|---|---|
| **Biometric bracelet** | Measures heart rate, temperature, and movement |
| **IoT sound device** | Captures audio and plays lullabies |
| **ML models (this repo)** | Detect crying from audio and sleep from motion data |
| **Backend** | Authentication, alert logic, and model serving |
| **Mobile app (Flutter)** | Shows vitals, sleep, and alerts to parents |


## Models

| Model | Input | Output | Algorithm |
|---|---|---|---|
| [Cry Detection](#1-cry-detection) | Audio from the sound device | Cry / not cry | CNN (TensorFlow/Keras) |
| [Sleep Detection](#2-sleep-detection) | Accelerometer data from the bracelet | Asleep / awake | XGBoost |

---

## 1. Cry Detection

A convolutional neural network that classifies short audio clips as **cry** or
**not cry**.

### Architecture

- Three `Conv2D` blocks
- Sigmoid output for binary classification
- Built with TensorFlow / Keras

### Results

| Metric | Value |
|---|---|
| Test accuracy | ~99% |



### Dataset

- **Cry samples:** <Kaggle dataset name / link>
- **Non-cry samples:** recorded by the team

### Preprocessing

<Describe the audio features, e.g. mel-spectrogram, sample rate, clip length, image size>

### Known Limitation: Domain Mismatch

The model scores about 99% on the test set but can misclassify **loud non-cry
sounds** when deployed. The cry data came from Kaggle while the non-cry data was
recorded by the team, so the model partly learned the differences between the
two *sources* rather than only the difference between crying and not crying.

Planned improvements:

- Collect recordings from the actual IoT microphone
- Add **hard-negative samples** (loud, cry-like sounds that are not crying)
- Use IoT recordings as a **held-out test set** for a realistic accuracy estimate

---

## 2. Sleep Detection

A gradient-boosted model (XGBoost) that predicts whether the baby is asleep from
the bracelet's motion data.

### Input

The bracelet sends raw **X/Y/Z** acceleration. Two motion features are computed
from it:

- **ENMO** (Euclidean Norm Minus One)
- **anglez** (arm angle)

These feed the model together with other derived features (9 in total).

### Results

| Metric | Value |
|---|---|
| F1 score | 0.8763 |
| ROC-AUC | 0.9576 |
| Decision threshold | 0.65 |
| Smoothing window | 11 steps |

The decision threshold of 0.65 was chosen to maximize F1. Predictions are then
smoothed over an 11-step window to remove short, unrealistic flips between
asleep and awake.

### Dataset

Trained on the **Child Mind Institute** sleep dataset from Kaggle.
