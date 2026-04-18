## 🐦 BeakSpeak

BeakSpeak is a project setting out to understand the subtle differences between bird species, and turning that into a usable ML product.

Fine-grained image classification is hard - many bird species look almost identical to both humans and models. This project explores how far we can push visual classification, and where it starts to break.

👉 [**Live demo** (Streamlit)]("https://beakspeak-309693526010.europe-west2.run.app/")

### 🎯 Goals

- Build a robust fine-grained image classifier for bird species
- Explore the challenges of visually similar classes
- Demonstrate end-to-end ML thinking:
```data → modelling → evaluation → product → deployment```

### 🎯 Overview

BeakSpeak is an end-to-end deep learning project that identifies bird species from images.

Using the CUB-200-2011 dataset of 200 classes, the project explores how well modern convolutional architectures can distinguish between visually similar classes, and what happens when they can’t.

The final output is a user-facing app that allows image upload and returns top predictions with confidence scores.

### 🧩 Problem

Fine-grained classification presents specific challenges:

- Many classes differ only in **subtle visual features**
- Strong performance requires **transfer learning**, not training from scratch
- Models often produce **high-confidence errors**
- Real-world inputs may fall **outside the training distribution**

### ⚙️ Approach

The project evolved in stages:

1. **Baseline modelling**

- Built custom CNN trained from scratch with ~2.2% accuracy (essentially unusable)
- Demonstrates difficulty of fine-grained classification without pretrained features

2. **Transfer learning**

- Model: EfficientNetB0 with ~66.9% top-1 test accuracy and 84.9% top-3 - a significant improvement over the baseline
- Strategy: Frozen base and fine-tuned top layers for improved performance

3. **Evaluation**

- Analysed confusion patterns between similar species and investigated where the model struggles
- Confusion patterns reveals repeated class pairs dominate errors, suggesting feature overlaps rather than general model failure
- Error analysis shows most classes perform well (F1 ~0.60-0.85), with misclassifications clustering within small subsets of similar species (many “errors” are visually ambiguous even to humans)

4. **Product decisions**

To improve usability:

- Introduced a confidence threshold (~0.53) based on validation data analysis
- Balances catching low-quality predictions versus avoiding over-warning
- Surface top-3 predictions instead of single label
- Explicitly communicate uncertainty

### 4. Product decisions

Model behaviour informed several design choices in the app:

**Confidence threshold (~0.53)**
- Low-confidence predictions are flagged rather than shown as definitive.
- This captures a meaningful proportion of likely errors while avoiding excessive warnings on correct predictions.

**Shortlist predictions (top-1 plus 3 alternatives)**
- The correct species is often present even when the top prediction is wrong.
- Presenting a small set of likely candidates reflects how the model actually behaves and supports user judgement.

**Explicit uncertainty handling**
- The app distinguishes between confident predictions (shown normally) and uncertain predictions (flagged with guidance) to avoid presenting all outputs as equally reliable.

### 🖥️ Application

BeakSpeak is available as a Streamlit app where users can:

1. Upload an image
2. View top predictions and confidence scores, alongside alternative likely species
3. Receive low-confidence warnings when appropriate
4. Check whether a species exists in the model’s training set
5. Explore more about a species through signposted content

The interface is designed to reflect model behaviour:

- Confident predictions are surfaced clearly
- Uncertain predictions are flagged and contextualised
- Multiple plausible species are shown where ambiguity exists

Rather than forcing a single answer, the app exposes uncertainty where it genuinely exists in the data.

### ☁️ Deployment

The app is:

- Containerised with Docker
- Deployed on Google Cloud Run
- Served as a lightweight inference service

For simplicity, the deployed container includes:
- the trained model (~31MB)
- a minimal species mapping file

Raw training data is not included.

### 📦 Data

This project uses the CUB-200-2011 dataset. Key characteristics include:

- 200 bird species
- ~6,000 training images
- Fine-grained annotations (parts, attributes)

⚠️ **Notes**: Raw image data is not included in this repo. The deployed app uses only class labels and trained model weights.

### 🛠️ Tech stack

Python

TensorFlow / Keras

pandas, numpy

Streamlit

Docker

Google Cloud Run


### 💡 Why this project

This project brings together:

- Deep learning for fine-grained computer vision
- Product thinking around uncertainty and user trust
- Practical deployment using containerised ML systems
- Real-world challenges such as class similarity, data limitations and out-of-distribution inputs

### 🔮 Potential next steps

- Improve robustness to non-bird / low-quality images
- Incorporate part-based features (e.g. beak, wing, colour)
- Add richer species information in the UI
- Evaluate model calibration more formally
- Move artefacts to external storage for scalability

### Reference

This project makes use of the [CUB-200-2011 dataset](https://authors.library.caltech.edu/records/cvm3y-5hh21) - thanks to the authors over at Caltech.

> Welinder, P., Branson, S., Mita, T., Wah, C., Schroff, F., Belongie, S., & Perona, P.
> *Caltech-UCSD Birds 200*.
> California Institute of Technology. CNS-TR-2010-001, 2010.
