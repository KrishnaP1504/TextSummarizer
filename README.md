# 📘 Text Summarization using Deep Learning (Seq2Seq + Attention)

This project implements an **Abstractive Text Summarization** system using a **Sequence-to-Sequence (Seq2Seq)** architecture with **stacked LSTMs** and an **Attention Mechanism**.  
The model is trained on the **Amazon Fine Food Reviews** dataset and generates short, meaningful summaries from long review texts.

---

## ⭐ Features

- Abstractive summarization using **Encoder–Decoder LSTM architecture**  
- **Bahdanau-style Attention** for improved context retention  
- Full NLP preprocessing pipeline:
  - HTML cleaning  
  - Tokenization  
  - Stopword removal  
  - Contraction expansion  
  - Stemming  
  - Sequence vectorization  
- Custom inference pipeline (separate encoder + decoder models)  
- Real-time summary generation for unseen inputs  
- Saved TensorFlow model (`saved_model.pb` + weight files)

---

## 📂 Project Structure

TextSummarizer/
│
├── text_summarizer.py                                     # Main file for training + inference
├── Reviews.csv                                            # Dataset (Amazon Fine Food Reviews)
│
├── s2s/                                                   # Saved model folder created after training
│ ├── saved_model.pb
│ └── variables/
│ ├── variables.index
│ └── variables.data-00000-of-00001
│
└── README.md                                              # Documentation


> Note:  
> `variables.data-00000-of-00001` is a **TensorFlow weight file** containing all trained parameters.  
> Do NOT delete it.

---

## 🧠 Model Architecture Overview

### 🔹 Encoder
- Word Embedding Layer  
- 3 × LSTM Layers (500 units each)  
- Outputs encoder hidden states + final (`h`, `c`) states  

### 🔹 Decoder
- Target Embedding layer  
- LSTM initialized with encoder states  
- Attention over encoder outputs  
- Dense Softmax output layer  

### 🔹 Attention
- Helps the decoder focus on important words in the input text  
- Improves summary quality significantly  

---

## 📊 Dataset Details
The model uses the **Amazon Fine Food Reviews** dataset.

- 500,000+ total reviews  
- Each review includes:  
  - `Text` → full review  
  - `Summary` → short summary  
- This project uses **first 100,000 rows**  

Dataset Link:  
https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

---

## ⚙️ Installation Guide (Python 3.10 Required)



