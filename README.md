# Movie-Review-Language-Model
NLP  project using Unigram, Bigram, and Trigram models with Streamlit and NLTK movie reviews dataset.

# 🎬 Movie Review Language Model using NLP

This project is a **Natural Language Processing (NLP) mini project** built using **Python and Streamlit**.  
It uses the **NLTK Movie Reviews dataset** to build and compare **Unigram, Bigram, and Trigram language models**.

The project allows users to:
- Analyze sentence probability
- Calculate perplexity
- Auto-complete text
- Compare N-gram models using graph

---

## 🚀 Features

### ✍️ Sentence Probability & Perplexity
- Calculates the probability of a given movie review sentence
- Computes perplexity score
- Supports:
  - Unigram
  - Bigram
  - Trigram

---

## ✨ Text Auto-Completion
Predicts the next words based on the selected model.

Example:

Input: the movie was  
Output: the movie was amazing

---

## 📊 Model Comparison
Displays **average perplexity comparison graph** for:

- Unigram
- Bigram
- Trigram

This helps in understanding which model performs better.

---

## 🛠️ Technologies Used
- Python
- Streamlit
- NLTK
- Matplotlib
- Counter
- N-grams

---

## 📚 Dataset Used
This project uses the **NLTK Movie Reviews Dataset**.

```python
from nltk.corpus import movie_reviews
