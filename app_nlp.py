import math
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import movie_reviews
from nltk.util import ngrams
import nltk

# ---------------------------------
# Initial Setup
# ---------------------------------
st.set_page_config(
    page_title="Movie Review Language Model",
    page_icon="🎬",
    layout="wide"
)

# Download once (remove later for full offline use)
nltk.download('movie_reviews')

# ---------------------------------
# Load Dataset
# ---------------------------------
documents = [movie_reviews.raw(fid).lower() for fid in movie_reviews.fileids()]
train_docs = documents[:1600]
test_docs = documents[1600:]

def tokenize(docs):
    tokens = []
    for d in docs:
        tokens.extend(d.split())
    return tokens

train_tokens = tokenize(train_docs)

# ---------------------------------
# Build N-gram Models
# ---------------------------------
unigram_counts = Counter(train_tokens)
bigram_counts = Counter(ngrams(train_tokens, 2))
trigram_counts = Counter(ngrams(train_tokens, 3))
total_words = len(train_tokens)

def safe_prob(p):
    return p if p > 0 else 1e-6

def unigram_prob(w):
    return unigram_counts[w] / total_words if w in unigram_counts else 0

def bigram_prob(w1, w2):
    return bigram_counts[(w1, w2)] / unigram_counts[w1] if unigram_counts[w1] > 0 else 0

def trigram_prob(w1, w2, w3):
    return trigram_counts[(w1, w2, w3)] / bigram_counts[(w1, w2)] if bigram_counts[(w1, w2)] > 0 else 0

# ---------------------------------
# Sentence Probability & Perplexity
# ---------------------------------
def sentence_probability(sentence, model):
    words = sentence.lower().split()
    prob = 1

    if model == "Unigram":
        for w in words:
            prob *= safe_prob(unigram_prob(w))

    elif model == "Bigram":
        for i in range(len(words) - 1):
            prob *= safe_prob(bigram_prob(words[i], words[i+1]))

    else:  # Trigram
        for i in range(len(words) - 2):
            prob *= safe_prob(trigram_prob(words[i], words[i+1], words[i+2]))

    return prob

def perplexity(prob, N):
    return pow(prob, -1 / N)

# ---------------------------------
# Text Auto-Completion Logic
# ---------------------------------
def predict_next_word(text, model):
    words = text.lower().split()

    if model == "Unigram":
        return unigram_counts.most_common(1)[0][0]

    elif model == "Bigram" and len(words) >= 1:
        candidates = {
            w2: bigram_prob(words[-1], w2)
            for (w1, w2) in bigram_counts
            if w1 == words[-1]
        }

    elif model == "Trigram" and len(words) >= 2:
        candidates = {
            w3: trigram_prob(words[-2], words[-1], w3)
            for (w1, w2, w3) in trigram_counts
            if w1 == words[-2] and w2 == words[-1]
        }
    else:
        return None

    if candidates:
        return max(candidates, key=candidates.get)
    return None

def autocomplete_sentence(text, model, n_words):
    current = text
    for _ in range(n_words):
        next_word = predict_next_word(current, model)
        if not next_word:
            break
        current += " " + next_word
    return current

# ---------------------------------
# Sidebar
# ---------------------------------
st.sidebar.title("⚙️ Controls")
model = st.sidebar.radio(
    "Select N-gram Model",
    ["Unigram", "Bigram", "Trigram"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "🎯 **Project Objective**\n\n"
    "Learn and compare Unigram, Bigram, and Trigram "
    "language models on movie review text."
)

# ---------------------------------
# Main UI
# ---------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🎬 Movie Review Language Model</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; font-size:18px;'>"
    "Domain-specific N-gram modeling using NLTK Movie Reviews dataset"
    "</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# -------- Sentence Analysis --------
st.subheader("✍️ Sentence Probability & Perplexity")

sentence = st.text_input(
    "Enter a movie review sentence:",
    placeholder="the movie was amazing"
)

if st.button("🔍 Analyze Sentence"):
    if sentence.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        prob = sentence_probability(sentence, model)
        pp = perplexity(prob, len(sentence.split()))

        st.success("Analysis Completed ✅")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("📐 Sentence Probability", f"{prob:.2e}")
        with col2:
            st.metric("📊 Perplexity", f"{pp:.2f}")

# -------- Text Auto-Completion --------
st.markdown("---")
st.subheader("✨ Text Auto-Completion")

seed_text = st.text_input(
    "Start typing a movie review:",
    placeholder="the movie was"
)

num_words = st.slider(
    "Number of words to predict",
    min_value=1,
    max_value=5,
    value=3
)

if st.button("🚀 Auto-Complete"):
    if seed_text.strip() == "":
        st.warning("Please enter starting words.")
    else:
        completed = autocomplete_sentence(seed_text, model, num_words)
        st.success("Auto-Completion Result")
        st.write(f"📝 **Completed Text:** {completed}")

# -------- Graph Comparison --------
st.markdown("---")
st.subheader("📊 Model Comparison (Perplexity)")

def avg_perplexity(model_name):
    total = 0
    count = 0
    for doc in test_docs[:40]:
        words = doc.split()
        if len(words) < 5:
            continue
        sent = " ".join(words[:10])
        p = sentence_probability(sent, model_name)
        total += perplexity(p, len(sent.split()))
        count += 1
    return total / count

if st.button("📈 Show Comparison Graph"):
    u = avg_perplexity("Unigram")
    b = avg_perplexity("Bigram")
    t = avg_perplexity("Trigram")

    models = ["Unigram", "Bigram", "Trigram"]
    values = [u, b, t]

    plt.figure()
    plt.bar(models, values)
    plt.xlabel("N-gram Models")
    plt.ylabel("Average Perplexity")
    plt.title("N-gram Model Comparison on Movie Reviews")
    st.pyplot(plt)

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("---")
st.caption("📌 NLP Mini Project | Domain-Specific Language Modeling")
