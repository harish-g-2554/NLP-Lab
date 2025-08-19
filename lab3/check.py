# streamlit_app.py
import streamlit as st
import nltk
from nltk import word_tokenize, ngrams
from nltk.probability import FreqDist
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download("punkt")

st.title("🧠 NLP Toolkit — N-gram Models")

uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file:
    corpus = uploaded_file.read().decode("utf-8").lower()
    st.subheader("📄 Corpus Text")
    st.text(corpus)

    # Add sentence boundary tokens
    tokens = ["<s>"] + word_tokenize(corpus) + ["</s>"]
    bigrams = list(ngrams(tokens, 2))

    # Frequency distributions
    unigram_fd = FreqDist(tokens)
    bigram_fd = FreqDist(bigrams)
    prev_word_counts = FreqDist(w1 for (w1, w2) in bigrams)

    # UI Options
    st.subheader("🔍 Choose an Option")
    choice = st.radio("Select one:", [
        "1️⃣ Show all unigrams",
        "2️⃣ Lookup unigram probability of a word",
        "3️⃣ Show all bigrams",
        "4️⃣ Lookup bigram probability for two words",
        "5️⃣ Compute Perplexity for a sentence",
        "6️⃣ Visualize Top 20 Unigrams"
    ])

    def bigram_prob(w1, w2):
        return bigram_fd[(w1, w2)] / prev_word_counts[w1] if prev_word_counts[w1] > 0 else 0

    if choice.startswith("1"):
        st.subheader("📊 Unigram Table")
        unigram_table = pd.DataFrame({
            "Word": list(unigram_fd.keys()),
            "Count": list(unigram_fd.values()),
            "Probability": [round(unigram_fd[w]/len(tokens), 6) for w in unigram_fd]
        })
        st.dataframe(unigram_table)

    elif choice.startswith("2"):
        word = st.text_input("Enter a word to look up its unigram probability:")
        if word:
            count = unigram_fd[word]
            prob = count / len(tokens) if len(tokens) > 0 else 0
            st.write(f"**Count:** {count}")
            st.write(f"**Probability:** {round(prob, 6)}")

    elif choice.startswith("3"):
        st.subheader("📊 Bigram Table")
        bigram_data = {
            "Bigram": [],
            "Count": [],
            "Probability": []
        }
        for (w1, w2), count in bigram_fd.items():
            prob = count / prev_word_counts[w1] if prev_word_counts[w1] > 0 else 0
            bigram_data["Bigram"].append(f"{w1} {w2}")
            bigram_data["Count"].append(count)
            bigram_data["Probability"].append(round(prob, 6))

        df_bigram = pd.DataFrame(bigram_data).sort_values(by="Bigram").reset_index(drop=True)
        st.dataframe(df_bigram)

    elif choice.startswith("4"):
        col1, col2 = st.columns(2)
        with col1:
            w1 = st.text_input("First word")
        with col2:
            w2 = st.text_input("Second word")
        if w1 and w2:
            count = bigram_fd[(w1, w2)]
            prob = count / prev_word_counts[w1] if prev_word_counts[w1] > 0 else 0
            st.write(f"**Bigram:** ({w1}, {w2})")
            st.write(f"**Count:** {count}")
            st.write(f"**Probability:** {round(prob, 6)}")

    elif choice.startswith("5"):
        st.subheader("📉 Perplexity Calculator")
        test_sentence = st.text_input("Enter test sentence:", "I like green vegetables")
        single_word_check=test_sentence.split()
        if len(single_word_check) == 1:
            test_tokens =word_tokenize(test_sentence.lower())
            test_bigrams = list(ngrams(["<s>"] + test_tokens, 2))
            N = len(test_bigrams)
            log_prob_sum = 0
            zero_found = False

            for bg in test_bigrams:
                p = bigram_prob(bg[0], bg[1])
                if p > 0:
                    log_prob_sum += math.log2(p)
                else:
                    zero_found = True
                    break

            perplexity = 2 ** (-log_prob_sum / N) if not zero_found else float('inf')
            st.write(f"**Perplexity:** {perplexity:.4f}" if not zero_found else "**Perplexity:** ∞ (Zero-probability bigram detected)")
        else:
            test_tokens =word_tokenize(test_sentence.lower())
            test_bigrams = list(ngrams(test_tokens, 2))
            N = len(test_bigrams)
            log_prob_sum = 0
            zero_found = False

            for bg in test_bigrams:
                p = bigram_prob(bg[0], bg[1])
                if p > 0:
                    log_prob_sum += math.log2(p)
                else:
                    zero_found = True
                    break

            perplexity = 2 ** (-log_prob_sum / N) if not zero_found else float('inf')
            st.write(f"**Perplexity:** {perplexity:.4f}" if not zero_found else "**Perplexity:** ∞ (Zero-probability bigram detected)")

    elif choice.startswith("6"):
        st.subheader("📊 Top 20 Unigrams: Frequency & Probability")
        top_unigrams = unigram_fd.most_common(20)
        words, freqs = zip(*top_unigrams)
        probs = [unigram_fd[w]/len(tokens) for w in words]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        sns.barplot(y=list(words), x=list(freqs), ax=ax1)
        ax1.set_title("Top 20 Unigrams - Frequency")
        ax1.set_xlabel("Frequency")

        sns.barplot(y=list(words), x=list(probs), ax=ax2)
        ax2.set_title("Top 20 Unigrams - Probability")
        ax2.set_xlabel("Probability")

        st.pyplot(fig)
