# streamlit_app.py
import streamlit as st
import nltk
from nltk import word_tokenize, pos_tag
from nltk.util import ngrams
from nltk.probability import FreqDist, ConditionalFreqDist, LidstoneProbDist
from nltk.corpus import brown
import pandas as pd
import math
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import streamlit.components.v1 as components



st.title(" NLP Toolkit — N-gram Models, POS & NER Visuals")

mode = st.radio("Select a task:", [
    "N-gram Model",
    "POS Tagging",
    "Named Entity Recognition (IOB Format)"
])

# ------------------ N-GRAM MODEL ------------------
if mode.startswith("📊"):
    st.header("N-gram Model")
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

    if uploaded_file:
        corpus = uploaded_file.read().decode("utf-8").lower()
        st.subheader("📄 Corpus Text")
        st.text(corpus)

        tokens = [token for token in word_tokenize(corpus) if token.isalpha()]

        total_tokens = len(tokens)
        bigrams = list(ngrams(tokens, 2))

        # Unigram and Bigram Models
        unigram_fd = FreqDist(tokens)
        bigram_fd = FreqDist(bigrams)
        prev_word_counts = FreqDist(w1 for (w1, w2) in bigrams)

        st.subheader("🔍 Choose an Option")
        choice = st.radio("Select one:", [
            "1️⃣ Show all unigrams",
            "2️⃣ Lookup unigram probability of a word",
            "3️⃣ Show all bigrams",
            "4️⃣ Lookup bigram probability for two words",
            "5️⃣ Compute Perplexity for a sentence",
            "6️⃣ Visualize N-grams"
        ])

        def bigram_prob(w1, w2):
            return bigram_fd[(w1, w2)] / prev_word_counts[w1] if prev_word_counts[w1] > 0 else 0

        if choice.startswith("1"):
            st.subheader("📊 Unigram Table")
            unigram_table = pd.DataFrame({
                "Word": list(unigram_fd.keys()),
                "Count": list(unigram_fd.values()),
                "Probability": [round(unigram_fd[w]/total_tokens, 6) for w in unigram_fd]
            })
            unigram_table["IsAlpha"] = unigram_table["Word"].str.isalpha()
            unigram_table = unigram_table.sort_values(by=["IsAlpha", "Word"], ascending=[False, True]).drop(columns="IsAlpha")
            st.dataframe(unigram_table)

        elif choice.startswith("2"):
            word = st.text_input("Enter a word to look up its unigram probability:")
            if word:
                count = unigram_fd[word]
                prob = count / total_tokens if total_tokens > 0 else 0
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
            test_sentence = st.text_input("Enter test sentence:", "I want English food")
            if test_sentence:
                test_tokens = word_tokenize(test_sentence.lower())
                test_bigrams = list(ngrams( test_tokens, 2))
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
            probs = [unigram_fd[w]/total_tokens for w in words]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            sns.barplot(y=list(words), x=list(freqs), ax=ax1)
            ax1.set_title("Top 20 Unigrams - Frequency")
            ax1.set_xlabel("Frequency")

            sns.barplot(y=list(words), x=list(probs), ax=ax2)
            ax2.set_title("Top 20 Unigrams - Probability")
            ax2.set_xlabel("Probability")

            st.pyplot(fig)

# ------------------ POS TAGGING ------------------
elif mode.startswith("🔤"):
    st.header("POS Tagging")

    sentence = st.text_input("Enter a sentence:", "I want to learn NLP")
    if sentence:
        tokens = word_tokenize(sentence)
        pos_tags = pos_tag(tokens)
        st.write("**POS Tags:**", pos_tags)

        # POS frequency calculation only (no graph)
        tag_freq = Counter(tag for word, tag in pos_tags)
        st.write("📊 POS Tag Frequencies:", dict(tag_freq))

        tagged_sents = brown.tagged_sents(categories='news')[:1000]
        cfd_trans = ConditionalFreqDist()
        cfd_emit = ConditionalFreqDist()

        for sent in tagged_sents:
            prev_tag = '<s>'
            for word, tag in sent:
                cfd_emit[tag][word.lower()] += 1
                cfd_trans[prev_tag][tag] += 1
                prev_tag = tag

        transition_probs = {tag: LidstoneProbDist(cfd_trans[tag], 0.1, bins=len(cfd_trans[tag])) for tag in cfd_trans}
        emission_probs = {tag: LidstoneProbDist(cfd_emit[tag], 0.1, bins=len(cfd_emit[tag])) for tag in cfd_emit}

        st.success("✅ HMM tagger created using Brown corpus")

        st.subheader("🔍 Probability Check")
        check_word = st.text_input("Word (for emission):", "food")
        check_tag = st.selectbox("Tag (for emission):", list(emission_probs.keys()))
        st.write(f"P({check_word}|{check_tag}) = {emission_probs[check_tag].prob(check_word):.6f}")

        check_next_tag = st.selectbox("Next tag (for transition from <s>):", list(transition_probs['<s>'].samples()))
        st.write(f"P({check_next_tag}|<s>) = {transition_probs['<s>'].prob(check_next_tag):.6f}")

# ...existing code...

elif mode.startswith("🧾"):
    st.header("Named Entity Recognition (IOB Format)")

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
        st.stop()

    text = st.text_input("Enter a sentence:", "John lives in New York")
    if text:
        doc = nlp(text)
        ner_data = [(token.text, token.ent_iob_, token.ent_type_ if token.ent_type_ else "O") for token in doc]
        ner_df = pd.DataFrame(ner_data, columns=["Token", "IOB Tag", "Entity Type"])
        st.dataframe(ner_df)

        st.subheader("📌 Visual Highlight (Displacy)")
        html = spacy.displacy.render(doc, style="ent")
        components.html(html, height=300, scrolling=True)

        entity_freq = Counter(ent.label_ for ent in doc.ents)
        if entity_freq:
            st.write("📊 Named Entity Counts:", dict(entity_freq))
        else:
            st.info("No named entities found in the sentence.")