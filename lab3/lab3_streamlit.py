import streamlit as st
import collections
import math
import numpy as np
import spacy
from spacy import displacy
import pandas as pd

# --- App Configuration ---
st.set_page_config(page_title="NLP Lab Suite", layout="wide")

# --- Helper Functions & Classes ---

# Q1: N-gram Model Functions
def build_ngram_models(corpus_text):
    tokens = corpus_text.lower().split()
    unigram_counts = collections.Counter(tokens)
    bigrams = list(zip(tokens, tokens[:-1]))
    bigram_counts = collections.Counter(bigrams)
    return unigram_counts, bigram_counts, set(tokens)

def calculate_bigram_prob(word1, word2, unigram_counts, bigram_counts):
    bigram = (word1.lower(), word2.lower())
    if bigram in bigram_counts and unigram_counts[word1.lower()] > 0:
        return bigram_counts[bigram] / unigram_counts[word1.lower()]
    return 0

def calculate_perplexity(test_sentence, unigram_counts, bigram_counts, vocab):
    test_tokens = test_sentence.lower().split()
    test_bigrams = list(zip(test_tokens, test_tokens[1:]))
    V = len(vocab)
    log_prob_sum = 0

    if not test_tokens:
        return 1.0

    for bigram in test_bigrams:
        word1, word2 = bigram
        numerator = bigram_counts.get(bigram, 0) + 1
        denominator = unigram_counts.get(word1, 0) + V
        if denominator > 0:
            prob = numerator / denominator
            log_prob_sum += math.log2(prob)
        else:
            return float('inf')

    N = len(test_tokens)
    if N == 0:
        return 1.0

    perplexity = 2 ** (-log_prob_sum / N)
    return perplexity

# Q2: HMM POS Tagger Functions
def float_default_dict():
    return collections.defaultdict(float)

@st.cache_data
def train_hmm(corpus):
    emission_counts = collections.defaultdict(int)
    transition_counts = collections.defaultdict(int)
    tag_counts = collections.defaultdict(int)

    for sentence in corpus:
        prev_tag = '<s>'
        tag_counts['<s>'] += 1
        for word, tag in sentence:
            emission_counts[(tag, word.lower())] += 1
            transition_counts[(prev_tag, tag)] += 1
            tag_counts[tag] += 1
            prev_tag = tag

    emission_probs = collections.defaultdict(float_default_dict)
    for (tag, word), count in emission_counts.items():
        emission_probs[tag][word] = count / tag_counts[tag]

    transition_probs = collections.defaultdict(float_default_dict)
    for (prev_tag, tag), count in transition_counts.items():
        transition_probs[prev_tag][tag] = count / tag_counts[prev_tag]

    return transition_probs, emission_probs

def viterbi_tagger(sentence_tokens, tags, transition_probs, emission_probs):
    if not sentence_tokens:
        return []

    viterbi = np.zeros((len(tags), len(sentence_tokens)))
    backpointer = np.zeros((len(tags), len(sentence_tokens)), dtype=int)

    for i, tag in enumerate(tags):
        transition_p = transition_probs['<s>'].get(tag, 0)
        emission_p = emission_probs[tag].get(sentence_tokens[0].lower(), 0)
        viterbi[i, 0] = transition_p * emission_p

    for t in range(1, len(sentence_tokens)):
        for i, tag in enumerate(tags):
            prev_viterbi = viterbi[:, t - 1]
            prev_transitions = np.array([transition_probs.get(prev_tag, {}).get(tag, 0) for prev_tag in tags])
            epsilon = 1e-10
            emission_p = emission_probs[tag].get(sentence_tokens[t].lower(), epsilon)
            probs = prev_viterbi * prev_transitions * emission_p
            viterbi[i, t] = np.max(probs)
            backpointer[i, t] = np.argmax(probs)

    best_path = []
    last_tag_index = np.argmax(viterbi[:, -1])
    best_path.append(tags[last_tag_index])

    for t in range(len(sentence_tokens) - 1, 0, -1):
        last_tag_index = backpointer[last_tag_index, t]
        best_path.insert(0, tags[last_tag_index])

    return best_path

# Q3: NER Function
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm' in your terminal.")
        return None

# --- Main App ---
st.sidebar.title("NLP Lab Suite")
st.sidebar.markdown("A Streamlit app for the exercises in lab3.ipynb.")

menu_choice = st.sidebar.radio(
    "Select an Exercise",
    ("Q1: N-gram Language Model", "Q2: POS Tagging (HMM)", "Q3: Named Entity Recognition (NER)")
)

# Q1 Page
if menu_choice == "Q1: N-gram Language Model":
    st.title("Q1: N-gram Language Model")
    st.markdown("Estimate n-gram probabilities and calculate perplexity.")

    st.header("1. Training Corpus")
    corpus_text = st.text_area(
        "Enter the corpus for training the model:",
        "<s> i am sam </s>\n<s> sam i am </s>\n<s> i do not like green eggs and ham </s>\n<s> i do not like them sam i am </s>",
        height=120
    )

    unigram_counts, bigram_counts, vocab = build_ngram_models(corpus_text)

    st.header("2. Unigram Probability Table")
    total_words = sum(unigram_counts.values())
    unigram_probs = {word: count / total_words for word, count in unigram_counts.items()}
    unigram_df = pd.DataFrame(unigram_probs.items(), columns=['Word', 'Probability']).sort_values(by="Probability", ascending=False)
    st.dataframe(unigram_df, use_container_width=True)

    st.header("3. Bigram Probability Calculation")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="P(sam | am)",
            value=f"{calculate_bigram_prob('am', 'sam', unigram_counts, bigram_counts):.4f}"
        )
    with col2:
        st.metric(
            label="P(green | like)",
            value=f"{calculate_bigram_prob('like', 'green', unigram_counts, bigram_counts):.4f}"
        )

    st.header("4. Perplexity Calculation")
    test_sentence = st.text_input("Enter a test sentence:", value="<s> i like ham </s>")
    if st.button("Calculate Perplexity"):
        perplexity = calculate_perplexity(test_sentence, unigram_counts, bigram_counts, vocab)
        st.success(f"The perplexity of the sentence is: {perplexity:.4f}")
        st.info("This calculation uses Add-1 (Laplace) smoothing.")

# Q2 Page
elif menu_choice == "Q2: POS Tagging (HMM)":
    st.title("Q2: Part-of-Speech (POS) Tagging with HMM")
    st.info("This implements a Viterbi-based Hidden Markov Model (HMM) tagger.")

    training_corpus = [
        [('Janet', 'NNP'), ('will', 'MD'), ('back', 'VB'), ('the', 'DT'), ('bill', 'NN')],
        [('the', 'DT'), ('company', 'NN'), ('will', 'MD'), ('back', 'VB'), ('the', 'DT'), ('bill', 'NN')],
        [('Janet', 'NNP'), ('can', 'MD'), ('pass', 'VB'), ('the', 'DT'), ('bill', 'NN')]
    ]

    all_tags = sorted(list(set(tag for sent in training_corpus for word, tag in sent)))
    transition_probs, emission_probs = train_hmm(training_corpus)

    st.header("1. Trained HMM Model")
    with st.expander("View Training Corpus"):
        st.json(training_corpus)

    # Show transition probabilities as DataFrame
    st.subheader("Transition Probability Table")
    transition_df = pd.DataFrame(transition_probs).fillna(0).T
    st.dataframe(transition_df.style.format("{:.3f}"), use_container_width=True)

    # Show emission probabilities as DataFrame
    st.subheader("Emission Probability Table")
    emission_df = pd.DataFrame(emission_probs).fillna(0).T
    st.dataframe(emission_df.style.format("{:.3f}"), use_container_width=True)

    st.header("2. Tag a Sentence")
    sentence_to_tag_str = st.text_input("Enter a sentence to tag:", value="Janet will back the bill")

    if st.button("Tag Sentence"):
        sentence_tokens = sentence_to_tag_str.split()
        predicted_tags = viterbi_tagger(sentence_tokens, all_tags, transition_probs, emission_probs)
        st.subheader("Tagged Result")
        result_df = pd.DataFrame({'Token': sentence_tokens, 'Predicted Tag': predicted_tags})
        st.table(result_df)

# Q3 Page
elif menu_choice == "Q3: Named Entity Recognition (NER)":
    st.title("Q3: Named Entity Recognition (NER) with spaCy")
    st.markdown("This uses a pre-trained spaCy model to identify and classify named entities like persons, locations, and organizations.")

    nlp = load_spacy_model()

    if nlp:
        input_text = st.text_area("Enter text for NER:", "John lives in New York and works for Google.", height=100)

        if st.button("Recognize Entities"):
            doc = nlp(input_text)

            st.header("Visualized Entities")
            html = displacy.render(doc, style="ent", jupyter=False)
            st.write(html, unsafe_allow_html=True)

            st.header("Entity Breakdown")
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            if entities:
                ent_df = pd.DataFrame(entities, columns=["Entity Text", "Label"])
                st.table(ent_df)
            else:
                st.info("No named entities were found in the text.")
