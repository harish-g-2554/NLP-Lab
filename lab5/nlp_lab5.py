import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from math import log2

st.title("Lab 5: Sparse Vector (Embedding)")

st.sidebar.header("Corpus Input")
input_mode = st.sidebar.radio("Choose input mode:", ["Sample Corpus", "Manual Input"])

# Sample corpus
sample_corpus = [
    "Car insurance is important for safety.",
    "Auto and car are similar terms in transportation.",
    "The best insurance covers all damages."
]

corpus = []
if input_mode == "Sample Corpus":
    corpus = sample_corpus
    st.sidebar.write("Using preloaded sample corpus.")
else:
    num_docs = st.sidebar.number_input("Enter number of documents to upload:", min_value=1, max_value=20, step=1)
    uploaded_files = [st.sidebar.file_uploader(f"Upload Document {i+1}", type=["txt"], key=f"file_{i}") for i in range(num_docs)]
    if all(uploaded_files):
        for uploaded_file in uploaded_files:
            text = uploaded_file.read().decode("utf-8").strip()
            if text:
                corpus.append(text)

if corpus:
    st.subheader("Documents in Corpus")
    for i, doc in enumerate(corpus):
        st.markdown(f"**Doc{i+1}:** {doc}")

    # TF-IDF Vectorization
    st.header("1. TF-IDF Computation")
    vectorizer = TfidfVectorizer(use_idf=True)
    X = vectorizer.fit_transform(corpus)
    terms = vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(X.toarray().T, index=terms, columns=[f"Doc{i+1}" for i in range(len(corpus))])

    # Show raw term frequency and computed idf
    st.subheader("Term Frequency Table")
    tf_vectorizer = TfidfVectorizer(use_idf=False, norm=None)
    tf_matrix = tf_vectorizer.fit_transform(corpus).toarray().T
    tf_table = pd.DataFrame(tf_matrix, index=terms, columns=[f"Doc{i+1}" for i in range(len(corpus))])
    st.dataframe(tf_table)

    st.subheader("IDF Table")
    idf_values = vectorizer.idf_
    idf_table = pd.DataFrame({
        'term': terms,
        'df_t': np.sum(tf_matrix > 0, axis=1),
        'idf_t': np.round(idf_values, 2)
    })
    st.dataframe(idf_table)

    st.subheader("TF-IDF Table")
    st.dataframe(tfidf_df)

    # Normalization
    st.subheader("Normalized TF-IDF (Euclidean)")
    normalized = normalize(tfidf_df, axis=0)
    norm_df = pd.DataFrame(normalized, index=terms, columns=[f"Doc{i+1}" for i in range(len(corpus))])
    st.dataframe(norm_df)

    # Query input and scoring
    st.subheader("TF-IDF Query Scoring")
    query = st.text_input("Enter query terms (space-separated)", value="car insurance")
    query_terms = query.lower().split()

    query_score = {}
    for doc in tfidf_df.columns:
        score = sum(tfidf_df.loc[term, doc] for term in query_terms if term in tfidf_df.index)
        query_score[doc] = round(score, 4)

    st.write("Scores:", query_score)

    # Cosine similarity
    st.header("2. Cosine Similarity")
    sim_matrix = cosine_similarity(tfidf_df.values)
    sim_df = pd.DataFrame(sim_matrix, index=terms, columns=terms)
    st.dataframe(sim_df)

    selected_term = st.selectbox("Choose a word to find its nearest neighbour:", terms)
    if selected_term:
        idx = list(terms).index(selected_term)
        nearest_idx = np.argsort(-sim_matrix[idx])[1]  # excluding self
        st.write(f"Nearest word to '{selected_term}' is '{terms[nearest_idx]}'")

    # PMI Section
    st.header("3. Pointwise Mutual Information (PMI)")
    st.latex(r"PMI(w_i, w_j) = \log \frac{p(w_i, w_j)}{p(w_i)p(w_j)}")
    st.latex(r"P(w) = \frac{\text{Freq}(w)}{\text{totalWordCount}}")
    st.write("Compute the PMI instead of tf and tfidf, and recalculate the score for similarity check.")

    bin_vectorizer = CountVectorizer(binary=True)
    binary_X = bin_vectorizer.fit_transform(corpus)
    binary_terms = bin_vectorizer.get_feature_names_out()
    bin_df = pd.DataFrame(binary_X.toarray(), columns=binary_terms)

    word1 = st.selectbox("PMI Word 1", options=binary_terms, key='pmi1')
    word2 = st.selectbox("PMI Word 2", options=binary_terms, key='pmi2')

    if word1 and word2 and word1 != word2:
        if st.button("Calculate PMI"):
            try:
                col1 = binary_X[:, bin_vectorizer.vocabulary_[word1]].toarray().flatten()
                col2 = binary_X[:, bin_vectorizer.vocabulary_[word2]].toarray().flatten()
                co_occur = ((col1 > 0) & (col2 > 0)).sum()
                p1 = (col1 > 0).sum() / len(corpus)
                p2 = (col2 > 0).sum() / len(corpus)
                p12 = co_occur / len(corpus)

                st.write(f"p({word1}) = {p1:.4f}")
                st.write(f"p({word2}) = {p2:.4f}")
                st.write(f"p({word1}, {word2}) = {p12:.4f}")

                if p1 > 0 and p2 > 0 and p12 > 0:
                    pmi_score = round(log2(p12 / (p1 * p2 + 1e-10)), 4)
                    st.success(f"PMI({word1}, {word2}) = {pmi_score}")
                else:
                    st.warning("One or both words do not appear together in any document.")
            except Exception as e:
                st.error(f"Error calculating PMI: {e}")
else:
    st.info("Please input a valid corpus to proceed.")