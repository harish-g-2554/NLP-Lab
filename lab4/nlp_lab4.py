import streamlit as st
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.metrics.distance import edit_distance
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk import pos_tag, word_tokenize
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass

download_nltk_data()

# Set up the main app
st.set_page_config(
    page_title="NLP Programs Suite",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔤 Natural Language Processing Programs Suite")
st.markdown("---")

# Sidebar for program selection
st.sidebar.title("📋 Program Selection")
program = st.sidebar.selectbox(
    "Choose a program:",
    [
        "Program 1: Positional Index",
        "Program 2: Word Matrix",
        "Program 3: Text Preprocessing & Analysis",
        "Program 4: Edit Distance & Alignment",
        "Program 5: POS Tagging (Viterbi)",
        "Program 6: Word Sense Disambiguation"
    ]
)

# Program 1: Positional Index
if program == "Program 1: Positional Index":
    st.header("📍 Program 1: Positional Index Builder")
    st.markdown("Build a positional index showing word positions in documents.")
    
    # Default documents
    default_doc1 = "I am a student, and I currently take MDS472C. I was a student in MDS331 last trimester."
    default_doc2 = "I was a student. I have taken MDS472C."
    
    col1, col2 = st.columns(2)
    with col1:
        doc1 = st.text_area("Document 1:", value=default_doc1, height=100)
    with col2:
        doc2 = st.text_area("Document 2:", value=default_doc2, height=100)
    
    if st.button("Build Positional Index", key="pos_index"):
        documents = [doc1, doc2]
        
        def tokenize(text):
            return re.findall(r'\w+', text.lower())

        def build_positional_index(docs):
            index = defaultdict(lambda: defaultdict(list))
            for doc_id, doc in enumerate(docs, 1):
                tokens = tokenize(doc)
                for pos, word in enumerate(tokens):
                    index[word][f"Doc{doc_id}"].append(pos)
            return index

        pos_index = build_positional_index(documents)
        
        st.subheader("Positional Index Results:")
        
        # Query specific words
        query_words = st.text_input("Enter words to query (comma-separated):", 
                                  value="student, MDS472C").split(",")
        query_words = [w.strip().lower() for w in query_words if w.strip()]
        
        for word in query_words:
            if word in pos_index:
                st.write(f"**'{word}'**: {dict(pos_index[word])}")
            else:
                st.write(f"**'{word}'**: Not found")

# Program 2: Word Matrix
elif program == "Program 2: Word Matrix":
    st.header("📊 Program 2: Document-Term Matrix")
    st.markdown("Create a binary word matrix showing word presence in documents.")
    
    default_doc1 = "I am a student, and I currently take MDS472C. I was a student in MDS331 last trimester."
    default_doc2 = "I was a student. I have taken MDS472C."
    
    col1, col2 = st.columns(2)
    with col1:
        doc1 = st.text_area("Document 1:", value=default_doc1, height=100, key="wm_doc1")
    with col2:
        doc2 = st.text_area("Document 2:", value=default_doc2, height=100, key="wm_doc2")
    
    if st.button("Generate Word Matrix", key="word_matrix"):
        documents = [doc1, doc2]
        
        doc1_words = set(re.sub(r'[^\w\s]', '', doc1).lower().split())
        doc2_words = set(re.sub(r'[^\w\s]', '', doc2).lower().split())
        vocabulary = sorted(list(doc1_words.union(doc2_words)))
        
        word_matrix_data = {}
        word_matrix_data['Doc1'] = [1 if word in doc1_words else 0 for word in vocabulary]
        word_matrix_data['Doc2'] = [1 if word in doc2_words else 0 for word in vocabulary]
        
        df_word_matrix = pd.DataFrame(word_matrix_data, index=vocabulary).T
        
        st.subheader("Word Matrix:")
        st.dataframe(df_word_matrix, use_container_width=True)
        
        # Visualization
        fig = px.imshow(df_word_matrix.values, 
                       x=df_word_matrix.columns, 
                       y=df_word_matrix.index,
                       aspect="auto",
                       color_continuous_scale="Blues",
                       title="Document-Term Matrix Heatmap")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# Program 3: Text Preprocessing & Analysis
elif program == "Program 3: Text Preprocessing & Analysis":
    st.header("🔧 Program 3: Text Preprocessing & Analysis")
    st.markdown("Perform stemming, lemmatization, and frequency analysis.")
    
    default_text1 = "Machine learning allows systems to learn from data."
    default_text2 = "Learning from examples is a powerful method in data science."
    
    col1, col2 = st.columns(2)
    with col1:
        text1 = st.text_area("Document 1:", value=default_text1, height=100, key="prep_doc1")
    with col2:
        text2 = st.text_area("Document 2:", value=default_text2, height=100, key="prep_doc2")
    
    if st.button("Analyze Texts", key="preprocess"):
        def tokenize(text):
            return re.findall(r'\w+', text.lower())
        
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        
        def full_preprocess(text):
            tokens = tokenize(text)
            normalized = [lemmatizer.lemmatize(stemmer.stem(token)) for token in tokens]
            return normalized
        
        corpus = [text1, text2]
        processed = [full_preprocess(doc) for doc in corpus]
        
        # Frequency index
        freq_index = [Counter(doc) for doc in processed]
        
        st.subheader("Frequency Index:")
        for i, counter in enumerate(freq_index, 1):
            st.write(f"**Doc{i}**: {dict(counter)}")
        
        # Sorted words by frequency
        all_words = sum(processed, [])
        word_freq = Counter(all_words)
        sorted_words = word_freq.most_common()
        
        st.subheader("Words by Frequency:")
        freq_df = pd.DataFrame(sorted_words, columns=['Word', 'Frequency'])
        st.dataframe(freq_df, use_container_width=True)
        
        # Visualization
        top_10 = sorted_words[:10]
        if top_10:
            words, freqs = zip(*top_10)
            fig = px.bar(x=words, y=freqs, title="Top 10 Most Frequent Words")
            fig.update_layout(xaxis_title="Words", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
        
        # Edit distance
        st.subheader("Edit Distance Example:")
        w1 = st.text_input("Word 1:", value="learn", key="edit_w1")
        w2 = st.text_input("Word 2:", value="learning", key="edit_w2")
        
        if w1 and w2:
            distance = edit_distance(w1, w2)
            st.write(f"Edit distance between '{w1}' and '{w2}': **{distance}**")

# Program 4: Edit Distance & Alignment
elif program == "Program 4: Edit Distance & Alignment":
    st.header("📏 Program 4: Edit Distance & Sequence Alignment")
    st.markdown("Calculate Levenshtein distance with detailed alignment visualization.")
    
    col1, col2 = st.columns(2)
    with col1:
        word_a = st.text_input("Word A:", value="characterization", key="align_a")
    with col2:
        word_b = st.text_input("Word B:", value="categorization", key="align_b")
    
    if st.button("Calculate Alignment", key="alignment") and word_a and word_b:
        def full_levenshtein_alignment(wordA, wordB):
            lenA, lenB = len(wordA), len(wordB)
            dp = np.zeros((lenA + 1, lenB + 1), dtype=int)

            # Initialize DP matrix
            for i in range(lenA + 1):
                dp[i][0] = i
            for j in range(lenB + 1):
                dp[0][j] = j

            # Fill the matrix
            for i in range(1, lenA + 1):
                for j in range(1, lenB + 1):
                    cost = 0 if wordA[i - 1] == wordB[j - 1] else 1
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,      # deletion
                        dp[i][j - 1] + 1,      # insertion
                        dp[i - 1][j - 1] + cost # substitution
                    )

            # Backtrace path
            i, j = lenA, lenB
            ops = []
            alignedA, alignedB, op_seq = [], [], []
            insertions = deletions = substitutions = matches = 0

            while i > 0 or j > 0:
                if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (wordA[i - 1] != wordB[j - 1]):
                    alignedA.append(wordA[i - 1])
                    alignedB.append(wordB[j - 1])
                    if wordA[i - 1] == wordB[j - 1]:
                        op_seq.append('*')
                        ops.append(f"Match: {wordA[i - 1]}")
                        matches += 1
                    else:
                        op_seq.append('s')
                        ops.append(f"Substitute: {wordA[i - 1]} → {wordB[j - 1]}")
                        substitutions += 1
                    i -= 1
                    j -= 1
                elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                    alignedA.append(wordA[i - 1])
                    alignedB.append('-')
                    op_seq.append('d')
                    ops.append(f"Delete: {wordA[i - 1]}")
                    deletions += 1
                    i -= 1
                else:
                    alignedA.append('-')
                    alignedB.append(wordB[j - 1])
                    op_seq.append('i')
                    ops.append(f"Insert: {wordB[j - 1]}")
                    insertions += 1
                    j -= 1

            ops.reverse()
            alignedA.reverse()
            alignedB.reverse()
            op_seq.reverse()

            return dp, ops, alignedA, alignedB, op_seq, insertions, deletions, substitutions, matches

        dp, ops, alignedA, alignedB, op_seq, ins, dels, subs, matches = full_levenshtein_alignment(word_a, word_b)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Alignment Results:")
            st.code(f"Word A  : {''.join(alignedA)}")
            st.code(f"Word B  : {''.join(alignedB)}")
            st.code(f"Operations: {''.join(op_seq)}")
            
            st.subheader("Summary:")
            st.write(f"**Total Edit Distance**: {dp[-1][-1]}")
            st.write(f"**Matches**: {matches}")
            st.write(f"**Substitutions**: {subs}")
            st.write(f"**Insertions**: {ins}")
            st.write(f"**Deletions**: {dels}")
        
        with col2:
            st.subheader("Step-by-Step Operations:")
            for i, op in enumerate(ops, 1):
                st.write(f"{i:2d}. {op}")
        
        # Visualization of DP matrix
        st.subheader("Dynamic Programming Matrix:")
        df_matrix = pd.DataFrame(dp, 
                                index=[" "] + list(word_a), 
                                columns=[" "] + list(word_b))
        
        fig = px.imshow(df_matrix.values, 
                       x=df_matrix.columns, 
                       y=df_matrix.index,
                       aspect="auto",
                       color_continuous_scale="YlOrRd",
                       title="Levenshtein Distance Matrix")
        
        # Add text annotations
        for i in range(len(df_matrix.index)):
            for j in range(len(df_matrix.columns)):
                fig.add_annotation(x=j, y=i, text=str(df_matrix.iloc[i, j]), 
                                 showarrow=False, font=dict(color="black", size=10))
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# Program 5: POS Tagging
elif program == "Program 5: POS Tagging (Viterbi)":
    st.header("🏷️ Program 5: POS Tagging using Viterbi Algorithm")
    st.markdown("Simple POS tagging implementation using the Viterbi algorithm.")
    
    st.subheader("Training Corpus (Pre-defined):")
    training_sentences = [
        [('The', 'DET'), ('cat', 'NOUN'), ('chased', 'VERB'), ('the', 'DET'), ('rat', 'NOUN')],
        [('A', 'DET'), ('rat', 'NOUN'), ('can', 'MODAL'), ('run', 'VERB')],
        [('The', 'DET'), ('dog', 'NOUN'), ('can', 'MODAL'), ('chase', 'VERB'), ('the', 'DET'), ('cat', 'NOUN')]
    ]
    
    for i, sentence in enumerate(training_sentences, 1):
        st.write(f"**Sentence {i}**: {' '.join([f'{word}/{tag}' for word, tag in sentence])}")
    
    test_sentence = st.text_input("Enter test sentence:", 
                                value="The rat can chase the cat",
                                key="pos_test")
    
    if st.button("Tag Sentence", key="pos_tag") and test_sentence:
        # Calculate probabilities
        transition_counts = defaultdict(Counter)
        emission_counts = defaultdict(Counter)
        tag_counts = Counter()

        for sentence in training_sentences:
            previous_tag = None
            for word, tag in sentence:
                tag_counts[tag] += 1
                emission_counts[tag][word.lower()] += 1
                if previous_tag:
                    transition_counts[previous_tag][tag] += 1
                previous_tag = tag

        def calculate_probabilities(counts_dict):
            probabilities = defaultdict(dict)
            for key in counts_dict:
                total = sum(counts_dict[key].values())
                for subkey in counts_dict[key]:
                    probabilities[key][subkey] = counts_dict[key][subkey] / total
            return probabilities

        transition_probs = calculate_probabilities(transition_counts)
        emission_probs = calculate_probabilities(emission_counts)
        tags = list(tag_counts.keys())

        def viterbi(sentence):
            sentence = [word.lower() for word in sentence]
            V = [{}]
            path = {}

            # Initialization
            for tag in tags:
                emission = emission_probs[tag].get(sentence[0], 1e-6)
                V[0][tag] = emission
                path[tag] = [tag]

            # Recursion
            for t in range(1, len(sentence)):
                V.append({})
                new_path = {}

                for curr_tag in tags:
                    (prob, prev_tag) = max(
                        (V[t-1][pt] * transition_probs[pt].get(curr_tag, 1e-6) * 
                         emission_probs[curr_tag].get(sentence[t], 1e-6), pt)
                        for pt in tags
                    )
                    V[t][curr_tag] = prob
                    new_path[curr_tag] = path[prev_tag] + [curr_tag]

                path = new_path

            # Termination
            (prob, final_tag) = max((V[-1][tag], tag) for tag in tags)
            return path[final_tag]

        test_words = test_sentence.split()
        predicted_tags = viterbi(test_words)
        
        st.subheader("Tagged Result:")
        tagged_result = []
        for word, tag in zip(test_words, predicted_tags):
            tagged_result.append(f"{word}/{tag}")
        
        st.write(" ".join(tagged_result))
        
        # Show probability tables
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Emission Probabilities:")
            emission_df = pd.DataFrame.from_dict(
                {tag: probs for tag, probs in emission_probs.items()}, 
                orient='index'
            ).fillna(0)
            st.dataframe(emission_df)
        
        with col2:
            st.subheader("Transition Probabilities:")
            transition_df = pd.DataFrame.from_dict(
                {tag: probs for tag, probs in transition_probs.items()}, 
                orient='index'
            ).fillna(0)
            st.dataframe(transition_df)

# Program 6: Word Sense Disambiguation
elif program == "Program 6: Word Sense Disambiguation":
    st.header("🎯 Program 6: Word Sense Disambiguation")
    st.markdown("Disambiguate word senses using the Lesk algorithm.")
    
    # Helper function
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return None
    
    # Default corpus
    default_corpus = [
        "The bank can guarantee deposits will eventually cover future tuition costs.",
        "He went to the bank to deposit his paycheck.",
        "The river overflowed the bank during the storm.",
        "They saw a bat flying in the dark.",
        "She held the bat tightly during the game."
    ]
    
    st.subheader("Text Corpus:")
    corpus_text = st.text_area("Enter sentences (one per line):", 
                              value="\n".join(default_corpus), 
                              height=150)
    
    corpus = [line.strip() for line in corpus_text.split('\n') if line.strip()]
    
    if st.button("Analyze Word Senses", key="wsd") and corpus:
        results = []
        
        for sentence_idx, sentence in enumerate(corpus):
            tokens = word_tokenize(sentence)
            tagged = pos_tag(tokens)
            
            sentence_results = {
                'sentence': sentence,
                'words': []
            }
            
            for word, tag in tagged:
                wn_pos = get_wordnet_pos(tag)
                if wn_pos:  # Open-class word
                    senses = wn.synsets(word, pos=wn_pos)
                    if senses:
                        # Apply Lesk algorithm
                        lesk_sense = lesk(tokens, word, wn_pos)
                        
                        word_result = {
                            'word': word,
                            'pos': tag,
                            'senses_count': len(senses),
                            'top_senses': [(s.name(), s.definition()) for s in senses[:3]],
                            'predicted_sense': (lesk_sense.name(), lesk_sense.definition()) if lesk_sense else None
                        }
                        sentence_results['words'].append(word_result)
            
            results.append(sentence_results)
        
        # Display results
        for result in results:
            st.subheader(f"📝 Sentence: {result['sentence']}")
            
            if result['words']:
                for word_info in result['words']:
                    with st.expander(f"🔍 {word_info['word']} ({word_info['pos']}) - {word_info['senses_count']} senses"):
                        st.write("**Available senses:**")
                        for i, (sense_name, definition) in enumerate(word_info['top_senses'], 1):
                            st.write(f"{i}. `{sense_name}` - {definition}")
                        
                        if word_info['predicted_sense']:
                            sense_name, definition = word_info['predicted_sense']
                            st.success(f"**Predicted sense (Lesk):** `{sense_name}` - {definition}")
                        else:
                            st.warning("No sense predicted by Lesk algorithm")
            else:
                st.info("No open-class words found for disambiguation.")
            
            st.markdown("---")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 About")
st.sidebar.markdown("""
This application demonstrates various NLP techniques:
- **Positional Indexing**: Document word position mapping
- **Word Matrix**: Binary document-term representation
- **Text Preprocessing**: Stemming, lemmatization, frequency analysis
- **Edit Distance**: Sequence alignment with Levenshtein distance
- **POS Tagging**: Part-of-speech tagging using Viterbi algorithm
- **WSD**: Word sense disambiguation using Lesk algorithm
""")

st.sidebar.markdown("---")
st.sidebar.markdown("*Built with Streamlit* 🚀")