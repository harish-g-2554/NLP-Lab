import streamlit as st
import re
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.metrics.distance import edit_distance
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk import pos_tag, word_tokenize
import plotly.express as px
import plotly.graph_objects as go

# Set up the main app
st.set_page_config(
    page_title="NLP Programs Suite",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# --- Program 1: Positional Index ---
if program == "Program 1: Positional Index":
    st.header("📍 Program 1: Positional Index Builder")
    st.markdown("Build a positional index showing word positions in documents.")
    
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
        
        query_words_input = st.text_input("Enter words to query (comma-separated):", 
                                          value="student, mds472c")
        query_words = [w.strip().lower() for w in query_words_input.split(',') if w.strip()]
        
        for word in query_words:
            if word in pos_index:
                st.write(f"**'{word}'**: {dict(pos_index[word])}")
            else:
                st.write(f"**'{word}'**: Not found")

# --- Program 2: Word Matrix ---
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
        doc1_words = set(re.sub(r'[^\w\s]', '', doc1).lower().split())
        doc2_words = set(re.sub(r'[^\w\s]', '', doc2).lower().split())
        vocabulary = sorted(list(doc1_words.union(doc2_words)))
        
        word_matrix_data = {
            'Doc1': [1 if word in doc1_words else 0 for word in vocabulary],
            'Doc2': [1 if word in doc2_words else 0 for word in vocabulary]
        }
        
        df_word_matrix = pd.DataFrame(word_matrix_data, index=vocabulary).T
        
        st.subheader("Word Matrix:")
        st.dataframe(df_word_matrix, use_container_width=True)
        
        fig = px.imshow(df_word_matrix.values, 
                        x=df_word_matrix.columns, 
                        y=df_word_matrix.index,
                        aspect="auto",
                        color_continuous_scale="Blues",
                        title="Document-Term Matrix Heatmap")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# --- Program 3: Text Preprocessing & Analysis ---
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
        
        processed = [full_preprocess(doc) for doc in [text1, text2]]
        freq_index = [Counter(doc) for doc in processed]
        
        st.subheader("Frequency Index:")
        for i, counter in enumerate(freq_index, 1):
            st.write(f"**Doc{i}**: {dict(counter)}")
        
        all_words = sum(processed, [])
        word_freq = Counter(all_words)
        sorted_words = word_freq.most_common()
        
        st.subheader("Words by Frequency:")
        freq_df = pd.DataFrame(sorted_words, columns=['Word', 'Frequency'])
        st.dataframe(freq_df, use_container_width=True)
        
        top_10 = sorted_words[:10]
        if top_10:
            words, freqs = zip(*top_10)
            fig = px.bar(x=words, y=freqs, title="Top 10 Most Frequent Words")
            fig.update_layout(xaxis_title="Words", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)

# --- Program 4: Edit Distance & Alignment ---
elif program == "Program 4: Edit Distance & Alignment":
    st.header("📏 Program 4: Edit Distance & Sequence Alignment with Backtracking")
    st.markdown("Calculate Levenshtein distance with detailed alignment visualization and backtracking path.")
    
    col1, col2 = st.columns(2)
    with col1:
        word_a = st.text_input("Word A:", value="characterization", key="align_a")
    with col2:
        word_b = st.text_input("Word B:", value="categorization", key="align_b")
    
    show_backtrack = st.checkbox("Show Backtracking Path", value=True)
    show_step_by_step = st.checkbox("Show Step-by-Step Backtracking", value=False)
    
    if st.button("Calculate Alignment", key="alignment") and word_a and word_b:
        def full_levenshtein_with_backtrack(wordA, wordB):
            lenA, lenB = len(wordA), len(wordB)
            dp = np.zeros((lenA + 1, lenB + 1), dtype=int)
            for i in range(lenA + 1): dp[i][0] = i
            for j in range(lenB + 1): dp[0][j] = j
            for i in range(1, lenA + 1):
                for j in range(1, lenB + 1):
                    cost = 0 if wordA[i - 1] == wordB[j - 1] else 1
                    dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

            i, j = lenA, lenB
            path = [(i, j)]
            ops, alignedA, alignedB, op_seq = [], [], [], []
            ins, dels, subs, matches = 0, 0, 0, 0
            backtrack_steps = []

            while i > 0 or j > 0:
                current_cost = dp[i][j]
                step_info = {'position': (i, j), 'cost': current_cost, 'operation': '', 'from_position': None, 'explanation': ''}
                
                cost_diag = dp[i-1][j-1] if i > 0 and j > 0 else float('inf')
                cost_up = dp[i-1][j] if i > 0 else float('inf')
                cost_left = dp[i][j-1] if j > 0 else float('inf')
                
                is_match_sub = (i > 0 and j > 0 and current_cost == cost_diag + (wordA[i-1] != wordB[j-1]))

                if is_match_sub:
                    alignedA.append(wordA[i-1])
                    alignedB.append(wordB[j-1])
                    if wordA[i-1] == wordB[j-1]:
                        op_seq.append('|') # Using pipe for match for alignment
                        ops.append(f"Match: {wordA[i-1]}")
                        step_info.update({'operation': 'Match', 'explanation': f"'{wordA[i-1]}' matches '{wordB[j-1]}'"})
                        matches += 1
                    else:
                        op_seq.append('s')
                        ops.append(f"Substitute: {wordA[i-1]} → {wordB[j-1]}")
                        step_info.update({'operation': 'Substitute', 'explanation': f"'{wordA[i-1]}' → '{wordB[j-1]}'"})
                        substitutions += 1
                    step_info['from_position'] = (i-1, j-1)
                    i, j = i-1, j-1
                elif i > 0 and current_cost == cost_up + 1:
                    alignedA.append(wordA[i-1])
                    alignedB.append('-')
                    op_seq.append('d')
                    ops.append(f"Delete: {wordA[i-1]}")
                    step_info.update({'operation': 'Delete', 'explanation': f"Delete '{wordA[i-1]}' from Word A"})
                    step_info['from_position'] = (i-1, j)
                    deletions += 1
                    i -= 1
                else:
                    alignedA.append('-')
                    alignedB.append(wordB[j-1])
                    op_seq.append('i')
                    ops.append(f"Insert: {wordB[j-1]}")
                    step_info.update({'operation': 'Insert', 'explanation': f"Insert '{wordB[j-1]}' into Word A"})
                    step_info['from_position'] = (i, j-1)
                    insertions += 1
                    j -= 1
                path.append((i,j))
                backtrack_steps.append(step_info)
            
            path.reverse(); ops.reverse(); alignedA.reverse(); alignedB.reverse(); op_seq.reverse(); backtrack_steps.reverse()
            return dp, ops, alignedA, alignedB, op_seq, ins, dels, subs, matches, path, backtrack_steps

        dp, ops, alignedA, alignedB, op_seq, ins, dels, subs, matches, path, backtrack_steps = full_levenshtein_with_backtrack(word_a, word_b)
        
        tab1, tab2, tab3 = st.tabs(["🎯 Results", "🔄 Backtracking", "📊 Matrix Visualization"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Alignment Results:")
                st.code(f"Word A    : {''.join(alignedA)}\nWord B    : {''.join(alignedB)}\nOperations: {''.join(op_seq)}")
                st.subheader("Summary Statistics:")
                summary_df = pd.DataFrame({
                    'Operation': ['Total Edit Distance', 'Matches', 'Substitutions', 'Insertions', 'Deletions'],
                    'Count': [dp[-1][-1], matches, subs, ins, dels]
                })
                st.dataframe(summary_df, use_container_width=True)
            with col2:
                st.subheader("Step-by-Step Operations:")
                for i, op in enumerate(ops, 1):
                    st.write(f"{i:2d}. {op}")
        
        with tab2:
            st.subheader("🔄 Backtracking Path Analysis")
            if show_step_by_step:
                st.markdown("**Detailed Backtracking Steps:**")
                for i, step in enumerate(backtrack_steps, 1):
                    if step['from_position']:
                        st.markdown(f"**Step {i}:** Position `{step['position']}` → `{step['from_position']}`\n"
                                    f"- **Operation:** {step['operation']}\n"
                                    f"- **Explanation:** {step['explanation']}\n"
                                    f"- **Cost:** {step['cost']}\n"
                                    "---")
            st.subheader("Backtracking Path Coordinates:")
            path_df = pd.DataFrame(path, columns=['Row (Word A)', 'Col (Word B)'])
            path_df['Step'] = range(len(path))
            st.dataframe(path_df, use_container_width=True)
        
        with tab3:
            st.subheader("📊 Dynamic Programming Matrix with Backtracking Path")
            
            fig = go.Figure()

            fig.add_trace(go.Heatmap(
                z=dp,
                x=list(" " + word_b),
                y=list(" " + word_a),
                colorscale='Blues',
                showscale=True,
                name='',
                colorbar=dict(title='Cost')
            ))

            annotations = []
            for i in range(len(word_a) + 1):
                for j in range(len(word_b) + 1):
                    max_cost = np.max(dp)
                    text_color = "white" if dp[i][j] > max_cost / 2 else "black"
                    annotations.append(go.layout.Annotation(
                        text=str(dp[i][j]),
                        x=j, y=i,
                        xref='x1', yref='y1',
                        showarrow=False,
                        font=dict(color=text_color, size=11, family="monospace")
                    ))
            
            if show_backtrack and path:
                path_y, path_x = zip(*path)

                fig.add_trace(go.Scatter(
                    x=path_x,
                    y=path_y,
                    mode='lines+markers',
                    name='Backtracking Path',
                    line=dict(color='crimson', width=3),
                    marker=dict(
                        color='crimson', 
                        size=9,
                        symbol='circle',
                        line=dict(width=1, color='white')
                    ),
                    showlegend=True
                ))

                for i in range(len(path) - 1):
                    start_point, end_point = path[i], path[i+1]
                    annotations.append(go.layout.Annotation(
                        ax=start_point[1], ay=start_point[0],
                        x=end_point[1], y=end_point[0],
                        xref='x1', yref='y1', axref='x1', ayref='y1',
                        showarrow=True, arrowhead=2, arrowsize=1.5,
                        arrowwidth=2, arrowcolor='crimson'
                    ))

            fig.update_layout(
                title_text=f"<b>Levenshtein Alignment: '{word_a}' vs '{word_b}'</b>",
                xaxis_title=f"Word B: {word_b}",
                yaxis_title=f"Word A: {word_a}",
                yaxis=dict(autorange="reversed"),
                xaxis=dict(side='top'),
                height=650,
                showlegend=True,
                legend=dict(x=0.01, y=0.98, bgcolor='rgba(255,255,255,0.7)'),
                annotations=annotations,
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Matrix Statistics:")
            c1, c2, c3 = st.columns(3)
            c1.metric("Matrix Size", f"{len(word_a)+1} × {len(word_b)+1}")
            c2.metric("Path Length", len(path))
            c3.metric("Final Cost", dp[-1][-1])

# --- Program 5: POS Tagging (Viterbi) ---
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
        st.write(f"**Sentence {i}**: {' '.join([f'{w}/{t}' for w, t in sentence])}")
    
    test_sentence = st.text_input("Enter test sentence:", 
                                  value="The rat can chase the cat",
                                  key="pos_test")
    
    if st.button("Tag Sentence", key="pos_tag") and test_sentence:
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

        def get_prob(prob_dict, key, subkey, default=1e-6):
            return prob_dict.get(key, {}).get(subkey, default)

        tags = list(tag_counts.keys())
        total_tags = sum(tag_counts.values())

        def viterbi(sentence_tokens, tags, transition_counts, emission_counts, tag_counts):
            viterbi_matrix = [{}]
            backpointer = [{}]
            
            # Initialization step
            for tag in tags:
                emission_prob = (emission_counts[tag].get(sentence_tokens[0], 0) + 1) / (tag_counts[tag] + len(tag_counts))
                start_prob = (tag_counts[tag] + 1) / (total_tags + len(tags)) # Simple starting prob
                viterbi_matrix[0][tag] = start_prob * emission_prob
                backpointer[0][tag] = None

            # Recursion step
            for t in range(1, len(sentence_tokens)):
                viterbi_matrix.append({})
                backpointer.append({})
                for tag in tags:
                    emission_prob = (emission_counts[tag].get(sentence_tokens[t], 0) + 1) / (tag_counts[tag] + len(tag_counts))
                    max_prob, best_prev_tag = max(
                        (viterbi_matrix[t-1][prev_tag] * ((transition_counts[prev_tag].get(tag, 0) + 1) / (tag_counts[prev_tag] + len(tags))), prev_tag)
                        for prev_tag in tags
                    )
                    viterbi_matrix[t][tag] = max_prob * emission_prob
                    backpointer[t][tag] = best_prev_tag

            # Termination step
            best_path_prob = max(viterbi_matrix[-1].values())
            best_last_tag = max(viterbi_matrix[-1], key=viterbi_matrix[-1].get)
            
            best_path = [best_last_tag]
            for t in range(len(viterbi_matrix) - 1, 0, -1):
                best_path.insert(0, backpointer[t][best_path[0]])

            return best_path

        test_words = test_sentence.lower().split()
        predicted_tags = viterbi(test_words, tags, transition_counts, emission_counts, tag_counts)
        
        st.subheader("Tagged Result:")
        tagged_result = " ".join([f"{w}/{t}" for w, t in zip(test_sentence.split(), predicted_tags)])
        st.success(tagged_result)


# --- Program 6: Word Sense Disambiguation ---
elif program == "Program 6: Word Sense Disambiguation":
    st.header("🎯 Program 6: Word Sense Disambiguation")
    st.markdown("Disambiguate word senses using the Lesk algorithm.")
    
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'): return wn.ADJ
        elif treebank_tag.startswith('V'): return wn.VERB
        elif treebank_tag.startswith('N'): return wn.NOUN
        elif treebank_tag.startswith('R'): return wn.ADV
        else: return None
    
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
        for sentence in corpus:
            st.markdown(f"--- \n### 📝 Sentence: `{sentence}`")
            tokens = word_tokenize(sentence)
            tagged = pos_tag(tokens)
            
            for word, tag in tagged:
                wn_pos = get_wordnet_pos(tag)
                if wn_pos:
                    senses = wn.synsets(word, pos=wn_pos)
                    if senses:
                        with st.expander(f"🔍 **{word}** ({tag}) - Found {len(senses)} sense(s)"):
                            lesk_sense = lesk(tokens, word, wn_pos)
                            
                            if lesk_sense:
                                st.success(f"**Predicted Sense (Lesk):** `{lesk_sense.name()}`")
                                st.info(f"**Definition:** {lesk_sense.definition()}")
                            else:
                                st.warning("No specific sense could be determined by Lesk.")
                            
                            st.markdown("**All Available Senses:**")
                            for i, s in enumerate(senses[:5], 1): # Show top 5
                                st.write(f"{i}. **{s.name()}**: {s.definition()}")


# --- Footer ---
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