import streamlit as st
import nltk
from nltk.corpus import words
from nltk.metrics import edit_distance
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import warnings
warnings.filterwarnings("ignore")

# Download corpus
nltk.download('words')
word_list = words.words()

# Title
st.title("NLP Dashboard: Edit Distance, Sequence Alignment & Spell Correction")

# --- Section 1: Edit Distance Calculator ---
st.header("1️⃣ Edit Distance Calculator")

str1 = st.text_input("Enter first word:")
str2 = st.text_input("Enter second word:")

if str1 and str2:
    distance = edit_distance(str1, str2)
    st.success(f"Edit distance between '{str1}' and '{str2}' is: **{distance}**")

# --- Section 2: Sequence Alignment ---
st.header("2️⃣ Sequence Alignment (Global)")

default_seqA = "AGGCTATCACCTGACCTCCAGGCCGATGCCC"
default_seqB = "TAGCTATCACGACCGCGGTCGATTTGCCCGAC"

textA = st.text_area("Enter Text A (Sequence 1):", default_seqA, height=100)
textB = st.text_area("Enter Text B (Sequence 2):", default_seqB, height=100)

if st.button("Align Sequences"):
    alignments = pairwise2.align.globalxx(textA, textB)
    best = alignments[0]
    st.subheader("Best Global Alignment:")
    st.code(format_alignment(*best), language='text')



# --- Section 3: Spell Correction ---
st.header("3️⃣ Spell Correction")

misspelled = st.text_input("Enter a misspelled word:")

def correct_word(word, vocab, max_suggestions=5):
    candidates = [w for w in vocab if abs(len(w) - len(word)) <= 2]
    distances = [(w, edit_distance(word, w)) for w in candidates]
    distances.sort(key=lambda x: x[1])
    return distances[:max_suggestions]

if misspelled:
    suggestions = correct_word(misspelled, word_list)
    st.subheader("Suggestions:")
    for word, dist in suggestions:
        st.write(f"🔹 {word} (edit distance = {dist})")
