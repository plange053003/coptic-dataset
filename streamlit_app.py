import streamlit as st
import pandas as pd
import re
from collections import defaultdict
from gensim.models import Word2Vec

# Load model (adjust path if needed)
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format("models/coptic_vectors_v2.vec", binary=False)


st.set_page_config(page_title="Coptic Word Explorer", layout="wide")

# ---------- Load CSV ----------
@st.cache_data
def load_data():
    df = pd.read_csv("data/copticmark.csv")
    df["Chapter"] = df["Chapter"].astype(str).str.zfill(2)
    return df

df = load_data()

# ---------- Session State Setup ----------
if "chapter_idx" not in st.session_state:
    st.session_state.chapter_idx = 0
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "Coptic"
if "selected_word" not in st.session_state:
    st.session_state.selected_word = None
if "selected_sentence" not in st.session_state:
    st.session_state.selected_sentence = ("", "")

# ---------- Chapter Selection ----------
chapters = sorted(df["Chapter"].unique().tolist())  # already zfilled and string
current_chapter = chapters[st.session_state.chapter_idx]
chapter_data = df[df["Chapter"] == current_chapter]

# ---------- UI Layout ----------
st.title("📜 Coptic Gospel Explorer")
st.markdown("## 📖 Coptic Gospel of Mark")
st.markdown(f"### Chapter {current_chapter}")

# --- Language Switch Centered ---
col_switch = st.columns([1, 1, 1])
with col_switch[1]:
    if st.button("🌍 Switch Language"):
        st.session_state.view_mode = (
            "English" if st.session_state.view_mode == "Coptic" else "Coptic"
        )
        st.session_state.selected_word = None

# --- Chapter Navigation Buttons ---
col_nav = st.columns([1, 1])
with col_nav[0]:
    if st.button("⬅️ Previous Chapter") and st.session_state.chapter_idx > 0:
        st.session_state.chapter_idx -= 1
        st.session_state.selected_word = None
with col_nav[1]:
    if st.button("➡️ Next Chapter") and st.session_state.chapter_idx < len(chapters) - 1:
        st.session_state.chapter_idx += 1
        st.session_state.selected_word = None

# ---------- Sentence Display Loop ----------
for idx, row in chapter_data.iterrows():
    coptic_block = row["Text"]
    english_block = row["Translation"]

    coptic_sents = [s.strip() for s in re.split(r"[.?!]", coptic_block) if s.strip()]
    english_sents = [s.strip() for s in re.split(r"[.?!]", english_block) if s.strip()]

    for sent_idx, (cop_sent, eng_sent) in enumerate(zip(coptic_sents, english_sents)):
        st.markdown("---")
        st.markdown(f"#### Sentence {sent_idx + 1}")

        sentence = cop_sent if st.session_state.view_mode == "Coptic" else eng_sent
        st.markdown(sentence)

        words = sentence.split()
        cols = st.columns(min(8, len(words)))
        for i, word in enumerate(words):
            with cols[i % 8]:
                if st.button(word, key=f"{idx}_{sent_idx}_{i}_{word}"):
                    st.session_state.selected_word = word
                    st.session_state.selected_sentence = (cop_sent, eng_sent)

# ---------- SIDEBAR DISPLAY ----------
if st.session_state.selected_word:
    word = st.session_state.selected_word
    cop_sent, eng_sent = st.session_state.selected_sentence

    st.sidebar.markdown(f"### 📌 Selected Word: `{word}`")
    st.sidebar.markdown("#### 🧠 Translation in Context")

    if st.session_state.view_mode == "Coptic":
        highlighted = cop_sent.replace(word, f"**{word}**")
        st.sidebar.markdown(f"**Coptic:** {highlighted}")
        st.sidebar.markdown(f"**English:** {eng_sent}")
    else:
        highlighted = eng_sent.replace(word, f"**{word}**")
        st.sidebar.markdown(f"**English:** {highlighted}")
        st.sidebar.markdown(f"**Coptic:** {cop_sent}")

    # ---------- CONTEXT TABLE IN SIDEBAR ----------
    if st.session_state.view_mode == "Coptic":
        # Build sentence context cache if not already present
        if "context_sentences" not in st.session_state:
            from collections import defaultdict
            context_sentences = defaultdict(list)
            for _, row in df.iterrows():
                coptic_sents = re.split(r"[.?!]", str(row["Text"]))
                english_sents = re.split(r"[.?!]", str(row["Translation"]))
                for c_sent, e_sent in zip(coptic_sents, english_sents):
                    c_sent = c_sent.strip()
                    e_sent = e_sent.strip()
                    for w in re.findall(r"\b\w+\b", c_sent):
                        context_sentences[w].append((c_sent, e_sent))
            st.session_state.context_sentences = context_sentences
        else:
            context_sentences = st.session_state.context_sentences

        # Grab all matched sentences
        matches = context_sentences.get(word, [])
        if matches:
            context_df = pd.DataFrame(matches, columns=["Coptic Sentence", "English Translation"])
            context_df["Coptic Sentence"] = context_df["Coptic Sentence"].apply(
                lambda s: s.replace(word, f"**{word}**")
            )

            with st.sidebar.expander(f"📖 All `{word}` Sentences"):
                st.markdown("Coptic word usage across all chapters:")
                st.dataframe(context_df, use_container_width=True)
        else:
            st.sidebar.markdown("_No other sentences found with this word._")

        # ---------- COSINE SIMILARITY BLOCK ----------
        st.sidebar.markdown("#### 🔁 Similar Coptic Words")
        if word in model:
            similar_words = model.most_similar(word, topn=5)
            for sim_word, score in similar_words:
                st.sidebar.markdown(f"- `{sim_word}` (score: {score:.2f})")
        else:
            st.sidebar.markdown("_This word is not in the model vocabulary._")
