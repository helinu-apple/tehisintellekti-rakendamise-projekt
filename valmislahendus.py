import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Kursuse Nõustaja", page_icon="🎓", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg:        #0d0f1a;
    --surface:   #151828;
    --border:    #2a2f4a;
    --accent1:   #6c63ff;
    --accent2:   #ff6b9d;
    --accent3:   #00e5c3;
    --text:      #e8eaf6;
    --muted:     #8b90b8;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif;
}

/* Title */
h1 { 
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, var(--accent1), var(--accent2), var(--accent3));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.4rem !important;
    letter-spacing: -1px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Filter group label */
.filter-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin: 12px 0 6px 0;
}

/* Toggle buttons (we fake them with st.button + session state) */
div[data-testid="stHorizontalBlock"] button {
    border-radius: 20px !important;
    border: 1px solid var(--border) !important;
    background: transparent !important;
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    transition: all 0.2s !important;
    padding: 4px 14px !important;
}

div[data-testid="stHorizontalBlock"] button:hover {
    border-color: var(--accent1) !important;
    color: var(--accent1) !important;
}

/* Active button hack via primary */
div[data-testid="stHorizontalBlock"] button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent1), var(--accent2)) !important;
    border-color: transparent !important;
    color: white !important;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: #ffffff !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: 8px !important;
    color: #000000 !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: #ffffff !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: #000000 !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent1) !important;
    box-shadow: 0 0 0 2px rgba(108,99,255,0.3) !important;
}

/* Password input */
input[type="password"], input[type="text"] {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* Info boxes */
[data-testid="stInfo"] {
    background: rgba(108,99,255,0.1) !important;
    border-left: 3px solid var(--accent1) !important;
    border-radius: 0 8px 8px 0 !important;
    font-size: 0.8rem;
}

/* Spinner */
[data-testid="stSpinner"] { color: var(--accent3) !important; }

/* Active filter badge */
.active-filters {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 8px 0;
}
.badge {
    background: rgba(108,99,255,0.15);
    border: 1px solid var(--accent1);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    color: var(--accent1);
    font-family: 'Space Mono', monospace;
}

/* Warning */
[data-testid="stAlert"] {
    background: rgba(255,107,157,0.1) !important;
    border-left: 3px solid var(--accent2) !important;
    border-radius: 0 8px 8px 0 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
def _init(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

_init("messages", [])
_init("sem_kevad", False)
_init("sem_sugis", False)
_init("eap_lt3", False)
_init("eap_lt6", False)
_init("eap_6plus", False)
_init("hind_eristav", False)
_init("hind_eristamata", False)
_init("total_tokens_in", 0)
_init("total_tokens_out", 0)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Seaded")
    api_key = st.text_input("OpenRouter API Key", type="password")

    st.markdown('<div class="filter-label">Semester</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🌸 Kevad",
                     type="primary" if st.session_state.sem_kevad else "secondary",
                     use_container_width=True):
            st.session_state.sem_kevad = not st.session_state.sem_kevad
    with c2:
        if st.button("🍂 Sügis",
                     type="primary" if st.session_state.sem_sugis else "secondary",
                     use_container_width=True):
            st.session_state.sem_sugis = not st.session_state.sem_sugis

    st.markdown('<div class="filter-label">EAP maht</div>', unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    with e1:
        if st.button("<3",
                     type="primary" if st.session_state.eap_lt3 else "secondary",
                     use_container_width=True):
            st.session_state.eap_lt3 = not st.session_state.eap_lt3
    with e2:
        if st.button("<6",
                     type="primary" if st.session_state.eap_lt6 else "secondary",
                     use_container_width=True):
            st.session_state.eap_lt6 = not st.session_state.eap_lt6
    with e3:
        if st.button("6+",
                     type="primary" if st.session_state.eap_6plus else "secondary",
                     use_container_width=True):
            st.session_state.eap_6plus = not st.session_state.eap_6plus

    st.markdown('<div class="filter-label">Hindamine</div>', unsafe_allow_html=True)
    h1, h2 = st.columns(2)
    with h1:
        if st.button("Eristav",
                     type="primary" if st.session_state.hind_eristav else "secondary",
                     use_container_width=True):
            st.session_state.hind_eristav = not st.session_state.hind_eristav
    with h2:
        if st.button("Eristamata",
                     type="primary" if st.session_state.hind_eristamata else "secondary",
                     use_container_width=True):
            st.session_state.hind_eristamata = not st.session_state.hind_eristamata

    st.divider()

    # Show active filters summary
    active = []
    if st.session_state.sem_kevad: active.append("Kevad")
    if st.session_state.sem_sugis: active.append("Sügis")
    if st.session_state.eap_lt3:  active.append("EAP <3")
    if st.session_state.eap_lt6:  active.append("EAP <6")
    if st.session_state.eap_6plus: active.append("EAP 6+")
    if st.session_state.hind_eristav: active.append("Eristav")
    if st.session_state.hind_eristamata: active.append("Eristamata")

    if active:
        badges = "".join(f'<span class="badge">{f}</span>' for f in active)
        st.markdown(f'<div class="active-filters">{badges}</div>', unsafe_allow_html=True)
    else:
        st.info("Filtrid pole valitud – otsitakse kõikide kursuste hulgast.")

    if st.button("🗑️ Tühjenda vestlus", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_tokens_in = 0
        st.session_state.total_tokens_out = 0
        st.rerun()

    # Token counter
    st.divider()
    st.markdown('<div class="filter-label">Tokeni kasutus</div>', unsafe_allow_html=True)
    total_tok = st.session_state.total_tokens_in + st.session_state.total_tokens_out
    # Approximate cost: OpenRouter gemma-3-27b ~$0.10/M input, $0.20/M output
    cost_usd = (st.session_state.total_tokens_in * 0.10 + st.session_state.total_tokens_out * 0.20) / 1_000_000
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("📥 Sisend", f"{st.session_state.total_tokens_in:,}")
        st.metric("📤 Väljund", f"{st.session_state.total_tokens_out:,}")
    with col_b:
        st.metric("∑ Kokku", f"{total_tok:,}")
        st.metric("💰 ~Hind", f"${cost_usd:.5f}")
    st.caption("Hinnad: $0.10/M sisend · $0.20/M väljund (gemma-3-27b-it)")

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🎓 AI Kursuse Nõustaja")
st.caption("RAG süsteem koos eel-filtreerimisega · Vali filtrid vasakult ja küsi julgelt!")

# ── Load models & data ────────────────────────────────────────────────────────
@st.cache_resource
def get_models():
    embedder = SentenceTransformer("BAAI/bge-m3")
    df = pd.read_csv("puhtad_andmed.csv")
    embeddings_df = pd.read_pickle("puhtad_andmed_embeddings.pkl")
    return embedder, df, embeddings_df

embedder, df, embeddings_df = get_models()

# ── Build filter mask from session state ─────────────────────────────────────
def build_mask(merged_df):
    mask = pd.Series([True] * len(merged_df), index=merged_df.index)

    # Semester filter (OR between selected)
    sem_vals = []
    if st.session_state.sem_kevad: sem_vals.append("kevad")
    if st.session_state.sem_sugis: sem_vals.append("sügis")
    if sem_vals:
        mask &= merged_df["semester"].isin(sem_vals)

    # EAP filter (OR between selected ranges)
    eap_masks = []
    if st.session_state.eap_lt3:   eap_masks.append(merged_df["eap"] < 3)
    if st.session_state.eap_lt6:   eap_masks.append(merged_df["eap"] < 6)
    if st.session_state.eap_6plus: eap_masks.append(merged_df["eap"] >= 6)
    if eap_masks:
        combined = eap_masks[0]
        for m in eap_masks[1:]:
            combined |= m
        mask &= combined

    # Hindamine filter (OR between selected)
    hind_vals = []
    if st.session_state.hind_eristav: hind_vals.append("Eristav")
    if st.session_state.hind_eristamata: hind_vals.append("Eristamata")
    if hind_vals:
        mask &= merged_df["hindamisviis"].isin(hind_vals)

    return mask

# ── Chat history ──────────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Kirjelda, mida soovid õppida..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not api_key:
            err = "⚠️ Palun sisesta API võti külgribal!"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
        else:
            with st.spinner("🔍 Otsin sobivaid kursusi..."):
                merged_df = pd.merge(df, embeddings_df, on="unique_ID")
                mask = build_mask(merged_df)
                filtered_df = merged_df[mask].copy()

                if filtered_df.empty:
                    context_text = "Valitud filtritele vastavaid kursusi ei leitud. Palun laienda filtreid."
                    st.warning("Ühtegi kursust ei vasta valitud filtritele.")
                else:
                    query_vec = embedder.encode([prompt])[0]
                    filtered_df["score"] = cosine_similarity(
                        [query_vec], np.stack(filtered_df["embedding"])
                    )[0]
                    results_df = filtered_df.sort_values("score", ascending=False).head(5)
                    results_df = results_df.drop(["score", "embedding"], axis=1)
                    context_text = results_df.to_string()

            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
            system_prompt = {
                "role": "system",
                "content": (
                    "Oled ülikooli kursuste nõustaja. Kasuta ainult järgmisi kursusi vastuse koostamisel. "
                    "Põhjenda oma soovitusi lühidalt ja sõbralikult.\n\n"
                    "Vasta ALATI järgmises formaadis – mitte rohkem kui 5 kursust, iga kursus uuel real:\n\n"
                    "Iga kursus PEAB olema eraldi real. Kasuta tühja rida kursuste vahel:\n\n"
                    "• **Kursuse nimi** (EAP, semester) – märksõna1, märksõna2, märksõna3\n\n"
                    "• **Teine kursus** (EAP, semester) – märksõna1, märksõna2\n\n"
                    f"KURSUSED:\n{context_text}"
                ),
            }

            messages_to_send = [system_prompt] + st.session_state.messages

            try:
                stream = client.chat.completions.create(
                    model="google/gemma-3-27b-it",
                    messages=messages_to_send,
                    stream=True,
                    stream_options={"include_usage": True},
                )

                # Manually stream so we can capture the final usage chunk
                response_placeholder = st.empty()
                full_response = ""
                usage_data = None
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        response_placeholder.markdown(full_response + "▌")
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage_data = chunk.usage
                response_placeholder.markdown(full_response)

                st.session_state.messages.append({"role": "assistant", "content": full_response})

                if usage_data:
                    st.session_state.total_tokens_in += usage_data.prompt_tokens or 0
                    st.session_state.total_tokens_out += usage_data.completion_tokens or 0
            except Exception as e:
                st.error(f"Viga API kutsel: {e}")