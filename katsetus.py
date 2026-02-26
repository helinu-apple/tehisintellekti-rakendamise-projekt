import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="AI Kursuse Nõustaja", layout="wide")

# ── Custom CSS for toggle buttons ─────────────────────────────────────────────
st.markdown("""
<style>
/* Toggle button style */
div[data-testid="stButton"] > button {
    background-color: #ffffff;
    border: 2px solid #a4cff0;
    border-radius: 20px;
    color: #1a1a2e;
    font-size: 0.82rem;
    padding: 4px 12px;
    margin: 2px;
    transition: all 0.2s ease;
    width: 100%;
}
div[data-testid="stButton"] > button:hover {
    background-color: #a4cff0;
    color: #1a1a2e;
    border-color: #a4cff0;
}
/* Active / selected state  — add class via session state trick below */
button.selected-btn {
    background-color: #a4cff0 !important;
    color: #1a1a2e !important;
}

/* Token bar at bottom */
.token-bar {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    background: #f0f8ff;
    border-top: 1px solid #a4cff0;
    padding: 6px 20px;
    font-size: 0.78rem;
    color: #444;
    z-index: 9999;
    display: flex;
    gap: 24px;
    align-items: center;
}
.token-bar span { font-weight: 600; color: #1a6aa8; }
</style>
""", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🎓 AI Kursuse Nõustaja")
st.caption("RAG süsteem koos eel-filtreerimisega.")

# ── Session state defaults ────────────────────────────────────────────────────
for key, default in {
    "messages": [],
    "sel_semesters": [],
    "sel_hindamisviis": [],
    "sel_eap": [],
    "total_input_tokens": 0,
    "total_output_tokens": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Pricing (per 1 000 tokens, USD) — adjust to your model ───────────────────
PRICE_INPUT_PER_1K  = 0.0003   # e.g. gemma-3-27b via OpenRouter
PRICE_OUTPUT_PER_1K = 0.0006

# ── Sidebar – toggle buttons ──────────────────────────────────────────────────
with st.sidebar:
    api_key = st.text_input("OpenRouter API Key", type="password")
    st.markdown("---")
    st.markdown("### 🗓 Semester *(valikuline)*")

    cols = st.columns(2)
    for i, sem in enumerate(["kevad", "sügis"]):
        with cols[i]:
            label = f"{'✅ ' if sem in st.session_state.sel_semesters else ''}{sem}"
            if st.button(label, key=f"sem_{sem}"):
                if sem in st.session_state.sel_semesters:
                    st.session_state.sel_semesters.remove(sem)
                else:
                    st.session_state.sel_semesters.append(sem)
                st.rerun()

    st.markdown("### 📊 Hindamisviis *(valikuline)*")
    for hv in ["Eristav", "Eristamata"]:
        label = f"{'✅ ' if hv in st.session_state.sel_hindamisviis else ''}{hv}"
        if st.button(label, key=f"hv_{hv}"):
            if hv in st.session_state.sel_hindamisviis:
                st.session_state.sel_hindamisviis.remove(hv)
            else:
                st.session_state.sel_hindamisviis.append(hv)
            st.rerun()

    st.markdown("### 🎯 EAP maht *(valikuline)*")
    eap_options = ["1–3", "4–6", "7–10", "11–20"]
    eap_ranges  = [(1,3), (4,6), (7,10), (11,20)]
    cols2 = st.columns(2)
    for i, (label, rng) in enumerate(zip(eap_options, eap_ranges)):
        with cols2[i % 2]:
            disp = f"{'✅ ' if label in st.session_state.sel_eap else ''}{label} EAP"
            if st.button(disp, key=f"eap_{label}"):
                if label in st.session_state.sel_eap:
                    st.session_state.sel_eap.remove(label)
                else:
                    st.session_state.sel_eap.append(label)
                st.rerun()

    st.markdown("---")
    if st.button("🔄 Tühjenda filtrid"):
        st.session_state.sel_semesters = []
        st.session_state.sel_hindamisviis = []
        st.session_state.sel_eap = []
        st.rerun()

    # Show active filters summary
    active = []
    if st.session_state.sel_semesters:  active.append(f"Semester: {', '.join(st.session_state.sel_semesters)}")
    if st.session_state.sel_hindamisviis: active.append(f"Hindamisviis: {', '.join(st.session_state.sel_hindamisviis)}")
    if st.session_state.sel_eap:        active.append(f"EAP: {', '.join(st.session_state.sel_eap)}")
    if active:
        st.info("**Aktiivsed filtrid:**\n" + "\n".join(f"• {a}" for a in active))
    else:
        st.caption("ℹ️ Filtrid pole valitud — otsitakse kõigist kursustest.")

# ── Load models & data ────────────────────────────────────────────────────────
@st.cache_resource
def get_models():
    embedder = SentenceTransformer("BAAI/bge-m3")
    df = pd.read_csv("puhtad_andmed.csv")
    embeddings_df = pd.read_pickle("puhtad_andmed_embeddings.pkl")
    return embedder, df, embeddings_df

embedder, df, embeddings_df = get_models()

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
            st.error("Palun sisesta API võti!")
        else:
            with st.spinner("Otsin sobivaid kursusi..."):

                merged_df = pd.merge(df, embeddings_df, on='unique_ID')
                mask = pd.Series([True] * len(merged_df), index=merged_df.index)

                # Apply semester filter only if selections exist
                if st.session_state.sel_semesters:
                    mask &= merged_df['semester'].isin(st.session_state.sel_semesters)

                # Apply hindamisviis filter only if selections exist
                if st.session_state.sel_hindamisviis:
                    mask &= merged_df['hindamisviis'].isin(st.session_state.sel_hindamisviis)

                # Apply EAP filter only if selections exist
                if st.session_state.sel_eap:
                    eap_map = {"1–3":(1,3),"4–6":(4,6),"7–10":(7,10),"11–20":(11,20)}
                    eap_mask = pd.Series([False] * len(merged_df), index=merged_df.index)
                    for label in st.session_state.sel_eap:
                        lo, hi = eap_map[label]
                        eap_mask |= merged_df['eap'].between(lo, hi)
                    mask &= eap_mask

                filtered_df = merged_df[mask].copy()

                if filtered_df.empty:
                    st.warning("Ühtegi kursust ei vastanud filtritele.")
                    context_text = "Sobivaid kursusi ei leitud."
                else:
                    query_vec = embedder.encode([prompt])[0]
                    filtered_df['score'] = cosine_similarity(
                        [query_vec], np.stack(filtered_df['embedding'])
                    )[0]
                    results_df = filtered_df.sort_values('score', ascending=False).head(5)
                    results_df = results_df.drop(['score', 'embedding'], axis=1)
                    context_text = results_df.to_string()

            # ── LLM call ──────────────────────────────────────────────────
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

            system_prompt = {
                "role": "system",
                "content": (
                    f"Sa oled ülikooli kursuste nõustaja. Kasuta AINULT järgmisi kursusi:\n\n"
                    f"{context_text}\n\n"
                    "Reeglid:\n"
                    "• Näita top 5 sobivaimat kursust.\n"
                    "• Iga kursuse kohta kirjuta LÜHIKE kirjeldus (1–2 lauset).\n"
                    "• Kogu vastus peab olema MAKSIMAALSELT 200 sõna.\n"
                    "• Kasuta selget loendit (nt 1. Kursuse nimi – kirjeldus)."
                )
            }

            messages_to_send = [system_prompt] + st.session_state.messages

            try:
                # Use non-streaming so we get usage stats
                completion = client.chat.completions.create(
                    model="google/gemma-3-27b-it",
                    messages=messages_to_send,
                    max_tokens=350,   # safety cap ≈ 200 words
                    stream=False
                )
                response_text = completion.choices[0].message.content
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

                # ── Token accounting ───────────────────────────────────────
                if completion.usage:
                    st.session_state.total_input_tokens  += completion.usage.prompt_tokens
                    st.session_state.total_output_tokens += completion.usage.completion_tokens

            except Exception as e:
                st.error(f"Viga: {e}")

# ── Token / cost bar (fixed footer) ──────────────────────────────────────────
total_cost = (
    st.session_state.total_input_tokens  / 1000 * PRICE_INPUT_PER_1K +
    st.session_state.total_output_tokens / 1000 * PRICE_OUTPUT_PER_1K
)

st.markdown(
    f"""
    <div class="token-bar">
        📥 Sisend-tokenid: <span>{st.session_state.total_input_tokens:,}</span>
        &nbsp;|&nbsp;
        📤 Väljund-tokenid: <span>{st.session_state.total_output_tokens:,}</span>
        &nbsp;|&nbsp;
        💰 Hinnanguline kulu: <span>${total_cost:.5f}</span>
        &nbsp;&nbsp;
        <small style="color:#999">(seansi kogumaht)</small>
    </div>
    """,
    unsafe_allow_html=True
)