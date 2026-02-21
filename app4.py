import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Pealkirjad
st.title("🎓 AI Kursuse Nõustaja - RAGiga")
st.caption("Täisväärtuslik RAG süsteem semantilise otsinguga.")

# Külgriba
with st.sidebar:
    api_key = st.text_input("OpenRouter API Key", type="password")

# UUS
# Mudeli, andmetabeli ja vektoriseeritud andmete laadimine
# OLULINE: andmed on juba vektoriteks tehtud, loe need .pkl failist
# Eeldame, et puhtad_andmed_embeddings.pkl on pd.DataFrame: veergudega (unique_ID, embedding)
# tuleb kasutada streamliti cache_resource, et me mudelit ja andmeid pidevalt uuesti ei laeks
@st.cache_resource
def get_models():
    # 1) Embedder (sama mudel, millega tegid embeddings)
    embedder = SentenceTransformer("BAAI/bge-m3")

    # 2) Kursuste tabel (CSV)
    # Muuda teekondi vastavalt oma projektile
    df = pd.read_csv("puhtad_andmed.csv")

    # 3) Embeddingud (PKL)
    emb_df = pd.read_pickle("puhtad_andmed_embeddings.pkl")

    # Normalizeeri pkl kuju: dict {unique_ID: np.array}
    # (Kui sul on juba dict, siis jäta see osa alles, töötab ka.)
    if isinstance(emb_df, pd.DataFrame):
        embeddings_dict = dict(
            zip(
                emb_df["unique_ID"].astype(str).tolist(),
                emb_df["embedding"].apply(lambda x: np.array(x, dtype=np.float32)).tolist(),
            )
        )
    elif isinstance(emb_df, dict):
        embeddings_dict = {str(k): np.array(v, dtype=np.float32) for k, v in emb_df.items()}
    else:
        raise ValueError("puhtad_andmed_embeddings.pkl peab olema DataFrame või dict.")

    # veendu, et df-s on unique_ID olemas ja stringina
    if "unique_ID" not in df.columns:
        raise ValueError("CSV failis peab olema veerg 'unique_ID'.")

    df["unique_ID"] = df["unique_ID"].astype(str)

    return embedder, df, embeddings_dict


embedder, df, embeddings_dict = get_models()

# 1. Algatame vestluse ajaloo, kui seda veel pole
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Kirjelda, mida soovid õppida – otsin semantiliselt sobivaid kursusi."}
    ]

# 2. Kuvame vestluse senise ajaloo (History)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Korjame üles kasutaja sõnumi
if prompt := st.chat_input("Kirjelda, mida soovid õppida..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not api_key:
            error_msg = "Palun sisesta API võti!"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            # UUS Semantiline otsing (RAG)
            with st.spinner("Otsin sobivaid kursusi..."):
                # Teeme kasutaja küsimusest vektori (query)
                query_vec = embedder.encode(prompt, normalize_embeddings=True)
                query_vec = np.array(query_vec, dtype=np.float32).reshape(1, -1)

                # Ühendame .pkl embeddingud csv andmetabeliga
                # teeme df-le veeru "embedding" embeddings_dict põhjal
                df_work = df.copy()
                df_work["embedding"] = df_work["unique_ID"].map(embeddings_dict)

                # eemaldame read, millel embedding puudu
                df_work = df_work.dropna(subset=["embedding"]).reset_index(drop=True)

                # Arvutame koosinussarnasuse query ja embeddingute vahel
                emb_matrix = np.vstack(df_work["embedding"].values).astype(np.float32)
                # kui embeddingud pole normaliseeritud, cosine_similarity töötab ikka, aga parem on normaliseerida:
                # (kui sul embeddingud juba normitud, võib selle vahele jätta)
                norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
                emb_matrix = emb_matrix / np.clip(norms, 1e-12, None)

                scores = cosine_similarity(query_vec, emb_matrix)[0]
                df_work["score"] = scores

                # Sorteerime ja võtame top N=5
                results_df = df_work.sort_values("score", ascending=False).head(5).copy()

                # eemaldame vestluse jaoks ebavajalikud veerud
                drop_cols = [c for c in ["score", "embedding", "unique_ID"] if c in results_df.columns]
                results_df = results_df.drop(columns=drop_cols)

                context_text = results_df.to_string(index=False)

            # LLM vastus koos kontekstiga
            instruction_text = (
                "Sa oled kursusenõustaja.\n"
                "Kasuta ainult järgnevaid RAGi leitud kursusi vastamiseks.\n"
                "Kui info ei piisa, ütle seda.\n\n"
                f"KONTEKST:\n{context_text}\n\n"
                f"KASUTAJA KÜSIMUS:\n{prompt}"
            )

            messages_to_send = [
                {"role": "user", "content": instruction_text}
            ]
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

            
            # Soovitus: saada ajalugu + uus prompt (siin on ajalugu juba session_state.messages sees)
            #messages_to_send = [instruction_text] + st.session_state.messages

            try:
                stream = client.chat.completions.create(
                    model="google/gemma-3-27b-it:free",
                    messages=messages_to_send,
                    temperature=0.2,
                    stream=True,
                )

                # Streamlit vajab generatorit, mis yieldib teksti
                def stream_text():
                    for chunk in stream:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            yield delta.content

                response = st.write_stream(stream_text())
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Viga: {e}")