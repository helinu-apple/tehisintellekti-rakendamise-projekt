import streamlit as st
import pandas as pd
from openai import OpenAI

st.title("🎓 AI Kursuse Nõustaja")
st.caption("AI kasutab kursuste andmeid (esimesed 10 rida).")

# Külgriba API võtme jaoks
with st.sidebar:
    api_key = st.text_input("OpenRouter API Key", type="password")

# UUS
# Laeme andmed (puhatad_andmed.csv peab olema õiges asukohas)
# oluline on kasutada st.cache_data, et me ei laeks andmeid failist uuesti igal värskendamise korral
@st.cache_data
def load_courses(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

DATA_PATH = "puhtad_andmed.csv"  # vajadusel muuda: nt "data/puhtad_andmed.csv"

try:
    df = load_courses(DATA_PATH)
    st.subheader("📄 Andmete eelvaade (10 esimest rida)")
    st.dataframe(df.head(10), use_container_width=True)
except FileNotFoundError:
    st.error(f"Ei leidnud faili: {DATA_PATH}. Pane CSV samasse kausta või muuda DATA_PATH.")
    st.stop()
except Exception as e:
    st.error(f"Andmete laadimine ebaõnnestus: {e}")
    st.stop()

# JUBA OLEMAS
# 1. Algatame vestluse ajaloo, kui seda veel pole
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Tere! Kirjelda, mida soovid õppida, ja otsin sobiva kursuse neist andmetest."}
    ]

# 2. Kuvame vestluse senise ajaloo (History)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Korjame üles uue kasutaja sisendi
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
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

            # UUS: Muudame loetud andmed tekstiks, mida AI-le saata (ainult 10 rida)
            preview_df = df.head(10).copy()

            # Kui andmestik on lai, piira veerge (valikuline)
            # preview_df = preview_df[["course_name", "description", "level"]]

            courses_text = preview_df.to_csv(index=False)

            system_prompt = (
                "Sa oled kursusenõustaja. Kasuta AINULT alltoodud kursuste andmeid.\n"
                "Kui kasutaja küsib midagi, mida nendest ridadest ei saa järeldada, ütle ausalt, et andmetest ei piisa.\n"
                "Vasta eesti keeles ja lühidalt.\n\n"
                "KURSUSED (CSV, 10 rida):\n"
                f"{courses_text}"
            )

            # messages peab olema list dict-e kujul: [{"role":"system","content":...}, ...]
            #messages_to_send = [{"role": "system", "content": system_prompt}] + st.session_state.messages
            
            preview_df = df.head(10)
            courses_text = preview_df.to_csv(index=False)

            system_prompt = (
                "Sa oled kursusenõustaja. Kasuta ainult neid kursusi.\n\n"
                f"{courses_text}"
            )

            # Always construct fresh messages
            messages_to_send = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

                # Streamlitile sobiv generator: yield tekstijupid
            stream = client.chat.completions.create(
            model="google/gemma-3-27b-it:free",
            messages=messages_to_send,
            stream=True,
        )

        def stream_text():
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        response = st.write_stream(stream_text())
# TODO TESTi brauseris: "tere anna mulle kõigi kursuste nimed, mida sa tead"