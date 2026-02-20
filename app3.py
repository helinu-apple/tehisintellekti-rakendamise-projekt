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

#todo


# JUBA OLEMAS
# 1. Algatame vestluse ajaloo, kui seda veel pole
if "messages" not in st.session_state:
    st.session_state.messages = []

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
            
            # UUS Muudame loetud andmed tekstiks, mida AI-le saata.
            # Kasutame ainult 10 esimest rida.
            # lisame selle system promptile või "messages_to_send" muutujale (mis formaadis see on?)
            system_prompt = None
            messages_to_send = None #todo
            
            try:
                stream = client.chat.completions.create(
                    model="google/gemma-3-27b-it:free",
                    messages=messages_to_send,
                    stream=True
                )
                response = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Viga: {e}")

#TODO TESTi brauseris: tere anna mulle kõigi kursuste nimed, mida sa tead