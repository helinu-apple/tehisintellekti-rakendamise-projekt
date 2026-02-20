import streamlit as st
from openai import OpenAI

# Iluasjad: pealkiri, allkiri
st.title("🎓 AI Kursuse Nõustaja - Samm 2")
st.caption("Vestlus päris tehisintellektiga (Gemma 3).")

# UUS
# Külgriba API võtme jaoks (sidebar)
api_key = None
with st.sidebar:
    st.header("🔑 OpenRouter API võti")
    api_key = st.text_input("Sisesta API võti", type="password", placeholder="sk-or-...")
    st.caption("Võti jääb ainult selle sessiooni mällu.")

# 1. Algatame vestluse ajaloo, kui seda veel pole
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Tere! Kirjelda, mida soovid õppida, ja ma soovitan sobivaid kursusi."}
    ]

# 2. Kuvame vestluse senise ajaloo (History)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Korjame üles uue kasutaja sisendi
if prompt := st.chat_input("Kirjelda, mida soovid õppida..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Kuvame kohe kasutaja sõnumi
    with st.chat_message("user"):
        st.markdown(prompt)

    # defineerime süsteemiprompti,
    system_prompt = (
        "Sa oled kursusenõustaja. Küsi vajadusel 1 täpsustav küsimus. "
        "Soovita lühidalt ja konkreetselt, eesti keeles."
    )

    # Kuvame vastuse striimina, ilmub jooksvalt
    with st.chat_message("assistant"):
        if not api_key:
            err = "Palun lisa OpenRouter API võti külgribalt, et saaksin Gemma 3 mudelit kasutada."
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
        else:
            # defineeri OpenAI klient OpenRouteri base_url'iga (OpenAI-ühilduv)
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )

            # anna süsteemiprompt ja vestluse ajalugu
            messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages

            try:
                # Striimimine (Chat Completions)
                stream = client.chat.completions.create(
                    model="google/gemma-3-27b-it",
                    messages=messages,
                    temperature=0.2,
                    stream=True,
                    extra_headers={
                        # Soovi korral (OpenRouter soovitab – rank/analytics)
                        "HTTP-Referer": "http://localhost:8501",
                        "X-Title": "AI Kursuse Noustaja",
                    },
                )

                def stream_text():
                    for chunk in stream:
                        delta = chunk.choices[0].delta
                        if delta and delta.content:
                            yield delta.content

                response = st.write_stream(stream_text())
                st.session_state.messages.append({"role": "assistant", "content": response})

            except Exception as e:
                st.error(f"Viga: {e}")
