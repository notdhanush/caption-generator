import streamlit as st
import whisper
import tempfile
import os
import srt
import datetime

# Title
st.title("🎙️ Tamil & Thanglish Auto-Caption Generator")

# Upload
uploaded_file = st.file_uploader("Upload video/audio file", type=["mp4", "mp3", "wav", "m4a"])
lang_choice = st.radio("Choose Caption Language:", ["Tamil", "Thanglish"])

# When file is uploaded
if uploaded_file is not None:
    with st.spinner("Transcribing... Please wait ⏳"):
        # Save file temporarily
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        temp_file.close()

        # Load Whisper
        model = whisper.load_model("tiny")
        result = model.transcribe(temp_file.name, language="ta")

        # Display Tamil transcript
        tamil_text = result["text"]
        st.subheader("📝 Tamil Transcript:")
        st.write(tamil_text)

        # If Thanglish needed, rewrite using OpenAI
        if lang_choice == "Thanglish":
            import openai
            openai.api_key = st.text_input("Enter your OpenAI API key:", type="password")

            if openai.api_key:
                with st.spinner("Converting to Thanglish..."):
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "user", "content": f"Convert this Tamil text to Thanglish:\n{tamil_text}"}
                        ]
                    )
                    thanglish_text = response["choices"][0]["message"]["content"]
                    st.subheader("📝 Thanglish Transcript:")
                    st.write(thanglish_text)
            else:
                st.warning("Please enter your OpenAI API key to get Thanglish output.")

        # Generate SRT
        st.subheader("📄 Download Captions")
        captions = []
        for i, segment in enumerate(result['segments']):
            start = datetime.timedelta(seconds=segment['start'])
            end = datetime.timedelta(seconds=segment['end'])
            content = segment['text']
            captions.append(srt.Subtitle(index=i+1, start=start, end=end, content=content))

        srt_data = srt.compose(captions)

        with open("captions.srt", "w", encoding='utf-8') as f:
            f.write(srt_data)

        st.download_button("Download SRT File", srt_data, file_name="captions.srt", mime="text/plain")

        # Cleanup
        os.remove(temp_file.name)
