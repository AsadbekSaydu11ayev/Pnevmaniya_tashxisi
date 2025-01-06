import streamlit as st
from fastai.vision.all import *
import pathlib as pth
pl = platform.system()
if pl == "Linux": pth.WindowsPath = pth.PosixPath

st.title("Mening dasturimga xush kelibsizlar")
st.text("Made this codes by Asadbek Saydullayev")
file = st.file_uploader(label="file yuklang")
model = load_learner("pnevmaniya_model.pkl")
if file:
    try:
        pred, pred_id, probs = model.predict(PILImage.create(file))
        st.text(f"It is {pred}, accuracy: {probs[pred_id] * 100:.1f}%")
        st.image(PILImage.create(file), width=400)
    except:
        st.text("Siz noto'g'ri fayl turini kiritdingiz. Iltimos rasm farmatdagi file kiriting.")


name = st.text_input(label="write name: ")
surname = st.text_input(label='write surname: ')

st.text(f"{name} {surname}")
