import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.title("Saydullayev Asadbek")

# title
st.title("Deep Learning yordamida Pnevmaniya tashxisini qo'yish")

# rasm yuklash
file = st.file_uploader("Rasm yuklash", type=["png", "jpg", "gif",'svg'])
if file:
    img = PILImage.create(file)
    st.image(file)

    # modelni yuklash
    model = load_learner("pnevmaniya_model.pkl")

    # bashorat qilish
    pred, pred_id, probs = model.predict(img)

    # natijani ekranga chiqarish
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id] * 100:.1f}%")
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)

