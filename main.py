import streamlit as st
from fastai.vision.all import *
import pathlib, platform
import plotly.express as px
plt = platform.system()
if plt == "Linux": pathlib.WindowsPath = pathlib.PosixPath
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Rasm asosida Pnevmaniya tashxisini qo'yuvchi dastur")
st.text("Ogohlantirish: Rasmlar faqat Pnevmaniyaga oid bo'lishi kerak aks holda dastur noto'g'ri ishlashi mumkin.")
col1, col2, col3 = st.columns(3)
st.text("Misol uchun")
with col1
    st.img("person1_bacteria_2.jpeg", width=100)
with col2
    st.img("person1_bacteria_4.jpeg", width=100)
with col2
    st.img("person1_bacteria_12.jpeg", width=100)
file = st.file_uploader(label="Rasm yuklash", type=["Jpg", "Png", "Svg","Gif"])


if file:
    try:
        img = PILImage.create(file)

        model = load_learner("pnevmaniya_model.pkl")
        pred, pred_id, probs = model.predict(img)

        st.success(f"Bashorat: {pred}")
        st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")    
        st.image(img, width=400)


        fig = plt.figure(figsize=(10,4))
        sns.barplot(x=probs*100, y=model.dls.vocab)
        plt.xlabel("Ehtimollik")
        st.pyplot(fig)

    except:
        st.text("Siz rasm yuklamadingiz. Iltimos Rasm yuklang")
    st.text("Ishlab chiqaruvchi: Saydullayev Asadbek")
