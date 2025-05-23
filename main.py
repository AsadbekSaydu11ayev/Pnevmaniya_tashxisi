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
st.text("Misol uchun")
col1, col2, col3 = st.columns(3)
with col1:
    st.image(PILImage.create("person1_bacteria_2.jpeg"), width=100)
with col2:
    st.image(PILImage.create("person2_bacteria_4.jpeg"), width=95)
with col3:
    st.image(PILImage.create("person3_bacteria_12.jpeg"), width=100)
    
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
        st.text("Siz rasm yuklamadingiz. Iltimos Rasm yuklang:)")
    st.text("Ishlab chiqaruvchi: Saydullayev Asadbek")


# ==========================================================================================

# st.title("Deep learning yordamida Pnevmaniya tashxisini va buyrak toshisini qo'yuvchi dastur")
# st.text("Ogohlantirish: Rasmlar faqat Pnevmaniya va Buyraka oid bo'lishi kerak aks holda dastur noto'g'ri ishlashi mumkin.")
# st.text("Misol uchun")
# col1, col2, col3, col4 = st.columns(4)
# with col1:
#     st.image(PILImage.create("person1_bacteria_2.jpeg"), width=100)
# with col2:
#     st.image(PILImage.create("person2_bacteria_4.jpeg"), width=95)
# with col3:
#     st.image(PILImage.create("965.jpg"), width=120)
# with col4:
#     st.image(PILImage.create("981.jpg"), width=150)
    
# file = st.file_uploader(label="Pneumonia rasmini yuklash", type=["Jpg", "Png", "Svg","Gif"], key='file1')


# if file:
#     try:
#         img = PILImage.create(file)

#         model = load_learner("pnevmaniya_model.pkl")
#         pred, pred_id, probs = model.predict(img)

#         st.success(f"Bashorat: {pred}")
#         st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")    
#         st.image(img, width=400)


#         fig = plt.figure(figsize=(10,4))
#         sns.barplot(x=probs*100, y=model.dls.vocab)
#         plt.xlabel("Ehtimollik")
#         st.pyplot(fig)

#     except:
#         st.text("Siz rasm yuklamadingiz. Iltimos Rasm yuklang")


# # Kidney stone
# file2 = st.file_uploader(label="Qorin bo‘shlig‘ining aksial kompyuter tomografiyasi tasviri yuklash", type=["Jpg", "Png", "Svg","Gif"], key='file2')

# if file2:
#     try:
#         img2 = PILImage.create(file2)

#         model2 = load_learner("kidney_stone.pkl")
#         pred2, pred_id2, probs2 = model2.predict(img2)

#         st.success(f"Bashorat: {pred2}")
#         st.info(f"Ehtimollik: {probs2[pred_id2]*100:.1f}%")    
#         st.image(img2, width=400)


#         fig = plt.figure(figsize=(10,4))
#         sns.barplot(x=probs2*100, y=model2.dls.vocab)
#         plt.xlabel("Ehtimollik")
#         st.pyplot(fig)

#     except:
#         st.text("Siz rasm yuklamadingiz. Iltimos Rasm yuklang!")
#     st.text("Ishlab chiqaruvchi: Legion jamoasi")
