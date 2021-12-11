import streamlit as st
from PIL import Image

img = Image.open("images/crowd-gc46d97eb2_1920.jpg")

st.image(img)
st.title("Sooner or later?  Walkthrough to predict when an elite soccer player will get injured.")

st.write("Sravan Rajupalem") 
st.write("Renzo Maldonado")
st.write("Victor Ruiz")