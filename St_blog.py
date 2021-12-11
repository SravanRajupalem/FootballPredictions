import streamlit as st
from PIL import Image

img = Image.open("images/crowd-gc46d97eb2_1920.jpg")

st.image(img)
st.title("Sooner or later?  Walkthrough to predict when an elite soccer player will get injured.")

st.write("Sravan Rajupalem") 
st.write("Renzo Maldonado")
st.write("Victor Ruiz is in Orlando")

section = st.sidebar.("Sections", ("Scraping the Web for Data", "Data Manipulation", "Feature Engineering", 
    "Visual Exploration of Data", "Model Building"))

st.write("""For quite a while, 'Sports Analytics' has been the buzz-word in the world of Data Science. Magically using complex 
    algorithms, machine learning models and neural networks to predict sports results and players' performance attract the interest 
    of people for different reasons. Soccer is probably one of the most unpredictable sports out there. In the hope of aiding soccer 
    managers' decisions, we decided to apply Data Science tools to predict how likely a player was to have an injury within a 
    certain time frame.""")