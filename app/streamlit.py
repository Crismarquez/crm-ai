import time
import streamlit as st

from config.config import logger
from vision_analytic.main import createregister

st.title("Register in crm-vision")
st.subheader("Enter details below")

with st.form("form1", clear_on_submit=True):
    name = st.text_input("Enter full name")
    phone = st.text_input("Por favor ingrese su numero de teléfono")
    id_user = st.text_input("Por favor ingrese su numero de identificación")
    age = st.slider("Por favor ingrese su edad", min_value=18, max_value=100)
    st.write(age)
    accept = st.text_input("Acepta terminos y condiciones (y/n): ")

    submit = st.form_submit_button("Submit this form")
    if submit:
        time_register = time.strftime("%d-%m-%Y-%H-%M-%S", time.localtime())

        user_info = {
            "name": [name],
            "age": [age],
            "phone": [phone],
            "id_user": [id_user],
            "accept": [accept],
        }
        logger.info(user_info)
        createregister(user_info)
