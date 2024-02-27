import streamlit as st
import pandas as pd
import pickle
import numpy as np



st.header("distance_from_home")
distance_from_home = st.number_input("Число:",key='1' ,value=57.877857)

st.header("distance_from_last_transaction:")
distance_from_last_transaction = st.number_input("Число:",key='2' , value=0.311140)

st.header("ratio_to_median_purchase_price")
ratio_to_median_purchase_price = st.number_input("Число:",key='3' , value=1.945940)

st.header("repeat_retailer:")
repeat_retailer = st.number_input("Число:",key='4' , value=1.0)

st.header("used_chip:")
used_chip = st.number_input("Число:",key='5' , value=1.0)

st.header("used_pin_number:")
used_pin_number = st.number_input("Число:",key='6' , value=0.0)

st.header("online_order:")
online_order = st.number_input("Число:",key='7' , value=0.0)


data = pd.DataFrame({'distance_from_home': [distance_from_home],
                    'distance_from_last_transaction': [distance_from_last_transaction],
                    'ratio_to_median_purchase_price': [ratio_to_median_purchase_price],
                    'repeat_retailer': [repeat_retailer],
                    'used_chip': [used_chip],
                    'used_pin_number': [used_pin_number],
                    'online_order': [online_order],
               
                    })


button_clicked = st.button("Предсказать")

if button_clicked:

    from tensorflow.keras.models import load_model
    nn_model = load_model('models/nn.h5')



    st.header("Perceptron:")
    nn_pred = round(nn_model.predict(data)[0][0])

    st.write(f"{nn_pred}")
