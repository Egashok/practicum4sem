import streamlit as st
import pandas as pd
import pickle
import numpy as np



st.title("Получить предсказания пожара.")

st.header("height_cm")
height_cm = st.number_input("Число:",key="1", value=170)

st.header("weight_kg:")
weight_kg = st.number_input("Число:",key="2", value=72)

st.header("potential")
potential = st.number_input("Число:",key="3", value=93)

st.header("attacking_crossing:")
attacking_crossing = st.number_input("Число:",key="4", value=85)

st.header("attacking_finishing")
attacking_finishing = st.number_input("Число:",key="5", value=95)

st.header("attacking_heading_accuracy")
attacking_heading_accuracy = st.number_input("Число:",key="6", value=70)

st.header("attacking_short_passing:")
attacking_short_passing = st.number_input("Число:",key="7", value=91)

st.header("attacking_volleys")
attacking_volleys = st.number_input("Число:",key="8", value=88)

st.header("skill_dribbling:")
skill_dribbling = st.number_input("Число:",key="9", value=96)

st.header("skill_curve")
skill_curve = st.number_input("Число:", key="10", value=93)

st.header("skill_fk_accuracy:")
skill_fk_accuracy = st.number_input("Число:",key="11", value=94)

st.header("skill_long_passing:")
skill_long_passing = st.number_input("Число:", key="12", value=91)

st.header("skill_ball_control")
skill_ball_control = st.number_input("Число:", key="13", value=96)

st.header("movement_acceleration:")
movement_acceleration = st.number_input("Число:", key="14", value=91)

st.header("movement_sprint_speed:")
movement_sprint_speed = st.number_input("Число:",key="15", value=80)

st.header("movement_agility:")
movement_agility = st.number_input("Число:", key="16", value=91)

st.header("movement_reactions:")
movement_reactions = st.number_input("Число:", key="17", value=94)

st.header("movement_balance")
movement_balance = st.number_input("Число:", key="18", value=95)

st.header("defending_standing_tackle:")
defending_standing_tackle = st.number_input("Число:",key="19", value=35)

st.header("defending_sliding_tackle:")
defending_sliding_tackle = st.number_input("Число:",key="20", value=24)

st.header("goalkeeping_diving:")
goalkeeping_diving = st.number_input("Число:",key="21", value=6)

st.header("goalkeeping_handling:")
goalkeeping_handling = st.number_input("Число:",key="22", value=11)

st.header("goalkeeping_kicking")
goalkeeping_kicking = st.number_input("Число:",key="23", value=15)

st.header("goalkeeping_positioning:")
goalkeeping_positioning = st.number_input("Число:",key="24", value=14)

st.header("goalkeeping_reflexes:")
goalkeeping_reflexes = st.number_input("Число:",key="25", value=8)


data = pd.DataFrame({'height_cm': [height_cm],
                    'weight_kg': [weight_kg],
                    'potential': [potential],
                    'attacking_crossing': [attacking_crossing],
                    'attacking_finishing': [attacking_finishing],
                    'attacking_heading_accuracy': [attacking_heading_accuracy],
                    'attacking_short_passing': [attacking_short_passing],
                    'attacking_volleys': [attacking_volleys],
                    'skill_dribbling': [skill_dribbling],
                    'skill_curve': [skill_curve],
                    'skill_fk_accuracy': [skill_fk_accuracy],
                    'skill_long_passing': [skill_long_passing],
                    'skill_ball_control': [skill_ball_control],
                    'movement_acceleration': [movement_acceleration],          
                    'movement_sprint_speed': [movement_sprint_speed],   
                    'movement_agility': [movement_agility],   
                    'movement_reactions': [movement_reactions],   
                    'movement_balance': [movement_balance],   
                    'defending_standing_tackle': [defending_standing_tackle],   
                    'defending_sliding_tackle': [defending_sliding_tackle],   
                    'goalkeeping_diving': [goalkeeping_diving],   
                    'goalkeeping_handling': [goalkeeping_handling],   
                    'goalkeeping_kicking': [goalkeeping_kicking],   
                    'goalkeeping_positioning': [goalkeeping_positioning],   
                    'goalkeeping_reflexes': [goalkeeping_reflexes],   
                    })


button_clicked = st.button("Предсказать")

if button_clicked:
    print('xz')
    with open('models/multilasso.pkl', 'rb') as file:
        lr = pickle.load(file)


    st.header("multilasso:")
    pred =[]
    lasso_pred = lr.predict(data)[0]
    pred.append(lasso_pred)
    st.write(f"{lasso_pred}")
