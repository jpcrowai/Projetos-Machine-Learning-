import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# Título da Aplicação
st.title("Previsão de Dados")
# Upload do dataset
uploaded_file = st.file_uploader("Escolha o arquivo do seus dados", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dados carregados:")
    st.write(data.head())
    st.write("Processando os dados...")
    X = data.drop('Crop_Damage', axis=1) 
    y = data['Crop_Damage']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    st.write(f"Acurácia do Modelo: {accuracy * 100:.2f}%")
    # Previsões
    st.write("Faça uma previsão com novos dados:")
    resultado = str(' ')
    input_data = []
    for col in X.columns:
        input_val = st.number_input(f"Insira o valor para {col}", value=0)
        input_data.append(input_val)
    if st.button("Prever"):
        prediction = model.predict([input_data])
        if prediction[0] == 0:
            resultado = str('Saudável(viva)')
        elif prediction[0] == 1:
            resultado = str('Danificada por pesticidas')
        else:
            resultado = str('Danificada por outros')

        st.write(f"Previsão: {resultado}")
