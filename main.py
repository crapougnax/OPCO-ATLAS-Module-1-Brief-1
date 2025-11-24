import streamlit as st
import requests
from loguru import logger

st.title("Simuler votre montant de prêt")

st.slider("Votre âge", min_value=18, max_value=80, key="age")
st.number_input("Votre taille", key="taille", value=175)
st.number_input("Votre poids", key="poids", value=80)
st.radio("Votre sexe", ["H","F"], captions=["Homme", "Femme"], key="sexe")
st.radio("Licence sportive", ["oui","non"], captions=["Oui", "Non"], key="sport_licence")
st.selectbox("Niveau d'études", ["aucun", "bac", "bac+2","master", "doctorat"], key="niveau_etude")
st.selectbox("Région", ["Provence-Alpes-Côte d’Azur", "Île de France"], key="region")
st.radio("Fumeur", ["oui","non"], captions=["Oui", "Non"], key="smoker")
st.radio("Nationalité Française", ["oui","non"], captions=["Oui", "Non"], key="nationalité_francaise")
st.number_input("Revenu mensuel", key="revenu_estime_mois")

if st.button("Simuler"):

    if st.session_state:
        logger.info(f"Données à analyser: {st.session_state}")
        try:
            print(st.session_state)
            response = requests.post(
                "http://orignax.lp:9000/predict/", json={
                    "age": st.session_state.age,
                    "taille": st.session_state.taille,
                    "poids": st.session_state.poids,
                    "sexe": st.session_state.sexe,
                    "sport_licence": st.session_state.sport_licence
                }
            )
            # Lève une exception pour les codes d'erreur HTTP (4xx ou 5xx)
            response.raise_for_status()
            payload = response.json()
            st.write("Résultats de l'analyse :")
            print(payload)

        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de la requête : {e}")
