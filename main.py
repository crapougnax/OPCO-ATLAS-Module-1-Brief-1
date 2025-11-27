import streamlit as st
import requests
from loguru import logger
import json

def encoder(obj):
    if isinstance(obj, set):
        return list(obj)
    return obj

st.title("Simuler votre montant de prêt")

st.slider("Votre âge", min_value=18, max_value=80, key="age")
st.number_input("Votre taille", key="taille", value=175)
st.number_input("Votre poids", key="poids", value=80)
st.radio("Votre sexe", ["H","F"], captions=["Homme", "Femme"], key="sexe")
st.radio("Licence sportive", ["oui","non"], captions=["Oui", "Non"], key="sport_licence")
st.selectbox("Niveau d'études", ["aucun", "bac", "bac+2","master", "doctorat"], key="niveau_etude")
st.selectbox("Région", ["Provence-Alpes-Côte d’Azur", "Île de France"], key="region")
st.radio("Fumeur", ["oui","non"], captions=["Oui", "Non"], key="smoker")
st.radio("Nationalité Française", ["oui","non"], captions=["Oui", "Non"], key="nationalite")
st.number_input("Revenu mensuel", key="revenu_estime_mois")

if st.button("Simuler"):

    if st.session_state:
        logger.info(f"Données à analyser: {st.session_state}")
        
        gfg = [('age', st.session_state.age), ('taille', st.session_state.taille), ('poids', st.session_state.poids)]
        
        json_data = dict(gfg)

        json_string = json.dumps(json_data, default=encoder)

        data={
            "age": st.session_state.age,
            "taille": st.session_state.taille,
            "poids": st.session_state.poids,
            "sexe": st.session_state.sexe,
            "sport_licence": st.session_state.sport_licence,
            "niveau_etude": st.session_state.niveau_etude,
            "region": st.session_state.region,
            "smoker": st.session_state.smoker,
            "nationalite": st.session_state.nationalite,
            "revenu_estime_mois": st.session_state.revenu_estime_mois
        }
        
        print(json_string)
        
        try:
            response = requests.post(
                "http://orignax.lp:9000/predict/", json=gfg
            )
            # Lève une exception pour les codes d'erreur HTTP (4xx ou 5xx)
            response.raise_for_status()
            payload = response.json()
            st.write("Résultats de l'analyse :")
            print(payload)

        except requests.exceptions.RequestException as e:
            print(e)
            st.error(f"Erreur lors de la requête : {e}")
            
