import streamlit as st
import requests
from loguru import logger

st.title("Translator")

st.text_input("Votre message", key="texte")

if st.button("Analyser"):
    age = st.session_state.number
    # taille,poids,sexe,sport_licence,niveau_etude,region,smoker,nationalité_francaise,revenu_estime_mois,
    texte = st.session_state.texte
    # Si il y a du text alors
    if texte:
        logger.info(f"Texte à analyser: {texte}")
        try:
            response = requests.post(
                "http://127.0.0.1:9000/chat/", json={"texte": texte}
            )
            # Lève une exception pour les codes d'erreur HTTP (4xx ou 5xx)
            response.raise_for_status()
            payload = response.json()
            st.write("Résultats de l'analyse :")
            print(payload)
            st.write(f"Message : {payload['message']}")
            st.write(f"Traduction : {payload['translation']}")
            st.write(f"Sentiment Score : {payload['sentiment'][0]['score']}")
            st.write(f"Réponse : {payload['reply']}")

        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de la requête : {e}")
