import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import json
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from datetime import datetime
import math
from io import BytesIO

st.set_page_config(
    page_title="YB Tracking Analyzer",
    layout="wide",
    page_icon="⛵",
)

# ------------------------------------------------------------------
# Téléchargement (amélioré) des données YB
# ------------------------------------------------------------------

ENDPOINTS = {
    "json":    "https://cf.yb.tl/JSON/{race_id}/{file}",
    "binary":  "https://cf.yb.tl/BIN/{race_id}/{file}",
}

def safe_get_json(url: str, max_retries: int = 3, backoff: int = 2):
    """Effectue un GET JSON en gérant les erreurs 500 et la reconnexion exponentielle."""
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "YB-Analyzer/1.0"})
            # Cas fréquent : LatestPositions 500 lorsque pas encore généré
            if r.status_code == 500:
                return None
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException:
            if attempt == max_retries:
                raise
            time.sleep(backoff ** attempt)

def build_latest_from_all_positions(all_positions: dict) -> dict:
    """Construit un pseudo‑LatestPositions à partir du dernier point de chaque bateau."""
    latest = {"positions": []}
    for boat in all_positions.get("boats", []):
        if boat.get("positions"):
            latest["positions"].append({"boatID": boat["id"], **boat["positions"][-1]})
    return latest

@st.cache_data(ttl=300, show_spinner=False)
def fetch_yb_data(race_id: str) -> dict:
    """Télécharge et renvoie toutes les données nécessaires pour *race_id*."""
    data = {}

    # Fichiers JSON
    json_files = ["RaceSetup", "leaderboard", "LatestPositions"]
    for fname in json_files:
        url = ENDPOINTS["json"].format(race_id=race_id, file=fname)
        with st.spinner(f"Téléchargement de {fname}…"):
            try:
                payload = safe_get_json(url)
                if payload is None:
                    st.info(f"{fname} indisponible – HTTP 500 renvoyé par YB.")
                else:
                    st.success(f"{fname} récupéré.")
                data[fname] = payload
            except Exception as e:
                st.error(f"{fname} indisponible : {e}")
                data[fname] = None

    # Fichier binaire AllPositions3
    bin_url = ENDPOINTS["binary"].format(race_id=race_id, file="AllPositions3")
    with st.spinner("Téléchargement de l’historique des positions…"):
        try:
            r = requests.get(bin_url, timeout=120)
            r.raise_for_status()
            data["AllPositionsBin"] = r.content
            st.success("AllPositions3 téléchargé.")
        except Exception as e:
            st.error(f"AllPositions3 indisponible : {e}")
            data["AllPositionsBin"] = b""  # vide

    # Décodage (simulé) du binaire
    data["AllPositions"] = decode_all_positions(data.get("AllPositionsBin", b""))

    # Reconstruction LatestPositions si besoin
    if data.get("LatestPositions") is None and data.get("AllPositions"):
        data["LatestPositions"] = build_latest_from_all_positions(data["AllPositions"])
        st.info("LatestPositions reconstruit à partir du binaire.")

    return data

# ------------------------------------------------------------------
# Fonctions utilitaires (géodésie, décodage simplifié, etc.)
# ------------------------------------------------------------------

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Rayon de la Terre en km
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def initial_bearing(lat1, lon1, lat2, lon2):
    y = math.sin(math.radians(lon2 - lon1)) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(
        math.radians(lat1)
    ) * math.cos(math.radians(lat2)) * math.cos(math.radians(lon2 - lon1))
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360) % 360

def decode_all_positions(binary_data):
    """Décodage simplifié (stub) du binaire AllPositions3 – à remplacer par decyb."""
    if not binary_data:
        return {"boats": []}

    # Cette version de démonstration génère des données factices basées
    # sur un format minimal {boats: [{id, name, positions: [ {lat, lon, time, sog} ]}]}
    st.warning(
        "Décodage simplifié utilisé. Pour une production, utilisez une vraie "
        "implémentation de decyb ou un portage Python."
    )

    # Exemple de données simulées
    return {
        "boats": [
            {
                "id": 1,
                "name": "Demo Boat",
                "positions": [
                    {
                        "lat": 48.0,
                        "lon": -4.0,
                        "time": int(time.time()) - 3600,
                        "sog": 6.2,
                    },
                    {
                        "lat": 48.1,
                        "lon": -3.9,
                        "time": int(time.time()),
                        "sog": 6.5,
                    },
                ],
            }
        ]
    }

# ------------------------------------------------------------------
# Interface Streamlit
# ------------------------------------------------------------------

st.title("YB Tracking Analyzer")
st.markdown("Analysez les données de courses YB Tracking")

# Barre latérale : sélection de la course
st.sidebar.header("Téléchargez les données")
race_id = st.sidebar.text_input("Identifiant de la course", "dgb2025")

action = st.sidebar.button("Télécharger les données")

if action:
    data = fetch_yb_data(race_id)
    st.session_state["data"] = data

# Affichage d'un résumé lorsque des données sont chargées
data = st.session_state.get("data")
if data:
    st.success("Données chargées – choisissez une vue dans la barre latérale.")

    # Exemple : Afficher une carte Folium des positions actuelles
    latest = data.get("LatestPositions", {}).get("positions", [])
    if latest:
        # Centre de la carte : moyenne des positions
        lat0 = np.mean([p["lat"] for p in latest])
        lon0 = np.mean([p["lon"] for p in latest])

        m = folium.Map(location=[lat0, lon0], zoom_start=6, tiles="OpenStreetMap")
        cluster = MarkerCluster().add_to(m)
        for p in latest:
            folium.Marker(
                location=[p["lat"], p["lon"]],
                popup=f"Boat {p['boatID']}",
                icon=folium.Icon(color="blue", icon="ship", prefix="fa"),
            ).add_to(cluster)
        folium_static(m, width=1000, height=600)
    else:
        st.info("Aucune position disponible pour afficher la carte.")

# Footer
st.markdown(
    "<sub>Cette application est un prototype open‑source et n'est pas affiliée à YB Tracking.</sub>",
    unsafe_allow_html=True,
)
