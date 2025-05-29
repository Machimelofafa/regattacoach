import streamlit as st
import requests
import time
import math
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static

st.set_page_config(
    page_title="YB Tracking Analyzer",
    page_icon="⛵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Utilitaires bas niveau
# -----------------------------------------------------------------------------

HEADERS = {"User-Agent": "YB-Analyzer/1.0"}

@st.cache_data(ttl=300, show_spinner=False)
def safe_get_json(url: str, max_retries: int = 3, backoff: int = 2):
    """Tente de récupérer un JSON. Si le serveur renvoie 500 → None."""
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, timeout=30, headers=HEADERS)
            if r.status_code == 500:
                return None  # Fichier absent côté YB → on s'en passe
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == max_retries:
                st.error(f"{url.split('/')[-1]} indisponible : {e}")
                return None
            time.sleep(backoff ** attempt)

@st.cache_data(ttl=300, show_spinner=False)
def safe_get_binary(url: str, max_retries: int = 3, backoff: int = 2):
    """Récupère un binaire (AllPositions3). Renvoie None si 500."""
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, timeout=60, stream=True, headers=HEADERS)
            if r.status_code == 500:
                return None
            r.raise_for_status()
            return r.content
        except Exception as e:
            if attempt == max_retries:
                st.error(f"{url.split('/')[-1]} indisponible : {e}")
                return None
            time.sleep(backoff ** attempt)

# -----------------------------------------------------------------------------
# Décodage – démo simplifiée
# -----------------------------------------------------------------------------

def decode_all_positions(binary_data: bytes):
    """Essaie de décompresser le binaire AllPositions3.
    1. Certains fichiers sont simplement gzippés (header GZIP « 1F 8B »)
    2. D'autres sont zlib+JSON – on tente les deux.
    Si tout échoue → liste vide, mais on trace l'erreur.
    """
    import gzip, zlib, json, io

    # Tentative GZIP ------------------------------------------------------
    try:
        if binary_data[:2] == b"":
            txt = gzip.decompress(binary_data).decode("utf-8", errors="ignore")
            return json.loads(txt)
    except Exception as e:
        st.info(f"GZIP failed ({e}) – trying zlib …")

    # Tentative zlib/raw DEFLATE -----------------------------------------
    try:
        txt = zlib.decompress(binary_data, wbits=16 + zlib.MAX_WBITS).decode("utf-8", errors="ignore")
        return json.loads(txt)
    except Exception as e:
        st.warning(f"Décodage du binaire impossible : {e}")
        return {"boats": []}

# -----------------------------------------------------------------------------
# Reconstruction de LatestPositions si besoin
# -----------------------------------------------------------------------------

def build_latest_from_all_positions(all_positions: dict):
    latest = {"positions": []}
    for boat in all_positions.get("boats", []):
        if boat.get("positions"):
            latest["positions"].append({
                "boatID": boat["id"],
                **boat["positions"][-1]
            })
    return latest if latest["positions"] else None

# -----------------------------------------------------------------------------
# Téléchargement principal
# -----------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def fetch_yb_data(race_id: str):
    data = {}
    urls = {
        "RaceSetup":   f"https://cf.yb.tl/JSON/{race_id}/RaceSetup",
        "Leaderboard": f"https://cf.yb.tl/JSON/{race_id}/leaderboard",
        "LatestPositions": f"https://cf.yb.tl/JSON/{race_id}/LatestPositions",
    }

    with st.spinner("Téléchargement des JSON …"):
        for name, url in urls.items():
            data[name] = safe_get_json(url)
            if data[name] is None:
                st.info(f"{name} indisponible – HTTP 500 renvoyé par YB.")

    # Binaire AllPositions3 ----------------------------------------------------
    all_pos_url = f"https://cf.yb.tl/BIN/{race_id}/AllPositions3"
    with st.spinner("Téléchargement de l'historique des positions …"):
        binary = safe_get_binary(all_pos_url)
        if binary:
            decoded = decode_all_positions(binary)
            data["AllPositions"] = decoded
        else:
            data["AllPositions"] = None
            st.info("AllPositions3 indisponible – course à venir ou ID erroné.")

    # Fallback LatestPositions -------------------------------------------------
    if data.get("LatestPositions") is None and data.get("AllPositions"):
        rebuilt = build_latest_from_all_positions(data["AllPositions"])
        if rebuilt:
            data["LatestPositions"] = rebuilt
            st.success("LatestPositions reconstruit à partir du binaire.")

    return data

# -----------------------------------------------------------------------------
# Fonctions de calcul (distance, cap, vitesse)
# -----------------------------------------------------------------------------

R_EARTH_KM = 6371.0

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2 * R_EARTH_KM * math.asin(math.sqrt(a))

def calculate_distance_nm(lat1, lon1, lat2, lon2):
    return haversine(lat1, lon1, lat2, lon2) / 1.852

def bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    y = math.sin(lon2 - lon1) * math.cos(lat2)
    x = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(lon2 - lon1)
    return (math.degrees(math.atan2(y, x)) + 360) % 360

# -----------------------------------------------------------------------------
# Interface Streamlit
# -----------------------------------------------------------------------------

def main():
    st.title("YB Tracking Analyzer")
    st.write("Analysez les données de courses YB Tracking")

    with st.sidebar:
        race_id = st.text_input("Identifiant de la course", value="dgbr2025")
        if st.button("Télécharger les données"):
            if not race_id:
                st.error("Veuillez saisir un identifiant de course.")
            else:
                st.session_state["data"] = fetch_yb_data(race_id)

    # Affichage des résultats --------------------------------------------------
    if "data" in st.session_state:
        d = st.session_state["data"]
        if d["RaceSetup"] is None:
            st.warning("RaceSetup absent : l'identifiant semble incorrect ou la course n'est pas encore publiée.")
            return
        st.success("Données JSON principales chargées.")

        # Ici vous pouvez poursuivre avec les traitements existants
        st.write("TODO : Ajoutez vos analyses (vitesses, carte, etc.) …")

if __name__ == "__main__":
    main()
