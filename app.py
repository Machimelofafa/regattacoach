import streamlit as st
import requests
import time
import math
import struct
from datetime import datetime
import json
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
# Petites fonctions utilitaires réseau
# -----------------------------------------------------------------------------

HEADERS = {"User-Agent": "YB-Analyzer/1.0"}

@st.cache_data(ttl=300, show_spinner=False)
def safe_get_json(url: str, max_retries: int = 3, backoff: int = 2):
    """GET JSON avec retries. Renvoie None si le serveur répond 500."""
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, timeout=30, headers=HEADERS)
            if r.status_code == 500:
                return None
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == max_retries:
                st.error(f"{url.split('/')[-1]} indisponible : {e}")
                return None
            time.sleep(backoff ** attempt)

@st.cache_data(ttl=300, show_spinner=False)
def safe_get_binary(url: str, max_retries: int = 3, backoff: int = 2):
    """Télécharge un fichier binaire (AllPositions3)."""
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
# Décodage du binaire AllPositions3 (portage Python du C fourni)
# -----------------------------------------------------------------------------

def _read(fmt: str, buf: bytes, offset: int):
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, buf[offset:offset + size])[0], offset + size


def decode_all_positions(binary: bytes) -> dict:
    """Décodage complet du fichier AllPositions3 → dict JSON.
    Portage direct du fichier *decyb.c* fourni (MIT) vers Python.
    Structure retournée : {"boats": [{"id": int, "positions": [ ... ]}]}"""

    # Décompression automatique si besoin (gzip ou zlib)
    if binary[:2] == b"\x1f\x8b":
        import gzip
        binary = gzip.decompress(binary)
    elif binary[:2] in (b"\x78\x01", b"\x78\x9c", b"\x78\xda"):
        import zlib
        binary = zlib.decompress(binary)

    buf = binary  # alias
    pos = 0

    # Flags d'en‑tête ----------------------------------------------------------
    flags, pos = _read("!B", buf, pos)  # 1 octet non signé big‑endian
    a = bool(flags & 1)   # alt
    s = bool(flags & 2)   # dtf
    n = bool(flags & 4)   # lap
    r = bool(flags & 8)   # pc (performance?)

    ref_time, pos = _read("!I", buf, pos)  # uint32 big‑endian

    boats = []
    # Boucle sur chaque bateau -------------------------------------------------
    while pos < len(buf):
        boat_id, pos = _read("!H", buf, pos)         # uint16
        n_points, pos = _read("!H", buf, pos)        # uint16

        positions = []
        prev = {
            "lat": 0,
            "lon": 0,
            "at": 0,
            "dtf": 0,
            "pc": 0.0,
            "alt": 0,
            "lap": 0,
        }

        for _ in range(n_points):
            header_byte, pos = _read("!B", buf, pos)
            moment = {}

            if header_byte & 0x80:  # delta‑encoded point ----------------------
                w, pos = _read("!H", buf, pos)
                y, pos = _read("!h", buf, pos)  # int16
                M, pos = _read("!h", buf, pos)

                if a:
                    alt, pos = _read("!h", buf, pos)
                    moment["alt"] = alt
                if s:
                    f, pos = _read("!h", buf, pos)
                    moment["dtf"] = prev["dtf"] + f
                    if n:
                        lap, pos = _read("!B", buf, pos)
                        moment["lap"] = lap
                if r:
                    pc_delta, pos = _read("!h", buf, pos)
                    moment["pc"] = prev["pc"] + pc_delta / 32000.0

                w &= 0x7FFF  # 15 bits
                moment["lat"] = prev["lat"] + y
                moment["lon"] = prev["lon"] + M
                moment["at"] = prev["at"] - w
                # pc déjà mis à jour ci‑dessus
            else:  # point absolu ---------------------------------------------
                T, pos = _read("!I", buf, pos)
                b, pos = _read("!i", buf, pos)
                L, pos = _read("!i", buf, pos)

                if a:
                    alt, pos = _read("!h", buf, pos)
                    moment["alt"] = alt
                if s:
                    x, pos = _read("!i", buf, pos)
                    moment["dtf"] = x
                    if n:
                        lap, pos = _read("!B", buf, pos)
                        moment["lap"] = lap
                if r:
                    pc_val, pos = _read("!i", buf, pos)
                    moment["pc"] = pc_val / 21000000.0

                moment["lat"] = b
                moment["lon"] = L
                moment["at"] = ref_time + T

            # Mise à l'échelle et mémoire du précédent ------------------------
            moment["lat"] = moment["lat"] / 1e5
            moment["lon"] = moment["lon"] / 1e5
            positions.append(moment)

            # update prev with defaults for missing keys
            for key in ("lat", "lon", "at", "dtf", "pc", "alt", "lap"):
                prev[key] = moment.get(key, prev.get(key))

        boats.append({"id": boat_id, "positions": positions})

    return {"boats": boats}

# -----------------------------------------------------------------------------
# Construction de LatestPositions à partir de l'historique complet
# -----------------------------------------------------------------------------

def build_latest_from_all_positions(all_positions: dict):
    latest = {"positions": []}
    for boat in all_positions.get("boats", []):
        if boat["positions"]:
            last = boat["positions"][-1].copy()
            last["boatID"] = boat["id"]
            latest["positions"].append(last)
    return latest if latest["positions"] else None

# -----------------------------------------------------------------------------
# Téléchargement principal (JSON + binaire)
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
    bin_url = f"https://cf.yb.tl/BIN/{race_id}/AllPositions3"
    with st.spinner("Téléchargement du binaire AllPositions3 …"):
        binary = safe_get_binary(bin_url)
        if binary:
            data["AllPositions"] = decode_all_positions(binary)
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
# Fonctions géodésiques
# -----------------------------------------------------------------------------

R_EARTH_KM = 6371.0

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R_EARTH_KM * math.asin(math.sqrt(a))

def distance_nm(lat1, lon1, lat2, lon2):
    return haversine(lat1, lon1, lat2, lon2) / 1.852

def bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    y = math.sin(lon2 - lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
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
                st.session_state["data"] = fetch_yb_data(race_id.strip())

    # Affichage d'un résumé ----------------------------------------------------
    if "data" in st.session_state:
        d = st.session_state["data"]
        if d.get("RaceSetup") is None:
            st.warning("RaceSetup absent : l'identifiant semble incorrect ou la course n'est pas encore publiée.")
            return
        st.success("Données principales chargées !")

        # Exemple : afficher la liste des bateaux avec leur dernière position
        if d.get("LatestPositions"):
            latest_df = pd.DataFrame(d["LatestPositions"]["positions"])
            st.write("**Dernières positions disponibles :**", latest_df.head())
        else:
            st.info("Aucune position disponible pour le moment.")

        st.write("TODO : Ajoutez vos propres analyses (cartes, graphes, vitesses, etc.) …")


if __name__ == "__main__":
    main()
