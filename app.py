import streamlit as st
import requests
import time
import math
import struct
import json
from datetime import datetime
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static

# ------------------------------------------------------------
# Configuration Streamlit
# ------------------------------------------------------------

st.set_page_config(
    page_title="YB Tracking Analyzer",
    page_icon="⛵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Constantes & HTTP helpers
# ------------------------------------------------------------

HEADERS = {"User-Agent": "YB-Analyzer/1.0"}
GZIP_MAGIC = b"\x1f\x8b"
ZLIB_MAGICS = (b"\x78\x01", b"\x78\x9c", b"\x78\xda")

@st.cache_data(ttl=300, show_spinner=False)
def safe_get_json(url: str, max_retries: int = 3, backoff: int = 2):
    """GET JSON avec retries. Renvoie None si 500 ou erreur."""
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
            r = requests.get(url, timeout=60, headers=HEADERS)
            if r.status_code == 500:
                return None
            r.raise_for_status()
            return r.content
        except Exception as e:
            if attempt == max_retries:
                st.error(f"{url.split('/')[-1]} indisponible : {e}")
                return None
            time.sleep(backoff ** attempt)

# ------------------------------------------------------------
# Lecture binaire utilitaire
# ------------------------------------------------------------

def _read(fmt: str, buf: memoryview, offset: int):
    """Lit `fmt` à `offset` dans buf et renvoie (valeur, new_offset).
    Lève ValueError si dépassement de tampon."""
    size = struct.calcsize(fmt)
    if offset + size > len(buf):
        raise ValueError("buffer underrun")
    return struct.unpack_from(fmt, buf, offset)[0], offset + size

# ------------------------------------------------------------
# Décodage AllPositions3  (portage léger de decyb.c)
# ------------------------------------------------------------

def decode_all_positions(binary: bytes) -> dict:
    """Décodage complet du fichier AllPositions3 → dict JSON.
    Tente d'abord la décompression (gzip/zlib), puis essaye JSON direct,
    sinon parse le format binaire delta‑encodé inspiré de decyb.c."""

    # 1) décompression éventuelle -------------------------------------------
    if binary[:2] == GZIP_MAGIC:
        import gzip
        binary = gzip.decompress(binary)
    elif binary[:2] in ZLIB_MAGICS:
        import zlib
        binary = zlib.decompress(binary)

    # 2) JSON direct ---------------------------------------------------------
    if binary[:1] in (b"{", b"["):
        try:
            return json.loads(binary.decode())
        except Exception as e:
            st.warning(f"AllPositions3 semble être du JSON, échec parse : {e}")
            return {}

    # 3) Format binaire ------------------------------------------------------
    buf = memoryview(binary)
    pos = 0

    try:
        flags, pos = _read("!B", buf, pos)
    except ValueError:
        st.error("AllPositions3 vide ou corrompu (pas d'en‑tête)")
        return {}

    # Décodage des flags
    has_alt = bool(flags & 1)
    has_dtf = bool(flags & 2)
    has_lap = bool(flags & 4)
    has_pc  = bool(flags & 8)

    ref_time, pos = _read("!I", buf, pos)  # uint32

    boats = []
    try:
        while pos < len(buf):
            boat_id, pos   = _read("!H", buf, pos)
            n_points, pos  = _read("!H", buf, pos)

            positions = []
            prev = {"lat": 0, "lon": 0, "at": 0, "dtf": 0, "pc": 0.0, "alt": 0, "lap": 0}

            for _ in range(n_points):
                header, pos = _read("!B", buf, pos)
                moment = {}

                if header & 0x80:  # delta‑encoded --------------------------------
                    w , pos = _read("!H", buf, pos)
                    dy, pos = _read("!h", buf, pos)
                    dx, pos = _read("!h", buf, pos)

                    if has_alt:
                        alt, pos = _read("!h", buf, pos)
                        moment["alt"] = alt
                    if has_dtf:
                        d_dtf, pos = _read("!h", buf, pos)
                        moment["dtf"] = prev["dtf"] + d_dtf
                        if has_lap:
                            lap, pos = _read("!B", buf, pos)
                            moment["lap"] = lap
                    if has_pc:
                        d_pc, pos = _read("!h", buf, pos)
                        moment["pc"] = prev["pc"] + d_pc / 32000.0

                    w &= 0x7FFF
                    moment["lat"] = prev["lat"] + dy
                    moment["lon"] = prev["lon"] + dx
                    moment["at"]  = prev["at"]  - w
                else:  # point absolu -----------------------------------------
                    T , pos = _read("!I", buf, pos)
                    b , pos = _read("!i", buf, pos)
                    L , pos = _read("!i", buf, pos)

                    if has_alt:
                        alt, pos = _read("!h", buf, pos)
                        moment["alt"] = alt
                    if has_dtf:
                        dtf, pos = _read("!i", buf, pos)
                        moment["dtf"] = dtf
                        if has_lap:
                            lap, pos = _read("!B", buf, pos)
                            moment["lap"] = lap
                    if has_pc:
                        pc_val, pos = _read("!i", buf, pos)
                        moment["pc"] = pc_val / 21000000.0

                    moment["lat"] = b
                    moment["lon"] = L
                    moment["at"]  = ref_time + T

                # Mise à l'échelle & stockage --------------------------------
                moment["lat"] /= 1e5
                moment["lon"] /= 1e5
                positions.append(moment)

                # maj prev
                for k, v in moment.items():
                    prev[k] = v

            boats.append({"id": boat_id, "positions": positions})

    except (ValueError, struct.error) as e:
        st.warning(f"Décodage interrompu : {e}")

    return {"boats": boats}

# ------------------------------------------------------------
# Helpers LatestPositions
# ------------------------------------------------------------

def build_latest_from_all_positions(all_positions: dict):
    latest = {"positions": []}
    for boat in all_positions.get("boats", []):
        if boat["positions"]:
            last = boat["positions"][-1].copy()
            last["boatID"] = boat["id"]
            latest["positions"].append(last)
    return latest if latest["positions"] else None

# ------------------------------------------------------------
# Téléchargement principal
# ------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def fetch_yb_data(race_id: str):
    data = {}
    base_json = f"https://cf.yb.tl/JSON/{race_id}"
    urls = {
        "RaceSetup":       f"{base_json}/RaceSetup",
        "Leaderboard":     f"{base_json}/leaderboard",
        "LatestPositions": f"{base_json}/LatestPositions",
    }

    with st.spinner("Téléchargement des JSON …"):
        for name, url in urls.items():
            data[name] = safe_get_json(url)
            if data[name] is None:
                st.info(f"{name} indisponible – HTTP 500 (ou 404) renvoyé par YB.")

    # Binaire -----------------------------------------------------------------
    bin_url = f"https://cf.yb.tl/BIN/{race_id}/AllPositions3"
    with st.spinner("Téléchargement du binaire AllPositions3 …"):
        binary = safe_get_binary(bin_url)
        if binary:
            data["AllPositions"] = decode_all_positions(binary)
        else:
            data["AllPositions"] = None
            st.info("AllPositions3 indisponible – course à venir ou ID erroné.")

    # Reconstruit LatestPositions si besoin -----------------------------------
    if data.get("LatestPositions") is None and data.get("AllPositions"):
        rebuilt = build_latest_from_all_positions(data["AllPositions"])
        if rebuilt:
            data["LatestPositions"] = rebuilt
            st.success("LatestPositions reconstruit à partir du binaire.")

    return data

# ------------------------------------------------------------
# Interface Streamlit
# ------------------------------------------------------------

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

    # Résumé ------------------------------------------------------------------
    if "data" in st.session_state:
        d = st.session_state["data"]
        if d.get("RaceSetup") is None:
            st.warning("RaceSetup absent : identifiant incorrect ou course pas encore publiée.")
            return
        st.success("Données principales chargées !")

        if d.get("LatestPositions"):
            latest_df = pd.DataFrame(d["LatestPositions"]["positions"])
            st.write("**Dernières positions disponibles :**", latest_df.head())
        else:
            st.info("Aucune position disponible pour le moment.")

        st.write("TODO : Ajoutez vos propres analyses (carte, graphes, vitesses, etc.) …")


if __name__ == "__main__":
    main()
