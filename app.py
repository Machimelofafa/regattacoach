# app.py
# ---------------------------------------------------------------------------
# Analyseur YB Tracking – version 2025-06-01
# ---------------------------------------------------------------------------
import gzip, zlib, json, math, struct, io, time
from functools import lru_cache
from typing import Dict, List

import pandas as pd
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
ENDPOINTS = {
    "RaceSetup"      : "https://cf.yb.tl/JSON/{race}/RaceSetup",
    "Leaderboard"    : "https://cf.yb.tl/JSON/{race}/leaderboard",
    "LatestPositions": "https://cf.yb.tl/JSON/{race}/LatestPositions",
    "AllPositions3"  : "https://cf.yb.tl/BIN/{race}/AllPositions3",
}

GZIP_MAGIC = b"\x1f\x8b"
ZLIB_MAGIC = b"\x78\x9c"  # deflate « vanilla »

# ---------------------------------------------------------------------------
# Téléchargement avec gestion des erreurs YB
# ---------------------------------------------------------------------------
def safe_get(url: str, binary: bool = False, max_retries: int = 3, backoff: int = 2):
    """Télécharge (JSON ou bytes) en gérant les erreurs 500 renvoyées par YB."""
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, timeout=30, headers={"User-Agent": "YB-Analyzer/1.0"})
            if r.status_code in (500, 404):
                return None
            r.raise_for_status()
            return r.content if binary else r.json()
        except requests.RequestException as e:
            if attempt == max_retries:
                raise
            time.sleep(backoff ** attempt)

# ---------------------------------------------------------------------------
# Helpers little-endian
# ---------------------------------------------------------------------------
LE = "<"
def _u16(buf, pos): return struct.unpack_from(LE + "H", buf, pos)[0], pos + 2
def _i16(buf, pos): return struct.unpack_from(LE + "h", buf, pos)[0], pos + 2
def _u32(buf, pos): return struct.unpack_from(LE + "I", buf, pos)[0], pos + 4
def _i32(buf, pos): return struct.unpack_from(LE + "i", buf, pos)[0], pos + 4

# ---------------------------------------------------------------------------
# Décodage AllPositions3 -> dict JSON
# ---------------------------------------------------------------------------
def decode_all_positions(binary: bytes) -> Dict:
    """Décode AllPositions3 vers un dict proche de l’API JSON."""
    if not binary:
        return {}

    # Décompression éventuelle ------------------------------------------------
    if binary.startswith(GZIP_MAGIC):
        binary = gzip.decompress(binary)
    elif binary.startswith(ZLIB_MAGIC):
        binary = zlib.decompress(binary)

    # Certains dumps anciens sont déjà du JSON brut --------------------------
    first = binary[:1]
    if first in (b"{", b"["):
        try:
            return json.loads(binary.decode())
        except Exception:
            pass  # on tentera le parseur binaire

    buf = memoryview(binary)
    pos = 0
    boats: List[Dict] = []

    try:
        n_boats, pos = _u16(buf, pos)
        for _ in range(n_boats):
            boat_id, pos = _u32(buf, pos)
            n_pts,  pos = _u32(buf, pos)

            positions = []
            last_lat = last_lon = 0

            for _ in range(n_pts):
                # Header
                flags, pos = _u16(buf, pos)
                # delta relative ou absolue ?
                if flags & 0x8000:                 # delta-encoded
                    dy, pos = _i16(buf, pos)
                    dx, pos = _i16(buf, pos)
                    lat = last_lat + dy
                    lon = last_lon + dx
                else:                              # valeurs absolues
                    lat, pos = _i32(buf, pos)
                    lon, pos = _i32(buf, pos)

                last_lat, last_lon = lat, lon
                lat_deg = lat / 1e6
                lon_deg = lon / 1e6

                # timestamp (mandatory)
                at, pos = _u32(buf, pos)

                msg: Dict = {"lat": lat_deg, "lon": lon_deg, "at": at}

                # champs optionnels
                if flags & 0x0001:  # dtf
                    dtf, pos = _i32(buf, pos)
                    msg["dtf"] = dtf
                if flags & 0x0002:  # alt
                    alt, pos = _i32(buf, pos)
                    msg["alt"] = alt
                if flags & 0x0004:  # lap
                    lap, pos = _u16(buf, pos)
                    msg["lap"] = lap
                if flags & 0x0008:  # pc
                    pc, pos = _u16(buf, pos)
                    msg["pc"] = pc

                # Filtrer les points aberrants
                if -90 <= lat_deg <= 90 and -180 <= lon_deg <= 180:
                    positions.append(msg)

            boats.append({"id": boat_id, "positions": positions})

    except (struct.error, IndexError):
        st.warning("Décodage interrompu : buffer underrun – fichier tronqué ?")
    return {"boats": boats}

# ---------------------------------------------------------------------------
# Reconstruction LatestPositions
# ---------------------------------------------------------------------------
def build_latest_from_all_positions(all_pos: Dict) -> Dict:
    latest = {"positions": []}
    for boat in all_pos.get("boats", []):
        if boat["positions"]:
            latest["positions"].append({**boat["positions"][-1], "boatID": boat["id"]})
    return latest

# ---------------------------------------------------------------------------
# Calcul vitesse / distance
# ---------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 3440.065  # rayon Terre NM
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ   = math.radians(lat2 - lat1)
    dλ   = math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def enrich_positions(positions: List[Dict]) -> List[Dict]:
    for i in range(1, len(positions)):
        p0, p1 = positions[i-1], positions[i]
        dist_nm = haversine(p0["lat"], p0["lon"], p1["lat"], p1["lon"])
        dt_h = max((p1["at"] - p0["at"]) / 3600, 1e-6)
        p1["dist_nm"] = dist_nm
        p1["speed_kn"] = dist_nm / dt_h
    return positions

# ---------------------------------------------------------------------------
# Téléchargement principal (cache 5 min)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_yb_data(race_id: str) -> Dict:
    data: Dict = {}

    # RaceSetup sert de présence/validation
    setup_url = ENDPOINTS["RaceSetup"].format(race=race_id)
    with st.spinner("Vérification de l’ID de course…"):
        setup = safe_get(setup_url)
    if setup is None:
        st.error("Course inconnue ou pas encore publiée.")
        return {}

    data["RaceSetup"] = setup

    # Leaderboard
    with st.spinner("Leaderboard…"):
        data["Leaderboard"] = safe_get(ENDPOINTS["Leaderboard"].format(race=race_id))

    # LatestPositions (facultatif)
    with st.spinner("LatestPositions…"):
        lp = safe_get(ENDPOINTS["LatestPositions"].format(race=race_id))
        if lp is None:
            st.info("LatestPositions indisponible – on reconstruira plus tard.")
        data["LatestPositions"] = lp

    # AllPositions3 binaire
    with st.spinner("AllPositions3 (binaire)…"):
        bin_blob = safe_get(
            ENDPOINTS["AllPositions3"].format(race=race_id),
            binary=True
        )
        allpos = decode_all_positions(bin_blob)
        data["AllPositions"] = allpos

    # Reconstruction LP si besoin
    if data["LatestPositions"] is None and allpos:
        data["LatestPositions"] = build_latest_from_all_positions(allpos)
        st.success("LatestPositions reconstruit à partir du binaire.")

    # Ajout vitesses / distances
    for boat in data["AllPositions"].get("boats", []):
        boat["positions"] = enrich_positions(boat["positions"])

    return data

# ---------------------------------------------------------------------------
# Interface Streamlit
# ---------------------------------------------------------------------------
st.set_page_config(page_title="YB Tracking Analyzer", layout="wide")

st.sidebar.header("Configuration")
race_id = st.sidebar.text_input("Identifiant de la course", value="dgbr2025")
if st.sidebar.button("Télécharger les données"):
    st.session_state["data"] = fetch_yb_data(race_id.strip())

if st.sidebar.button("Réinitialiser"):
    st.session_state.clear()
    st.experimental_rerun()

data = st.session_state.get("data")
if not data:
    st.info("Saisissez un ID et cliquez sur *Télécharger les données*.")
    st.stop()

boats_meta = {b["id"]: f"Bateau {i+1}" for i, b in enumerate(data["AllPositions"]["boats"])}
boat_ids = list(boats_meta.keys())

st.sidebar.subheader("Filtrage")
selected_boats = st.sidebar.multiselect(
    "Bateaux à analyser",
    options=boat_ids,
    default=boat_ids[:1],
    format_func=lambda x: boats_meta.get(x, str(x)),
)

analysis_type = st.sidebar.selectbox("Type d’analyse", ["Carte", "Tableau comparatif"])

# ---------------------------------------------------------------------------
# Affichage Carte
# ---------------------------------------------------------------------------
st.title("YB Tracking Analyzer")
st.caption("Analysez les données de courses YB Tracking")

if analysis_type == "Carte":
    st.header("Carte des trajectoires")

    m = folium.Map(location=[50, -2], zoom_start=5, tiles="OpenStreetMap")
    colors = ["red", "blue", "green", "purple", "orange", "darkred"]

    for idx, boat in enumerate(data["AllPositions"]["boats"]):
        if boat["id"] not in selected_boats or not boat["positions"]:
            continue
        pts = [(p["lat"], p["lon"]) for p in boat["positions"]]
        folium.PolyLine(pts, weight=2, color=colors[idx % len(colors)]).add_to(m)
        folium.Marker(pts[-1], icon=folium.Icon(color=colors[idx % len(colors)], icon="ship", prefix="fa"),
                      tooltip=boats_meta.get(boat["id"], boat["id"])).add_to(m)

    st_folium(m, width=900, height=600)

# ---------------------------------------------------------------------------
# Tableau comparatif
# ---------------------------------------------------------------------------
else:
    st.header("Tableau comparatif")

    rows = []
    for boat in data["AllPositions"]["boats"]:
        if boat["id"] not in selected_boats or not boat["positions"]:
            continue
        df = pd.DataFrame(boat["positions"])
        dist_total = df["dist_nm"].sum()
        mean_speed = df["speed_kn"].mean()
        max_speed = df["speed_kn"].max()
        rows.append({
            "Bateau": boats_meta.get(boat["id"], boat["id"]),
            "Vitesse moyenne (kn)": round(mean_speed, 2),
            "Vitesse max (kn)": round(max_speed, 2),
            "Distance parcourue (NM)": round(dist_total, 2),
        })

    comp_df = pd.DataFrame(rows)
    st.dataframe(comp_df, use_container_width=True)

    csv = comp_df.to_csv(index=False).encode()
    st.download_button("Exporter le tableau (CSV)", csv, file_name="comparatif.csv", mime="text/csv")
