import streamlit as st
import requests
import time
import struct
import json
from datetime import datetime
import pandas as pd

from folium.plugins import MarkerCluster
import folium
from streamlit_folium import folium_static

# ----------------- CONFIG -----------------

st.set_page_config(
    page_title="YB Tracking Analyzer",
    page_icon="⛵",
    layout="wide",
    initial_sidebar_state="expanded",
)

HEADERS = {"User-Agent": "YB-Analyzer/1.0"}
GZIP_MAGIC = b"\x1f\x8b"
ZLIB_MAGICS = (b"\x78\x01", b"\x78\x9c", b"\x78\xda")

# --------------- HTTP utils --------------

@st.cache_data(ttl=300, show_spinner=False)
def safe_get_json(url: str, retries: int = 3):
    backoff = 2
    for i in range(retries):
        try:
            r = requests.get(url, timeout=30, headers=HEADERS)
            if r.status_code == 500:
                return None
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == retries - 1:
                return None
            time.sleep(backoff ** i)

@st.cache_data(ttl=300, show_spinner=False)
def safe_get_binary(url: str, retries: int = 3):
    backoff = 2
    for i in range(retries):
        try:
            r = requests.get(url, timeout=60, headers=HEADERS)
            if r.status_code == 500:
                return None
            r.raise_for_status()
            return r.content
        except Exception:
            if i == retries - 1:
                return None
            time.sleep(backoff ** i)

# -------- Binary helper --------

def _read(fmt: str, buf: memoryview, pos: int):
    size = struct.calcsize(fmt)
    if pos + size > len(buf):
        raise ValueError("buffer underrun")
    return struct.unpack_from(fmt, buf, pos)[0], pos + size

# -------- Decoder -------------

def decode_all_positions(binary: bytes) -> dict:
    """Return {"boats": [...]}. Handle gzip/zlib/raw/JSON."""

    # decompress
    if binary[:2] == GZIP_MAGIC:
        import gzip
        binary = gzip.decompress(binary)
    elif binary[:2] in ZLIB_MAGICS:
        import zlib
        binary = zlib.decompress(binary)

    # already JSON?
    if binary[:1] in (b"{", b"["):
        try:
            return json.loads(binary.decode())
        except Exception:
            return {}

    buf = memoryview(binary)
    pos = 0
    try:
        flags, pos = _read("!B", buf, pos)
    except ValueError:
        return {}

    has_alt = flags & 1
    has_dtf = flags & 2
    has_lap = flags & 4
    has_pc  = flags & 8

    ref_time, pos = _read("!I", buf, pos)

    boats = []
    try:
        while pos < len(buf):
            boat_id, pos = _read("!H", buf, pos)
            n_pts,   pos = _read("!H", buf, pos)
            positions = []
            prev = {"lat": 0, "lon": 0, "at": 0, "dtf": 0, "pc": 0.0, "alt": 0, "lap": 0}
            for _ in range(n_pts):
                header, pos = _read("!B", buf, pos)
                m = {}
                if header & 0x80:
                    w , pos = _read("!H", buf, pos)
                    dy, pos = _read("!h", buf, pos)
                    dx, pos = _read("!h", buf, pos)
                    if has_alt:
                        alt, pos = _read("!h", buf, pos)
                        m["alt"] = alt
                    if has_dtf:
                        d_dtf, pos = _read("!h", buf, pos)
                        m["dtf"] = prev["dtf"] + d_dtf
                        if has_lap:
                            lap, pos = _read("!B", buf, pos)
                            m["lap"] = lap
                    if has_pc:
                        d_pc, pos = _read("!h", buf, pos)
                        m["pc"] = prev["pc"] + d_pc / 32000.0
                    w &= 0x7FFF
                    m["lat"] = prev["lat"] + dy
                    m["lon"] = prev["lon"] + dx
                    m["at"]  = prev["at"] - w
                else:
                    T , pos = _read("!I", buf, pos)
                    lat_i, pos = _read("!i", buf, pos)
                    lon_i, pos = _read("!i", buf, pos)
                    if has_alt:
                        alt, pos = _read("!h", buf, pos)
                        m["alt"] = alt
                    if has_dtf:
                        dtf, pos = _read("!i", buf, pos)
                        m["dtf"] = dtf
                        if has_lap:
                            lap, pos = _read("!B", buf, pos)
                            m["lap"] = lap
                    if has_pc:
                        pc_val, pos = _read("!i", buf, pos)
                        m["pc"] = pc_val / 21000000.0
                    m["lat"] = lat_i
                    m["lon"] = lon_i
                    m["at"]  = ref_time + T
                m["lat"] /= 1e5
                m["lon"] /= 1e5
                positions.append(m)
                for k, v in m.items():
                    prev[k] = v
            boats.append({"id": boat_id, "positions": positions})
    except ValueError as e:
        st.warning(f"Décodage interrompu : {e}")
    return {"boats": boats}

# ------- build latest ---------

def build_latest(all_pos: dict):
    latest = {"positions": []}
    for b in all_pos.get("boats", []):
        if b["positions"]:
            last = b["positions"][-1].copy()
            last["boatID"] = b["id"]
            latest["positions"].append(last)
    return latest if latest["positions"] else None

# ------- fetch ---------------

@st.cache_data(ttl=300)
def fetch(race_id: str):
    base = f"https://cf.yb.tl/JSON/{race_id}"
    data = {name: safe_get_json(f"{base}/{name}") for name in ["RaceSetup", "leaderboard", "LatestPositions"]}
    bin_file = safe_get_binary(f"https://cf.yb.tl/BIN/{race_id}/AllPositions3")
    data["AllPositions"] = decode_all_positions(bin_file) if bin_file else None
    if data.get("LatestPositions") is None and data.get("AllPositions"):
        rebuilt = build_latest(data["AllPositions"])
        if rebuilt:
            data["LatestPositions"] = rebuilt
    return data

# ---------- UI ---------------

def main():
    st.title("YB Tracking Analyzer")
    st.write("Analysez les données de courses YB Tracking")
    with st.sidebar:
        race_id = st.text_input("Identifiant de la course", "dgbr2025")
        if st.button("Télécharger les données"):
            if race_id.strip():
                st.session_state.data = fetch(race_id.strip())
            else:
                st.error("Veuillez saisir un identifiant.")
    if "data" in st.session_state:
        d = st.session_state.data
        if not d.get("RaceSetup"):
            st.warning("RaceSetup manquant – ID erroné ou course non publiée.")
            return
        st.success("Données principales chargées !")
        if d.get("LatestPositions"):
            st.dataframe(pd.DataFrame(d["LatestPositions"]["positions"]).head())
        else:
            st.info("Aucune position disponible.")
        # TODO: add map/graphs

if __name__ == "__main__":
    main()
