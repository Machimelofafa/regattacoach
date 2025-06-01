import streamlit as st
import requests
import time
import struct
import json
from datetime import datetime
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

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
    """Récupère des données JSON avec gestion des erreurs et retries"""
    backoff = 2
    for i in range(retries):
        try:
            with st.spinner(f"Téléchargement de {url.split('/')[-1]}..."):
                r = requests.get(url, timeout=30, headers=HEADERS)
                if r.status_code == 500:
                    st.warning(f"Erreur 500 pour {url}")
                    return None
                r.raise_for_status()
                return r.json()
        except Exception as e:
            if i == retries - 1:
                st.error(f"Échec du téléchargement après {retries} tentatives: {str(e)}")
                return None
            time.sleep(backoff ** i)
            st.warning(f"Tentative {i+1} échouée pour {url.split('/')[-1]}: {str(e)}. Nouvelle tentative dans {backoff ** i}s...")

@st.cache_data(ttl=300, show_spinner=False)
def safe_get_binary(url: str, retries: int = 3, chunk_size: int = 1024*1024):
    """Récupère des données binaires avec gestion des erreurs, retries et téléchargement par morceaux"""
    backoff = 2
    for i in range(retries):
        try:
            with st.spinner(f"Téléchargement de l'historique des positions..."):
                # Téléchargement par morceaux pour éviter les timeouts
                r = requests.get(url, stream=True, timeout=60, headers=HEADERS)
                if r.status_code == 500:
                    st.warning(f"Erreur 500 pour {url}")
                    return None
                r.raise_for_status()
                
                # Obtenir la taille totale si disponible
                total_size = int(r.headers.get('content-length', 0))
                
                # Lecture par morceaux
                chunks = []
                downloaded = 0
                
                progress_bar = st.progress(0, text="Téléchargement: 0 Mo")
                
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        chunks.append(chunk)
                        downloaded += len(chunk)
                        if total_size:
                            progress = min(downloaded / total_size, 1.0)
                            progress_bar.progress(progress, text=f"Téléchargement: {downloaded/1024/1024:.1f} Mo")
                
                binary_data = b''.join(chunks)
                progress_bar.empty()
                return binary_data
                
        except Exception as e:
            if i == retries - 1:
                st.error(f"Échec du téléchargement après {retries} tentatives: {str(e)}")
                return None
            time.sleep(backoff ** i)
            st.warning(f"Tentative {i+1} échouée pour l'historique des positions: {str(e)}. Nouvelle tentative dans {backoff ** i}s...")

# -------- Binary helper --------

def _read(fmt: str, buf: memoryview, pos: int):
    """Lit une valeur formatée depuis un buffer binaire avec vérification de taille"""
    size = struct.calcsize(fmt)
    if pos + size > len(buf):
        raise ValueError("buffer underrun")
    return struct.unpack_from(fmt, buf, pos)[0], pos + size

# -------- Decoder -------------

def decode_all_positions(binary: bytes) -> dict:
    """
    Décode le fichier binaire AllPositions3 avec gestion robuste des erreurs
    Return {"boats": [...]}. Handle gzip/zlib/raw/JSON.
    """
    if binary is None or len(binary) < 10:
        st.error("Données binaires invalides ou manquantes")
        return {"boats": []}

    # Décompression si nécessaire
    try:
        if binary[:2] == GZIP_MAGIC:
            import gzip
            binary = gzip.decompress(binary)
        elif binary[:2] in ZLIB_MAGICS:
            import zlib
            binary = zlib.decompress(binary)
    except Exception as e:
        st.error(f"Erreur de décompression: {str(e)}")
        return {"boats": []}

    # Vérifier si c'est déjà du JSON
    if binary[:1] in (b"{", b"["):
        try:
            return json.loads(binary.decode())
        except Exception:
            st.error("Format JSON invalide")
            return {"boats": []}

    buf = memoryview(binary)
    pos = 0
    boats = []
    
    try:
        # Lecture des flags
        flags, pos = _read("!B", buf, pos)
        
        has_alt = flags & 1
        has_dtf = flags & 2
        has_lap = flags & 4
        has_pc  = flags & 8
        
        # Lecture du temps de référence
        ref_time, pos = _read("!I", buf, pos)
        
        # Lecture des bateaux
        boat_count = 0
        max_boats = 200  # Limite de sécurité
        
        while pos < len(buf) and boat_count < max_boats:
            boat_count += 1
            
            # Lecture de l'ID du bateau et du nombre de points
            boat_id, pos = _read("!H", buf, pos)
            n_pts, pos = _read("!H", buf, pos)
            
            # Limite de sécurité pour le nombre de points
            n_pts = min(n_pts, 10000)
            
            positions = []
            prev = {"lat": 0, "lon": 0, "at": 0, "dtf": 0, "pc": 0.0, "alt": 0, "lap": 0}
            
            for _ in range(n_pts):
                header, pos = _read("!B", buf, pos)
                m = {}
                
                if header & 0x80:  # Position relative
                    w, pos = _read("!H", buf, pos)
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
                    m["at"] = prev["at"] - w
                
                else:  # Position absolue
                    T, pos = _read("!I", buf, pos)
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
                    m["at"] = ref_time + T
                
                # Conversion des coordonnées
                m["lat"] /= 1e5
                m["lon"] /= 1e5
                
                # Vérification de validité des coordonnées
                if -90 <= m["lat"] <= 90 and -180 <= m["lon"] <= 180:
                    positions.append(m)
                    for k, v in m.items():
                        prev[k] = v
            
            if positions:  # N'ajouter que les bateaux avec des positions valides
                boats.append({"id": boat_id, "positions": positions})
    
    except ValueError as e:
        st.warning(f"Décodage interrompu : {e}")
        # Continuer avec les données partielles déjà décodées
    
    if not boats:
        st.warning("Aucune donnée de position n'a pu être décodée")
        # Créer des données de démonstration si aucune donnée n'a été décodée
        return create_demo_data()
    
    return {"boats": boats}

def create_demo_data():
    """Crée des données de démonstration si le décodage échoue complètement"""
    st.info("Utilisation de données de démonstration pour illustrer les fonctionnalités")
    
    now = int(datetime.now().timestamp())
    
    # Créer quelques bateaux de démonstration
    boats = []
    
    # Bateau 1 - MAORI III
    positions1 = []
    base_lat, base_lon = 50.7, -1.2
    for i in range(20):
        positions1.append({
            "lat": base_lat + i * 0.01,
            "lon": base_lon - i * 0.02,
            "at": now - (20-i) * 3600
        })
    boats.append({"id": 1, "name": "MAORI III", "positions": positions1})
    
    # Bateau 2 - CORA
    positions2 = []
    base_lat, base_lon = 50.71, -1.21
    for i in range(20):
        positions2.append({
            "lat": base_lat + i * 0.012,
            "lon": base_lon - i * 0.019,
            "at": now - (20-i) * 3600
        })
    boats.append({"id": 2, "name": "CORA", "positions": positions2})
    
    # Bateau 3 - F35 EXPRESS
    positions3 = []
    base_lat, base_lon = 50.69, -1.19
    for i in range(20):
        positions3.append({
            "lat": base_lat + i * 0.011,
            "lon": base_lon - i * 0.021,
            "at": now - (20-i) * 3600
        })
    boats.append({"id": 3, "name": "F35 EXPRESS", "positions": positions3})
    
    return {"boats": boats}

# ------- build latest ---------

def build_latest(all_pos: dict):
    """Construit un objet LatestPositions à partir des dernières positions de chaque bateau"""
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
    """Récupère toutes les données pour une course donnée"""
    base = f"https://cf.yb.tl/JSON/{race_id}"
    
    # Téléchargement des fichiers JSON
    data = {}
    
    # RaceSetup
    data["RaceSetup"] = safe_get_json(f"{base}/RaceSetup")
    if not data["RaceSetup"]:
        st.error(f"Impossible de récupérer les informations de la course {race_id}")
        return data
    
    # Leaderboard
    data["leaderboard"] = safe_get_json(f"{base}/leaderboard")
    
    # LatestPositions
    data["LatestPositions"] = safe_get_json(f"{base}/LatestPositions")
    
    # AllPositions3 (binaire)
    bin_file = safe_get_binary(f"https://cf.yb.tl/BIN/{race_id}/AllPositions3")
    data["AllPositions"] = decode_all_positions(bin_file) if bin_file else None
    
    # Si LatestPositions est manquant mais AllPositions est disponible, reconstruire LatestPositions
    if data.get("LatestPositions") is None and data.get("AllPositions"):
        rebuilt = build_latest(data["AllPositions"])
        if rebuilt:
            data["LatestPositions"] = rebuilt
    
    # Enrichir les données avec les noms des bateaux
    if data.get("AllPositions") and data.get("RaceSetup"):
        # Créer un dictionnaire de correspondance ID -> nom
        boat_names = {}
        for boat in data["RaceSetup"].get("boats", []):
            # Convertir l'ID en entier pour correspondre au format du fichier binaire
            try:
                boat_id = int(boat["id"])
                boat_names[boat_id] = boat.get("name", f"Bateau {boat_id}")
            except (ValueError, TypeError):
                # Si l'ID n'est pas un entier valide, utiliser l'ID original comme clé
                boat_names[boat["id"]] = boat.get("name", f"Bateau {boat['id']}")
        
        # Ajouter les noms aux bateaux décodés
        for boat in data["AllPositions"]["boats"]:
            # Essayer d'abord avec l'ID tel quel
            if boat["id"] in boat_names:
                boat["name"] = boat_names[boat["id"]]
            else:
                # Essayer avec l'ID converti en chaîne
                str_id = str(boat["id"])
                if str_id in boat_names:
                    boat["name"] = boat_names[str_id]
                else:
                    # Utiliser un nom générique si aucune correspondance n'est trouvée
                    boat["name"] = f"Bateau {boat['id']}"
    
    return data

# ---------- Fonctions d'analyse -------------

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calcule la distance entre deux points GPS (formule de Haversine)"""
    R = 6371.0  # Rayon de la Terre en km
    
    # Conversion en radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Différence de longitude et latitude
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    # Formule de Haversine
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Distance en km, convertie en milles nautiques
    distance_km = R * c
    distance_nm = distance_km / 1.852
    
    return distance_nm

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calcule le cap entre deux points GPS"""
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    y = math.sin(lon2_rad - lon1_rad) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(lon2_rad - lon1_rad)
    
    bearing = math.atan2(y, x)
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360
    
    return bearing

def calculate_speed_and_bearing(positions):
    """Calcule la vitesse et le cap pour chaque position"""
    if len(positions) < 2:
        return positions
    
    for i in range(1, len(positions)):
        prev_pos = positions[i-1]
        curr_pos = positions[i]
        
        # Calcul du temps écoulé en heures
        time_diff = (curr_pos["at"] - prev_pos["at"]) / (60 * 60)
        
        # Calcul de la distance en milles nautiques
        distance = calculate_distance(
            prev_pos["lat"], prev_pos["lon"],
            curr_pos["lat"], curr_pos["lon"]
        )
        
        # Calcul de la vitesse en nœuds
        if time_diff > 0:
            curr_pos["speed"] = distance / time_diff
        else:
            curr_pos["speed"] = 0
        
        # Calcul du cap
        curr_pos["bearing"] = calculate_bearing(
            prev_pos["lat"], prev_pos["lon"],
            curr_pos["lat"], curr_pos["lon"]
        )
    
    # Pour la première position, utiliser les mêmes valeurs que la deuxième
    if len(positions) > 1:
        positions[0]["speed"] = positions[1]["speed"]
        positions[0]["bearing"] = positions[1]["bearing"]
    
    return positions

def extract_boat_classes(race_setup):
    """Extrait les classes de bateaux à partir des données RaceSetup"""
    if not race_setup:
        return {}
    
    classes = {}
    
    # Extraction des classes (tags)
    if "tags" in race_setup:
        for tag in race_setup["tags"]:
            tag_id = tag.get("id")
            tag_name = tag.get("name")
            if tag_id and tag_name:
                classes[tag_id] = tag_name
    
    return classes

def extract_boats_by_class(race_setup, classes, all_positions=None):
    """
    Regroupe les bateaux par classe
    
    Args:
        race_setup: Données RaceSetup
        classes: Dictionnaire des classes {id: name}
        all_positions: Données AllPositions (optionnel)
    """
    if not race_setup:
        return {"Tous les bateaux": []}
    
    # Créer un dictionnaire pour stocker les bateaux par classe
    boats_by_class = {}
    
    # Ajouter une classe "Tous les bateaux"
    boats_by_class["Tous les bateaux"] = []
    
    # Ajouter les classes définies
    if classes:
        for class_name in classes.values():
            boats_by_class[class_name] = []
    
    # Extraction des bateaux depuis RaceSetup
    if "boats" in race_setup:
        for boat in race_setup["boats"]:
            boat_id = boat.get("id")
            boat_name = boat.get("name")
            boat_tags = boat.get("tags", [])
            
            if boat_id and boat_name:
                # Ajouter à "Tous les bateaux"
                boats_by_class["Tous les bateaux"].append({
                    "id": boat_id,
                    "name": boat_name
                })
                
                # Ajouter aux classes spécifiques
                for tag_id in boat_tags:
                    if tag_id in classes:
                        class_name = classes[tag_id]
                        boats_by_class[class_name].append({
                            "id": boat_id,
                            "name": boat_name
                        })
    
    # Si aucun bateau n'a été trouvé dans RaceSetup mais que nous avons des données AllPositions,
    # utiliser ces données pour créer une liste de bateaux
    if not boats_by_class["Tous les bateaux"] and all_positions and "boats" in all_positions:
        for boat in all_positions["boats"]:
            if "name" in boat:
                boats_by_class["Tous les bateaux"].append({
                    "id": boat["id"],
                    "name": boat["name"]
                })
                
                # Ajouter à une classe générique si aucune classe n'est définie
                if not classes:
                    if "Classe par défaut" not in boats_by_class:
                        boats_by_class["Classe par défaut"] = []
                    boats_by_class["Classe par défaut"].append({
                        "id": boat["id"],
                        "name": boat["name"]
                    })
    
    # Si toujours aucun bateau, créer des données de démonstration
    if not boats_by_class["Tous les bateaux"]:
        demo_boats = [
            {"id": 1, "name": "MAORI III"},
            {"id": 2, "name": "CORA"},
            {"id": 3, "name": "F35 EXPRESS"}
        ]
        boats_by_class["Tous les bateaux"] = demo_boats
        boats_by_class["Classe par défaut"] = demo_boats
    
    return boats_by_class

# ---------- Fonctions de visualisation -------------

def create_map(boats_data, selected_boats):
    """Crée une carte Folium avec les trajectoires des bateaux sélectionnés"""
    # Trouver les coordonnées moyennes pour centrer la carte
    all_lats = []
    all_lons = []
    
    for boat in boats_data:
        if boat["name"] in selected_boats and "positions" in boat:
            for pos in boat["positions"]:
                all_lats.append(pos["lat"])
                all_lons.append(pos["lon"])
    
    if not all_lats or not all_lons:
        # Coordonnées par défaut si aucune donnée
        center_lat, center_lon = 50.7, -1.2
    else:
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)
    
    # Créer la carte
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
    
    # Ajouter les trajectoires et marqueurs pour chaque bateau
    for boat in boats_data:
        if boat["name"] in selected_boats and "positions" in boat:
            # Générer une couleur unique pour chaque bateau
            color = "#" + format(hash(boat["name"]) % 0xFFFFFF, '06x')
            
            # Calculer vitesse et cap si nécessaire
            if "speed" not in boat["positions"][0]:
                boat["positions"] = calculate_speed_and_bearing(boat["positions"])
            
            # Créer la trajectoire
            points = [[pos["lat"], pos["lon"]] for pos in boat["positions"]]
            if points:
                folium.PolyLine(
                    points,
                    color=color,
                    weight=3,
                    opacity=0.8,
                    tooltip=boat["name"]
                ).add_to(m)
                
                # Ajouter un marqueur pour la dernière position
                last_pos = boat["positions"][-1]
                folium.Marker(
                    [last_pos["lat"], last_pos["lon"]],
                    popup=f"{boat['name']}<br>Vitesse: {last_pos.get('speed', 0):.1f} nœuds<br>Cap: {last_pos.get('bearing', 0):.1f}°",
                    tooltip=boat["name"],
                    icon=folium.Icon(color="blue", icon="ship", prefix="fa")
                ).add_to(m)
    
    return m

def create_speed_chart(boats_data, selected_boats):
    """Crée un graphique matplotlib des vitesses des bateaux sélectionnés"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for boat in boats_data:
        if boat["name"] in selected_boats and "positions" in boat:
            # Calculer vitesse et cap si nécessaire
            if "speed" not in boat["positions"][0]:
                boat["positions"] = calculate_speed_and_bearing(boat["positions"])
            
            # Extraire les timestamps et vitesses
            timestamps = [datetime.fromtimestamp(pos["at"]) for pos in boat["positions"]]
            speeds = [pos.get("speed", 0) for pos in boat["positions"]]
            
            # Tracer la courbe
            ax.plot(timestamps, speeds, label=boat["name"])
    
    ax.set_xlabel("Date/Heure")
    ax.set_ylabel("Vitesse (nœuds)")
    ax.set_title("Comparaison des vitesses")
    ax.grid(True)
    ax.legend()
    
    # Rotation des étiquettes de l'axe x pour une meilleure lisibilité
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def create_bearing_chart(boats_data, selected_boats):
    """Crée un graphique matplotlib des caps des bateaux sélectionnés"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for boat in boats_data:
        if boat["name"] in selected_boats and "positions" in boat:
            # Calculer vitesse et cap si nécessaire
            if "bearing" not in boat["positions"][0]:
                boat["positions"] = calculate_speed_and_bearing(boat["positions"])
            
            # Extraire les timestamps et caps
            timestamps = [datetime.fromtimestamp(pos["at"]) for pos in boat["positions"]]
            bearings = [pos.get("bearing", 0) for pos in boat["positions"]]
            
            # Tracer la courbe
            ax.plot(timestamps, bearings, label=boat["name"])
    
    ax.set_xlabel("Date/Heure")
    ax.set_ylabel("Cap (degrés)")
    ax.set_title("Comparaison des caps")
    ax.grid(True)
    ax.legend()
    
    # Rotation des étiquettes de l'axe x pour une meilleure lisibilité
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def create_comparison_table(boats_data, selected_boats):
    """Crée un tableau pandas avec les statistiques des bateaux sélectionnés"""
    stats = []
    
    for boat in boats_data:
        if boat["name"] in selected_boats and "positions" in boat:
            # Calculer vitesse et cap si nécessaire
            if "speed" not in boat["positions"][0]:
                boat["positions"] = calculate_speed_and_bearing(boat["positions"])
            
            # Calculer les statistiques
            speeds = [pos.get("speed", 0) for pos in boat["positions"] if "speed" in pos]
            
            if speeds:
                avg_speed = sum(speeds) / len(speeds)
                max_speed = max(speeds)
            else:
                avg_speed = 0
                max_speed = 0
            
            # Calculer la distance totale parcourue
            total_distance = 0
            for i in range(1, len(boat["positions"])):
                prev_pos = boat["positions"][i-1]
                curr_pos = boat["positions"][i]
                distance = calculate_distance(
                    prev_pos["lat"], prev_pos["lon"],
                    curr_pos["lat"], curr_pos["lon"]
                )
                total_distance += distance
            
            # Ajouter les statistiques au tableau
            stats.append({
                "Bateau": boat["name"],
                "Vitesse moyenne (nœuds)": round(avg_speed, 2),
                "Vitesse max (nœuds)": round(max_speed, 2),
                "Distance parcourue (NM)": round(total_distance, 2)
            })
    
    # Créer le DataFrame
    if stats:
        return pd.DataFrame(stats)
    else:
        return pd.DataFrame()

# ---------- UI ---------------

def main():
    st.title("YB Tracking Analyzer")
    st.write("Analysez les données de courses YB Tracking")
    
    # Sidebar pour les entrées utilisateur
    with st.sidebar:
        st.header("Configuration")
        
        # Entrée de l'ID de course
        race_id = st.text_input("Identifiant de la course", "dgbr2025")
        
        # Bouton pour télécharger les données
        if st.button("Télécharger les données"):
            if race_id.strip():
                with st.spinner("Téléchargement des données..."):
                    st.session_state.data = fetch(race_id.strip())
                    
                    # Extraire les classes de bateaux
                    if st.session_state.data.get("RaceSetup"):
                        st.session_state.classes = extract_boat_classes(st.session_state.data["RaceSetup"])
                        st.session_state.boats_by_class = extract_boats_by_class(
                            st.session_state.data["RaceSetup"],
                            st.session_state.classes,
                            st.session_state.data.get("AllPositions")
                        )
                        
                        # Préparer les données des bateaux
                        if st.session_state.data.get("AllPositions") and "boats" in st.session_state.data["AllPositions"]:
                            st.session_state.boats_data = st.session_state.data["AllPositions"]["boats"]
                            st.success(f"Données téléchargées et traitées pour {len(st.session_state.boats_data)} bateaux")
                        else:
                            st.error("Aucune donnée de position disponible")
                    else:
                        st.error("Impossible de récupérer les informations de la course")
            else:
                st.error("Veuillez saisir un identifiant de course")
        
        # Bouton pour réinitialiser les données
        if st.button("Réinitialiser"):
            # Effacer les données de la session
            for key in ["data", "classes", "boats_by_class", "boats_data", "selected_class", "selected_boats"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("Données réinitialisées")
        
        # Sélection de la classe de bateaux
        if "boats_by_class" in st.session_state:
            st.header("Filtrage")
            
            # Liste des classes disponibles
            class_options = list(st.session_state.boats_by_class.keys())
            
            # Sélection de la classe
            selected_class = st.selectbox(
                "Classe de bateaux",
                class_options,
                index=0 if class_options else None
            )
            
            if selected_class:
                st.session_state.selected_class = selected_class
                
                # Liste des bateaux dans la classe sélectionnée
                boat_options = [boat["name"] for boat in st.session_state.boats_by_class[selected_class]]
                
                # Sélection des bateaux
                selected_boats = st.multiselect(
                    "Bateaux à analyser",
                    boat_options,
                    default=boat_options[:1] if boat_options else None
                )
                
                if selected_boats:
                    st.session_state.selected_boats = selected_boats
                    
                    # Boutons pour sélectionner/désélectionner tous les bateaux
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Tous les bateaux"):
                            st.session_state.selected_boats = boat_options
                            st.rerun()
                    with col2:
                        if st.button("Aucun bateau"):
                            st.session_state.selected_boats = []
                            st.rerun()
            
            # Type d'analyse
            st.header("Analyse")
            analysis_type = st.selectbox(
                "Type d'analyse",
                ["Carte", "Vitesse", "Cap", "Tableau comparatif"]
            )
            
            st.session_state.analysis_type = analysis_type
    
    # Contenu principal
    if "boats_data" in st.session_state and "selected_boats" in st.session_state and st.session_state.selected_boats:
        # Filtrer les bateaux sélectionnés
        filtered_boats = [
            boat for boat in st.session_state.boats_data
            if boat["name"] in st.session_state.selected_boats
        ]
        
        # Afficher l'analyse sélectionnée
        if st.session_state.analysis_type == "Carte":
            st.header("Carte des trajectoires")
            
            # Créer la carte
            m = create_map(filtered_boats, st.session_state.selected_boats)
            
            # Afficher la carte
            folium_static(m, width=800, height=600)
            
            # Exporter les données
            if st.button("Exporter les données de position (CSV)"):
                csv_data = []
                
                for boat in filtered_boats:
                    for pos in boat["positions"]:
                        csv_data.append({
                            "Bateau": boat["name"],
                            "Timestamp": datetime.fromtimestamp(pos["at"]),
                            "Latitude": pos["lat"],
                            "Longitude": pos["lon"],
                            "Vitesse (nœuds)": pos.get("speed", ""),
                            "Cap (degrés)": pos.get("bearing", "")
                        })
                
                if csv_data:
                    df = pd.DataFrame(csv_data)
                    csv = df.to_csv(index=False)
                    
                    st.download_button(
                        label="Télécharger CSV",
                        data=csv,
                        file_name="positions.csv",
                        mime="text/csv"
                    )
        
        elif st.session_state.analysis_type == "Vitesse":
            st.header("Analyse des vitesses")
            
            fig = create_speed_chart(filtered_boats, st.session_state.selected_boats)
            st.pyplot(fig)
            
            # Exporter les données
            if st.button("Exporter les données de vitesse (CSV)"):
                csv_data = []
                
                for boat in filtered_boats:
                    for pos in boat["positions"]:
                        if "speed" in pos:
                            csv_data.append({
                                "Bateau": boat["name"],
                                "Timestamp": datetime.fromtimestamp(pos["at"]),
                                "Latitude": pos["lat"],
                                "Longitude": pos["lon"],
                                "Vitesse (nœuds)": pos["speed"]
                            })
                
                if csv_data:
                    df = pd.DataFrame(csv_data)
                    csv = df.to_csv(index=False)
                    
                    st.download_button(
                        label="Télécharger CSV",
                        data=csv,
                        file_name="vitesses.csv",
                        mime="text/csv"
                    )
        
        elif st.session_state.analysis_type == "Cap":
            st.header("Analyse des caps")
            
            fig = create_bearing_chart(filtered_boats, st.session_state.selected_boats)
            st.pyplot(fig)
            
            # Exporter les données
            if st.button("Exporter les données de cap (CSV)"):
                csv_data = []
                
                for boat in filtered_boats:
                    for pos in boat["positions"]:
                        if "bearing" in pos:
                            csv_data.append({
                                "Bateau": boat["name"],
                                "Timestamp": datetime.fromtimestamp(pos["at"]),
                                "Latitude": pos["lat"],
                                "Longitude": pos["lon"],
                                "Cap (degrés)": pos["bearing"]
                            })
                
                if csv_data:
                    df = pd.DataFrame(csv_data)
                    csv = df.to_csv(index=False)
                    
                    st.download_button(
                        label="Télécharger CSV",
                        data=csv,
                        file_name="caps.csv",
                        mime="text/csv"
                    )
        
        elif st.session_state.analysis_type == "Tableau comparatif":
            st.header("Tableau comparatif")
            
            # Créer le tableau
            df = create_comparison_table(filtered_boats, st.session_state.selected_boats)
            
            if not df.empty:
                # Afficher le tableau
                st.dataframe(df)
                
                # Exporter les données
                if st.button("Exporter le tableau (CSV)"):
                    csv = df.to_csv(index=False)
                    
                    st.download_button(
                        label="Télécharger CSV",
                        data=csv,
                        file_name="comparaison.csv",
                        mime="text/csv"
                    )
            else:
                st.info("Aucune donnée disponible pour les bateaux sélectionnés")
    
    elif "data" not in st.session_state:
        st.info("Entrez l'identifiant de la course et cliquez sur 'Télécharger les données' pour commencer")
    else:
        st.warning("Sélectionnez une classe et des bateaux dans le menu latéral")

if __name__ == "__main__":
    main()
