import streamlit as st
import requests
import time
import json
from datetime import datetime
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from folium.plugins import MarkerCluster
import folium
from streamlit_folium import st_folium

# ----------------- CONFIG -----------------

st.set_page_config(
    page_title="YB Tracking Analyzer",
    page_icon="⛵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Coordonnées de la zone de course (sud de l'Angleterre / Manche)
RACE_AREA = {
    "lat_min": 49.0,  # Limite sud
    "lat_max": 52.0,  # Limite nord
    "lon_min": -6.0,  # Limite ouest
    "lon_max": 2.0    # Limite est
}

# --------------- HTTP utils --------------

def safe_get_json(url: str, retries: int = 3, show_error: bool = True):
    """Récupère des données JSON avec gestion des erreurs et retries"""
    backoff = 2
    for i in range(retries):
        try:
            with st.spinner(f"Téléchargement de {url.split('/')[-1]}..."):
                r = requests.get(url, timeout=30)
                if r.status_code == 500:
                    if show_error:
                        st.error(f"Erreur 500 pour {url}")
                    if i == retries - 1:
                        return None
                    time.sleep(backoff ** i)
                    continue
                if r.status_code == 400:
                    if show_error:
                        st.error(f"Erreur 400 pour {url}")
                    if i == retries - 1:
                        return None
                    time.sleep(backoff ** i)
                    continue
                r.raise_for_status()
                return r.json()
        except Exception as e:
            if i == retries - 1 and show_error:
                st.error(f"Échec du téléchargement après {retries} tentatives: {str(e)}")
                return None
            time.sleep(backoff ** i)
            if show_error:
                st.warning(f"Tentative {i+1} échouée pour {url.split('/')[-1]}: {str(e)}. Nouvelle tentative dans {backoff ** i}s...")
    return None

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

def is_valid_position(lat, lon):
    """Vérifie si une position est dans la zone de course valide"""
    return (RACE_AREA["lat_min"] <= lat <= RACE_AREA["lat_max"] and 
            RACE_AREA["lon_min"] <= lon <= RACE_AREA["lon_max"])

def process_positions(positions_data):
    """
    Traite les données de position pour calculer vitesse, cap et filtrer les positions invalides
    
    Args:
        positions_data: Liste de dictionnaires contenant les positions
        
    Returns:
        Liste de dictionnaires avec positions valides et données calculées
    """
    if not positions_data:
        return []
    
    # Filtrer les positions valides
    valid_positions = []
    for pos in positions_data:
        # Vérifier que les coordonnées existent
        if "lat" not in pos or "lon" not in pos:
            continue
            
        # Vérifier que les coordonnées sont dans la zone de course
        if not is_valid_position(pos["lat"], pos["lon"]):
            continue
            
        # Ajouter la position valide
        valid_positions.append(pos)
    
    # Si aucune position valide, retourner une liste vide
    if not valid_positions:
        return []
    
    # Trier les positions par timestamp
    valid_positions.sort(key=lambda x: x.get("at", 0))
    
    # Calculer vitesse et cap pour chaque position
    for i in range(1, len(valid_positions)):
        prev_pos = valid_positions[i-1]
        curr_pos = valid_positions[i]
        
        # Calcul du temps écoulé en heures
        time_diff = (curr_pos["at"] - prev_pos["at"]) / (60 * 60)
        
        # Calcul de la distance en milles nautiques
        distance = calculate_distance(
            prev_pos["lat"], prev_pos["lon"],
            curr_pos["lat"], curr_pos["lon"]
        )
        
        # Calcul de la vitesse en nœuds (avec limite raisonnable)
        if time_diff > 0:
            speed = distance / time_diff
            # Limiter la vitesse à une valeur raisonnable pour un voilier (max 30 nœuds)
            curr_pos["speed"] = min(speed, 30.0)
        else:
            curr_pos["speed"] = 0
        
        # Calcul du cap
        curr_pos["bearing"] = calculate_bearing(
            prev_pos["lat"], prev_pos["lon"],
            curr_pos["lat"], curr_pos["lon"]
        )
    
    # Pour la première position, utiliser les mêmes valeurs que la deuxième
    if len(valid_positions) > 1:
        valid_positions[0]["speed"] = valid_positions[1]["speed"]
        valid_positions[0]["bearing"] = valid_positions[1]["bearing"]
    else:
        valid_positions[0]["speed"] = 0
        valid_positions[0]["bearing"] = 0
    
    return valid_positions

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

def extract_boats_by_class(race_setup, classes, latest_positions=None):
    """
    Regroupe les bateaux par classe
    
    Args:
        race_setup: Données RaceSetup
        classes: Dictionnaire des classes {id: name}
        latest_positions: Données LatestPositions (optionnel)
    """
    # Créer un dictionnaire pour stocker les bateaux par classe
    boats_by_class = {}
    
    # Ajouter une classe "Tous les bateaux"
    boats_by_class["Tous les bateaux"] = []
    
    # Ajouter les classes définies
    if classes:
        for class_name in classes.values():
            boats_by_class[class_name] = []
    
    # Créer un mapping des bateaux depuis RaceSetup
    boat_mapping = {}
    if race_setup and "boats" in race_setup:
        for boat in race_setup["boats"]:
            boat_id = boat.get("id")
            boat_name = boat.get("name")
            if boat_id and boat_name:
                boat_mapping[boat_id] = {
                    "name": boat_name,
                    "tags": boat.get("tags", [])
                }
    
    # Extraire les bateaux depuis LatestPositions (source principale)
    if latest_positions and "positions" in latest_positions:
        for pos in latest_positions["positions"]:
            if "boatID" in pos:
                boat_id = pos["boatID"]
                
                # Récupérer le nom du bateau depuis le mapping ou utiliser un nom générique
                if boat_id in boat_mapping:
                    boat_name = boat_mapping[boat_id]["name"]
                    boat_tags = boat_mapping[boat_id]["tags"]
                else:
                    boat_name = f"Bateau {boat_id}"
                    boat_tags = []
                
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
                
                # Si aucune classe n'est définie pour ce bateau, l'ajouter à une classe par défaut
                if not boat_tags and not classes:
                    if "Classe par défaut" not in boats_by_class:
                        boats_by_class["Classe par défaut"] = []
                    boats_by_class["Classe par défaut"].append({
                        "id": boat_id,
                        "name": boat_name
                    })
    
    # Si aucun bateau n'a été trouvé, utiliser les données de RaceSetup
    if not boats_by_class["Tous les bateaux"] and race_setup and "boats" in race_setup:
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
    
    return boats_by_class

def fetch_boat_positions(race_id, boat_id):
    """
    Récupère les positions d'un bateau spécifique
    
    Args:
        race_id: Identifiant de la course
        boat_id: Identifiant du bateau
        
    Returns:
        Liste de positions valides avec vitesse et cap calculés
    """
    # Liste des formats d'URL à essayer
    url_formats = [
        f"https://cf.yb.tl/JSON/{race_id}/boat/{boat_id}",
        f"https://cf.yb.tl/JSON/{race_id}/BoatPositions/{boat_id}",
        f"https://cf.yb.tl/JSON/{race_id}/BoatInfo/{boat_id}",
        f"https://yb.tl/JSON/{race_id}/boat/{boat_id}",
        f"https://yb.tl/JSON/{race_id}/BoatPositions/{boat_id}",
        f"https://yb.tl/JSON/{race_id}/BoatInfo/{boat_id}"
    ]
    
    # Essayer chaque format d'URL
    for url in url_formats:
        boat_data = safe_get_json(url, show_error=False)
        if boat_data and "positions" in boat_data and boat_data["positions"]:
            # Traiter les positions pour calculer vitesse, cap et filtrer les positions invalides
            return process_positions(boat_data["positions"])
    
    # Si aucune URL n'a fonctionné, afficher une erreur
    st.error(f"Impossible de récupérer les positions du bateau {boat_id}")
    return []

# ------- fetch ---------------

@st.cache_data(ttl=300)
def fetch_race_data(race_id: str):
    """Récupère les données de base pour une course donnée"""
    # Liste des formats d'URL à essayer
    base_formats = [
        f"https://cf.yb.tl/JSON/{race_id}",
        f"https://yb.tl/JSON/{race_id}"
    ]
    
    # Téléchargement des fichiers JSON
    data = {}
    
    # Essayer chaque format d'URL de base
    for base in base_formats:
        # RaceSetup
        data["RaceSetup"] = safe_get_json(f"{base}/RaceSetup", show_error=False)
        if data["RaceSetup"]:
            st.success(f"RaceSetup téléchargé avec succès")
            
            # Leaderboard
            data["leaderboard"] = safe_get_json(f"{base}/leaderboard", show_error=False)
            if data["leaderboard"]:
                st.success(f"Leaderboard téléchargé avec succès")
            
            # LatestPositions
            for i in range(3):  # Essayer 3 fois
                data["LatestPositions"] = safe_get_json(f"{base}/LatestPositions", show_error=False)
                if data["LatestPositions"]:
                    st.success(f"LatestPositions téléchargé avec succès")
                    break
                else:
                    st.warning(f"Tentative {i+1} échouée pour LatestPositions. Nouvelle tentative dans {2**(i+1)}s...")
                    time.sleep(2**(i+1))
            
            if not data["LatestPositions"]:
                st.error(f"Échec du téléchargement de LatestPositions après 3 tentatives")
            
            # Si on a au moins RaceSetup, on peut continuer
            return data
    
    # Si aucun format d'URL n'a fonctionné pour RaceSetup
    st.error(f"Impossible de récupérer les informations de la course {race_id}")
    return {}

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
        # Coordonnées par défaut si aucune donnée (sud de l'Angleterre)
        center_lat, center_lon = 50.7, -1.2
    else:
        center_lat = sum(all_lats) / len(all_lats)
        center_lon = sum(all_lons) / len(all_lons)
    
    # Créer la carte
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
    
    # Ajouter les trajectoires et marqueurs pour chaque bateau
    for boat in boats_data:
        if boat["name"] in selected_boats and "positions" in boat:
            # Générer une couleur unique pour chaque bateau
            color = "#" + format(hash(boat["name"]) % 0xFFFFFF, '06x')
            
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
                with st.spinner("Téléchargement des données de course..."):
                    # Récupérer les données de base de la course
                    st.session_state.race_data = fetch_race_data(race_id.strip())
                    
                    # Extraire les classes de bateaux
                    if st.session_state.race_data.get("RaceSetup"):
                        st.session_state.classes = extract_boat_classes(st.session_state.race_data["RaceSetup"])
                        st.session_state.boats_by_class = extract_boats_by_class(
                            st.session_state.race_data["RaceSetup"],
                            st.session_state.classes,
                            st.session_state.race_data.get("LatestPositions")
                        )
                        
                        # Initialiser la liste des bateaux avec données
                        st.session_state.boats_data = []
                        
                        # Afficher un message de succès
                        st.success("Données extraites avec succès pour la course " + race_id.strip())
                    else:
                        st.error("Impossible de récupérer les informations de la course")
            else:
                st.error("Veuillez saisir un identifiant de course")
        
        # Bouton pour réinitialiser les données
        if st.button("Réinitialiser"):
            # Effacer les données de la session
            for key in ["race_data", "classes", "boats_by_class", "boats_data", "selected_class", "selected_boats"]:
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
    if "boats_by_class" not in st.session_state:
        st.info("Entrez l'identifiant de la course et cliquez sur 'Télécharger les données' pour commencer")
    elif "selected_boats" not in st.session_state or not st.session_state.selected_boats:
        st.warning("Sélectionnez une classe et des bateaux dans le menu latéral")
    else:
        # Récupérer les données de position pour chaque bateau sélectionné
        if "race_data" in st.session_state and "boats_data" in st.session_state:
            race_id = st.session_state.race_data.get("RaceSetup", {}).get("id", "")
            
            # Récupérer les IDs des bateaux sélectionnés
            selected_boat_ids = []
            for boat_class in st.session_state.boats_by_class.values():
                for boat in boat_class:
                    if boat["name"] in st.session_state.selected_boats and boat["id"] not in [b["id"] for b in selected_boat_ids]:
                        selected_boat_ids.append(boat)
            
            # Récupérer les positions pour chaque bateau sélectionné
            boats_data = []
            
            with st.spinner("Récupération des données de position..."):
                for boat in selected_boat_ids:
                    boat_id = boat["id"]
                    boat_name = boat["name"]
                    
                    # Vérifier si les données de ce bateau sont déjà chargées
                    boat_loaded = False
                    for loaded_boat in st.session_state.boats_data:
                        if loaded_boat["id"] == boat_id:
                            boats_data.append(loaded_boat)
                            boat_loaded = True
                            break
                    
                    # Si les données ne sont pas déjà chargées, les récupérer
                    if not boat_loaded:
                        positions = fetch_boat_positions(race_id, boat_id)
                        
                        if positions:
                            # Ajouter le bateau aux données
                            boat_data = {
                                "id": boat_id,
                                "name": boat_name,
                                "positions": positions
                            }
                            boats_data.append(boat_data)
                            
                            # Ajouter aux données de session pour éviter de recharger
                            st.session_state.boats_data.append(boat_data)
            
            # Si aucune donnée n'a été récupérée, afficher un message d'erreur
            if not boats_data:
                st.error("Aucune donnée de position valide n'a pu être récupérée pour les bateaux sélectionnés.")
                st.stop()
            
            # Afficher l'analyse sélectionnée
            if st.session_state.analysis_type == "Carte":
                st.header("Carte des trajectoires")
                
                # Créer la carte
                m = create_map(boats_data, st.session_state.selected_boats)
                
                # Afficher la carte
                st_folium(m, width=800, height=600)
                
                # Exporter les données
                if st.button("Exporter les données de position (CSV)"):
                    csv_data = []
                    
                    for boat in boats_data:
                        if boat["name"] in st.session_state.selected_boats:
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
                
                fig = create_speed_chart(boats_data, st.session_state.selected_boats)
                st.pyplot(fig)
                
                # Exporter les données
                if st.button("Exporter les données de vitesse (CSV)"):
                    csv_data = []
                    
                    for boat in boats_data:
                        if boat["name"] in st.session_state.selected_boats:
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
                
                fig = create_bearing_chart(boats_data, st.session_state.selected_boats)
                st.pyplot(fig)
                
                # Exporter les données
                if st.button("Exporter les données de cap (CSV)"):
                    csv_data = []
                    
                    for boat in boats_data:
                        if boat["name"] in st.session_state.selected_boats:
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
                df = create_comparison_table(boats_data, st.session_state.selected_boats)
                
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
        else:
            st.warning("Veuillez d'abord télécharger les données de course")

if __name__ == "__main__":
    main()
