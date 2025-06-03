import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import json
import matplotlib.pyplot as plt
from datetime import datetime
import math

st.set_page_config(
    page_title="YB Tracking Analyzer",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Fonction pour calculer la distance entre deux points GPS (formule de Haversine)
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Rayon de la Terre en km
    
    # Conversion en radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Différences de longitude et de latitude
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    # Formule de Haversine
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Distance en km, convertie en milles nautiques
    distance_km = R * c
    distance_nm = distance_km / 1.852
    
    return distance_nm

# Fonction pour calculer le cap entre deux points GPS
def calculate_bearing(lat1, lon1, lat2, lon2):
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

# Fonction pour télécharger les données YB Tracking
def fetch_yb_data(race_id, max_retries=3):
    """
    Télécharge les données YB Tracking pour une course spécifique
    """
    st.write(f"Téléchargement des données pour la course {race_id}...")
    
    # URLs des différents fichiers à télécharger
    urls = {
        "RaceSetup": f"https://cf.yb.tl/JSON/{race_id}/RaceSetup",
        "LatestPositions": f"https://cf.yb.tl/JSON/{race_id}/LatestPositions",
        "Leaderboard": f"https://cf.yb.tl/JSON/{race_id}/leaderboard"
    }
    
    data = {}
    
    # Téléchargement des fichiers JSON
    for name, url in urls.items():
        progress_text = f"Téléchargement de {name}..."
        progress_bar = st.progress(0, text=progress_text)
        
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                data[name] = response.json()
                progress_bar.progress(1.0, text=f"{name} téléchargé avec succès")
                break
            except Exception as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    st.warning(f"Tentative {attempt} échouée pour {name}: {str(e)}. Nouvelle tentative dans {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    st.error(f"Échec du téléchargement de {name} après {max_retries} tentatives: {str(e)}")
                    data[name] = None
    
    # Simulation des données de positions pour la démo
    # Dans une application réelle, vous devriez implémenter le téléchargement et le décodage de AllPositions3
    st.info("Note: Cette démo utilise des données simulées pour l'historique des positions. Dans une application réelle, implémentez le décodage complet du fichier binaire AllPositions3.")
    
    data["AllPositions"] = {
        "boats": generate_simulated_positions(data.get("RaceSetup"), data.get("LatestPositions"))
    }
    
    return data

# Fonction pour générer des positions simulées
def generate_simulated_positions(race_setup, latest_positions):
    """
    Génère des positions simulées pour la démo
    """
    simulated_boats = []
    
    if not race_setup or not latest_positions:
        return simulated_boats
    
    # Utiliser les bateaux de RaceSetup et leurs dernières positions connues
    if "boats" in race_setup and "boats" in latest_positions:
        for boat_setup in race_setup["boats"]:
            boat_id = boat_setup.get("id")
            boat_name = boat_setup.get("name")
            
            if boat_id and boat_name:
                # Trouver la dernière position connue
                last_pos = None
                for boat_pos in latest_positions["boats"]:
                    if boat_pos.get("id") == boat_id:
                        last_pos = boat_pos
                        break
                
                if last_pos and "lat" in last_pos and "lon" in last_pos:
                    # Générer des positions simulées autour de la dernière position connue
                    positions = []
                    base_lat = last_pos["lat"]
                    base_lon = last_pos["lon"]
                    base_time = int(time.time() * 1000) - 24 * 60 * 60 * 1000  # 24h avant
                    
                    for i in range(10):  # 10 positions par bateau
                        time_offset = i * 3 * 60 * 60 * 1000  # 3h entre chaque position
                        lat_offset = (np.random.random() - 0.5) * 0.1
                        lon_offset = (np.random.random() - 0.5) * 0.1
                        
                        positions.append({
                            "timestamp": base_time + time_offset,
                            "lat": base_lat + lat_offset * i/10,
                            "lon": base_lon + lon_offset * i/10
                        })
                    
                    # Calculer les vitesses et caps
                    positions = calculate_speed_and_bearing(positions)
                    
                    simulated_boats.append({
                        "id": boat_id,
                        "name": boat_name,
                        "positions": positions
                    })
    
    return simulated_boats

# Fonction pour extraire les classes de bateaux
def extract_boat_classes(race_setup):
    """
    Extrait les classes de bateaux à partir des données RaceSetup
    """
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

# Fonction pour extraire les bateaux par classe
def extract_boats_by_class(race_setup, classes):
    """
    Regroupe les bateaux par classe
    """
    if not race_setup or not classes:
        return {}
    
    boats_by_class = {class_name: [] for class_id, class_name in classes.items()}
    
    # Ajout d'une classe "Tous les bateaux"
    boats_by_class["Tous les bateaux"] = []
    
    # Extraction des bateaux et de leurs classes
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
    
    return boats_by_class

# Fonction pour calculer les vitesses et caps à partir des positions
def calculate_speed_and_bearing(positions):
    """
    Calcule la vitesse et le cap pour chaque position
    """
    if len(positions) < 2:
        return positions
    
    for i in range(1, len(positions)):
        prev_pos = positions[i-1]
        curr_pos = positions[i]
        
        # Calcul du temps écoulé en heures
        time_diff = (curr_pos["timestamp"] - prev_pos["timestamp"]) / (1000 * 60 * 60)
        
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

# Fonction pour créer un graphique de vitesse
def create_speed_chart(boats_data, selected_boats):
    """
    Crée un graphique matplotlib des vitesses des bateaux sélectionnés
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for boat in boats_data:
        if boat["name"] in selected_boats and "positions" in boat:
            # Extraire les timestamps et vitesses
            timestamps = [datetime.fromtimestamp(pos["timestamp"]/1000) for pos in boat["positions"]]
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

# Fonction pour créer un graphique de cap
def create_bearing_chart(boats_data, selected_boats):
    """
    Crée un graphique matplotlib des caps des bateaux sélectionnés
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for boat in boats_data:
        if boat["name"] in selected_boats and "positions" in boat:
            # Extraire les timestamps et caps
            timestamps = [datetime.fromtimestamp(pos["timestamp"]/1000) for pos in boat["positions"]]
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

# Fonction pour créer un tableau comparatif
def create_comparison_table(boats_data, selected_boats):
    """
    Crée un tableau pandas avec les statistiques des bateaux sélectionnés
    """
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
        return pd.DataFrame(columns=["Bateau", "Vitesse moyenne (nœuds)", "Vitesse max (nœuds)", "Distance parcourue (NM)"])

# Interface utilisateur Streamlit
def main():
    st.title("🚢 YB Tracking Analyzer")
    
    st.markdown("""
    Cette application vous permet d'extraire et d'analyser les données de courses depuis YB Tracking.
    
    ### Instructions :
    1. Entrez l'identifiant de la course (ex: dgbr2025)
    2. Cliquez sur "Extraire les données"
    3. Sélectionnez une classe de bateaux et les bateaux spécifiques à analyser
    4. Explorez les analyses et visualisations
    """)
    
    # Formulaire pour l'extraction des données
    with st.form("extraction_form"):
        race_id = st.text_input("Identifiant de la course", "dgbr2025")
        submit_button = st.form_submit_button("Extraire les données")
    
    # Si le bouton est cliqué, extraire les données
    if submit_button:
        # Stocker l'ID de course dans la session
        st.session_state.race_id = race_id
        
        # Extraire les données
        data = fetch_yb_data(race_id)
        
        # Stocker les données dans la session
        st.session_state.data = data
        
        # Extraire les classes de bateaux
        if "RaceSetup" in data and data["RaceSetup"]:
            classes = extract_boat_classes(data["RaceSetup"])
            boats_by_class = extract_boats_by_class(data["RaceSetup"], classes)
            
            st.session_state.classes = classes
            st.session_state.boats_by_class = boats_by_class
    
    # Si des données ont été extraites, afficher les analyses
    if "data" in st.session_state and st.session_state.data:
        st.success(f"Données extraites avec succès pour la course {st.session_state.race_id}")
        
        # Sélection de la classe de bateaux
        if "boats_by_class" in st.session_state:
            class_options = list(st.session_state.boats_by_class.keys())
            selected_class = st.selectbox("Sélectionnez une classe de bateaux", class_options)
            
            # Sélection des bateaux spécifiques
            if selected_class in st.session_state.boats_by_class:
                boat_options = [boat["name"] for boat in st.session_state.boats_by_class[selected_class]]
                
                # Option pour sélectionner/désélectionner tous les bateaux
                select_all = st.checkbox("Sélectionner tous les bateaux", True)
                
                if select_all:
                    default_boats = boat_options
                else:
                    default_boats = []
                
                selected_boats = st.multiselect(
                    "Sélectionnez les bateaux à analyser",
                    boat_options,
                    default=default_boats
                )
                
                # Si des bateaux sont sélectionnés, afficher les analyses
                if selected_boats:
                    # Préparer les données des bateaux sélectionnés
                    boats_data = []
                    
                    if "AllPositions" in st.session_state.data and st.session_state.data["AllPositions"]:
                        all_positions = st.session_state.data["AllPositions"]
                        
                        for boat_info in st.session_state.boats_by_class[selected_class]:
                            if boat_info["name"] in selected_boats:
                                # Trouver les positions de ce bateau
                                for boat in all_positions.get("boats", []):
                                    if boat.get("id") == boat_info["id"] or boat.get("name") == boat_info["name"]:
                                        boats_data.append({
                                            "id": boat_info["id"],
                                            "name": boat_info["name"],
                                            "positions": boat.get("positions", [])
                                        })
                                        break
                    
                    # Afficher les analyses en onglets
                    tab1, tab2, tab3 = st.tabs(["Vitesses", "Caps", "Comparaison"])
                    
                    with tab1:
                        st.subheader("Graphique des vitesses")
                        if boats_data:
                            speed_chart = create_speed_chart(boats_data, selected_boats)
                            st.pyplot(speed_chart)
                        else:
                            st.warning("Pas de données de vitesse disponibles pour les bateaux sélectionnés")
                    
                    with tab2:
                        st.subheader("Graphique des caps")
                        if boats_data:
                            bearing_chart = create_bearing_chart(boats_data, selected_boats)
                            st.pyplot(bearing_chart)
                        else:
                            st.warning("Pas de données de cap disponibles pour les bateaux sélectionnés")
                    
                    with tab3:
                        st.subheader("Tableau comparatif")
                        if boats_data:
                            comparison_table = create_comparison_table(boats_data, selected_boats)
                            st.dataframe(comparison_table)
                            
                            # Option pour télécharger le tableau en CSV
                            csv = comparison_table.to_csv(index=False)
                            st.download_button(
                                label="Télécharger le tableau en CSV",
                                data=csv,
                                file_name=f"comparaison_{st.session_state.race_id}_{selected_class}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("Pas de données disponibles pour les bateaux sélectionnés")
                else:
                    st.warning("Veuillez sélectionner au moins un bateau pour afficher les analyses")
        else:
            st.error("Erreur lors de l'extraction des classes de bateaux")

if __name__ == "__main__":
    main()
