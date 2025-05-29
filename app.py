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
def fetch_yb_data(race_id, max_retries=3, chunk_size=1024*1024):
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
    
    # Téléchargement du fichier binaire AllPositions3
    all_positions_url = f"https://cf.yb.tl/BIN/{race_id}/AllPositions3"
    progress_text = "Téléchargement de l'historique des positions..."
    progress_bar = st.progress(0, text=progress_text)
    
    try:
        # Téléchargement par morceaux pour éviter les timeouts
        response = requests.get(all_positions_url, stream=True, timeout=60)
        response.raise_for_status()
        
        # Obtenir la taille totale si disponible
        total_size = int(response.headers.get('content-length', 0))
        
        # Lecture par morceaux
        chunks = []
        downloaded = 0
        
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                chunks.append(chunk)
                downloaded += len(chunk)
                if total_size:
                    progress = min(downloaded / total_size, 1.0)
                    progress_bar.progress(progress, text=f"Téléchargement: {downloaded/1024/1024:.1f} Mo")
        
        # Vérification que les données sont complètes
        binary_data = b''.join(chunks)
        if len(binary_data) < 1000:  # Taille minimale attendue
            st.error("Données incomplètes reçues pour l'historique des positions")
            data["AllPositions"] = None
        else:
            # Décodage des données binaires
            data["AllPositions"] = decode_all_positions(binary_data)
            progress_bar.progress(1.0, text="Historique des positions téléchargé et décodé avec succès")
    
    except Exception as e:
        st.error(f"Erreur lors du téléchargement de l'historique des positions: {str(e)}")
        data["AllPositions"] = None
    
    return data

# Fonction pour décoder le fichier binaire AllPositions3
def decode_all_positions(binary_data):
    """
    Version simplifiée du décodage des positions
    Cette fonction est une approximation - dans une application réelle,
    utilisez l'outil decyb ou une implémentation complète du décodage
    """
    # Simuler le décodage pour la démo
    # Dans une application réelle, implémentez le décodage complet
    st.info("Note: Cette démo utilise des données simulées pour l'historique des positions. Dans une application réelle, implémentez le décodage complet du fichier binaire.")
    
    # Création de données simulées basées sur LatestPositions
    # Dans une application réelle, décodez correctement le fichier binaire
    return {
        "boats": [
            {
                "id": "boat1",
                "name": "MAORI III",
                "positions": [
                    {"timestamp": 1621234567000, "lat": 50.7, "lon": -1.2},
                    {"timestamp": 1621238167000, "lat": 50.72, "lon": -1.25},
                    {"timestamp": 1621241767000, "lat": 50.74, "lon": -1.3},
                    {"timestamp": 1621245367000, "lat": 50.76, "lon": -1.35},
                    {"timestamp": 1621248967000, "lat": 50.78, "lon": -1.4},
                ]
            },
            {
                "id": "boat2",
                "name": "CORA",
                "positions": [
                    {"timestamp": 1621234567000, "lat": 50.71, "lon": -1.21},
                    {"timestamp": 1621238167000, "lat": 50.73, "lon": -1.26},
                    {"timestamp": 1621241767000, "lat": 50.75, "lon": -1.31},
                    {"timestamp": 1621245367000, "lat": 50.77, "lon": -1.36},
                    {"timestamp": 1621248967000, "lat": 50.79, "lon": -1.41},
                ]
            },
            {
                "id": "boat3",
                "name": "F35 EXPRESS",
                "positions": [
                    {"timestamp": 1621234567000, "lat": 50.69, "lon": -1.19},
                    {"timestamp": 1621238167000, "lat": 50.71, "lon": -1.24},
                    {"timestamp": 1621241767000, "lat": 50.73, "lon": -1.29},
                    {"timestamp": 1621245367000, "lat": 50.75, "lon": -1.34},
                    {"timestamp": 1621248967000, "lat": 50.77, "lon": -1.39},
                ]
            }
        ]
    }

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

# Fonction pour créer une carte avec les trajectoires des bateaux
def create_map(boats_data, selected_boats):
    """
    Crée une carte Folium avec les trajectoires des bateaux sélectionnés
    """
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
        return pd.DataFrame()

# Interface utilisateur principale
def main():
    st.title("YB Tracking Analyzer")
    st.write("Analysez les données de courses YB Tracking")
    
    # Sidebar pour les entrées utilisateur
    with st.sidebar:
        st.header("Configuration")
        
        # Entrée de l'ID de course
        race_id = st.text_input("ID de la course (ex: dgbr2025)", value="dgbr2025")
        
        # Bouton pour télécharger les données
        if st.button("Télécharger les données"):
            # Stocker les données dans la session
            if "data" not in st.session_state:
                st.session_state.data = fetch_yb_data(race_id)
                
                # Extraire les classes de bateaux
                if st.session_state.data["RaceSetup"]:
                    st.session_state.classes = extract_boat_classes(st.session_state.data["RaceSetup"])
                    st.session_state.boats_by_class = extract_boats_by_class(
                        st.session_state.data["RaceSetup"],
                        st.session_state.classes
                    )
                    
                    # Préparer les données des bateaux
                    if st.session_state.data["AllPositions"] and "boats" in st.session_state.data["AllPositions"]:
                        st.session_state.boats_data = []
                        
                        for boat in st.session_state.data["AllPositions"]["boats"]:
                            if "positions" in boat and boat["positions"]:
                                # Calculer la vitesse et le cap pour chaque position
                                positions = calculate_speed_and_bearing(boat["positions"])
                                
                                st.session_state.boats_data.append({
                                    "id": boat["id"],
                                    "name": boat["name"],
                                    "positions": positions
                                })
                        
                        st.success(f"Données téléchargées et traitées pour {len(st.session_state.boats_data)} bateaux")
                    else:
                        st.error("Aucune donnée de position disponible")
                else:
                    st.error("Impossible de récupérer les informations de la course")
            else:
                st.info("Les données sont déjà chargées. Utilisez le bouton 'Réinitialiser' pour charger une nouvelle course.")
        
        # Bouton pour réinitialiser les données
        if st.button("Réinitialiser"):
            # Effacer les données de la session
            if "data" in st.session_state:
                del st.session_state.data
            if "classes" in st.session_state:
                del st.session_state.classes
            if "boats_by_class" in st.session_state:
                del st.session_state.boats_by_class
            if "boats_data" in st.session_state:
                del st.session_state.boats_data
            if "selected_class" in st.session_state:
                del st.session_state.selected_class
            if "selected_boats" in st.session_state:
                del st.session_state.selected_boats
            
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
                    
                    # Bouton pour sélectionner tous les bateaux
                    if st.button("Sélectionner tous les bateaux"):
                        st.session_state.selected_boats = boat_options
                        st.experimental_rerun()
                    
                    # Bouton pour désélectionner tous les bateaux
                    if st.button("Désélectionner tous les bateaux"):
                        st.session_state.selected_boats = []
                        st.experimental_rerun()
            
            # Type d'analyse
            st.header("Analyse")
            analysis_type = st.selectbox(
                "Type d'analyse",
                ["Vitesse", "Cap", "Carte", "Tableau comparatif"]
            )
            
            st.session_state.analysis_type = analysis_type
    
    # Contenu principal
    if "boats_data" in st.session_state and "selected_boats" in st.session_state:
        # Filtrer les bateaux sélectionnés
        filtered_boats = [
            boat for boat in st.session_state.boats_data
            if boat["name"] in st.session_state.selected_boats
        ]
        
        # Afficher l'analyse sélectionnée
        if st.session_state.analysis_type == "Vitesse":
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
                                "Timestamp": datetime.fromtimestamp(pos["timestamp"]/1000),
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
                                "Timestamp": datetime.fromtimestamp(pos["timestamp"]/1000),
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
        
        elif st.session_state.analysis_type == "Carte":
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
                            "Timestamp": datetime.fromtimestamp(pos["timestamp"]/1000),
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
        st.info("Entrez l'ID de la course et cliquez sur 'Télécharger les données' pour commencer")
    else:
        st.warning("Sélectionnez une classe et des bateaux dans le menu latéral")

if __name__ == "__main__":
    main()
