import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from fuzzywuzzy import fuzz, process
import time

# Load the data
dummy = pd.read_csv('dummy.csv')

movie_title = dummy["primaryTitle"]
movie_tconst = dummy['tconst']

def fuzzy_find_movie(query, movie_list, threshold=80):
    """
    Trouve un film en utilisant la correspondance floue
    
    Args:
        query (str): Le titre recherché
        movie_list (list): Liste des titres de films
        threshold (int): Seuil de correspondance (0-100)
        
    Returns:
        str: Le titre du film le plus proche ou None si aucune correspondance
    """
    result = process.extractOne(query, movie_list)
    if result and result[1] >= threshold:
        return result[0]
    return None


# CSS pour améliorer le design
st.markdown("""
    <style>
        .poster {
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s;
        }
        .poster:hover {
            transform: scale(1.08);
        }
        .film-container {
            display: flex;
            align-items: center;
            gap: 20px;
        }
        .film-info {
            font-size: 18px;
        }
        .recommendations {
            display: flex;
            flex-wrap: wrap;
            justify-content: left;
            gap: 20px;
        }
        .card {
            padding: 10px;
            background: white;
            border-radius: 10px;
            text-align: center;
            width: 220px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }
        .back-button {
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize the page state if it doesn't exist
if "page" not in st.session_state:
    st.session_state["page"] = "main"

# Sidebar for movie selection and weights (always visible)
with st.sidebar:
    st.image('C:/Users/bmwsk/Desktop/WILD/project2/un_systeme_de_recommandation_de_films.png')
    st.title(f"Bienvenue sur le site de recommendation du groupe Peekaboo ")
    user_text_input = st.checkbox("🎬 Liste ?", value=True)
    
    if user_text_input:
        selected_movie = st.selectbox("Selectionnez dans la liste:", 
                                  options=movie_title,
                                  index=None)
    else:
        entered_movie = st.text_input("Entrez le nom du film: ")
        movie_titles = dummy["primaryTitle"].tolist()
        selected_movie = fuzzy_find_movie(entered_movie, movie_titles)
        if not selected_movie and entered_movie:
            st.write(f"Aucun film correspondant à '{entered_movie}' trouvé. Essayez un autre titre.")
        elif selected_movie:
            st.write(f"Film trouvé: {selected_movie}")
    
    # Only show weight sliders on the main page
    if st.session_state["page"] == "main":
        st.subheader("Quel critère est le plus important pour vous ? ")
        
        # Use a different key for the reset button
        if st.button('Reset', type='secondary', key="reset_button"):
            # Set default values directly in the session state
            st.session_state["actors_slider"] = 3.5
            st.session_state["directors_slider"] = 1.5
            st.session_state["genres_slider"] = 24.0
            st.session_state["studios_slider"] = 2.0
            st.rerun()

        # Initialize slider values if they don't exist
        if "actors_slider" not in st.session_state:
            st.session_state["actors_slider"] = 3.5
        if "directors_slider" not in st.session_state:
            st.session_state["directors_slider"] = 1.5
        if "genres_slider" not in st.session_state:
            st.session_state["genres_slider"] = 24.0
        if "studios_slider" not in st.session_state:
            st.session_state["studios_slider"] = 2.0

        # Create sliders using session state for values
        actors_weight = st.slider("Poids des Acteurs", min_value=0.0, max_value=10.0, 
                                value=st.session_state["actors_slider"], step=0.5, key="actors_slider")
        directors_weight = st.slider("Poids des Réalisateurs", min_value=0.0, max_value=10.0, 
                                    value=st.session_state["directors_slider"], step=0.5, key="directors_slider")
        genres_weight = st.slider("Poids des Genres", min_value=0.0, max_value=24.0, 
                                value=st.session_state["genres_slider"], step=0.5, key="genres_slider")
        studios_weight = st.slider("Poids des Studios", min_value=0.0, max_value=10.0, 
                                value=st.session_state["studios_slider"], step=0.5, key="studios_slider")

        # Store the actual weight values for calculations
        st.session_state["actors_weight"] = actors_weight
        st.session_state["directors_weight"] = directors_weight
        st.session_state["genres_weight"] = genres_weight
        st.session_state["studios_weight"] = studios_weight
            
# Main content area - controlled by the page state
if st.session_state["page"] == "main":
    # Main page with movie selection and recommendations
    if selected_movie is None:
        st.write("Please select a movie to get recommendations")
    else:
        with st.spinner("Loading..."):
            time.sleep(1)
            st.title(f"You have chosen {selected_movie}")
            
            # Data preprocessing
            X = dummy.select_dtypes(include='number')
            X_scaled = StandardScaler().fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

            # Feature weighting
            actors = X_scaled.loc[:, X_scaled.columns.str.contains('Actor|Actress')]
            directors = X_scaled.loc[:, X_scaled.columns.str.contains('Director')]
            genres = X_scaled.loc[:, X_scaled.columns.str.contains('Genre')]
            studios = X_scaled.loc[:, X_scaled.columns.str.contains('Studio')]
            writers = X_scaled.loc[:, X.columns.str.contains('Writer')]
            producers = X_scaled.loc[:, X.columns.str.contains('Producer')]
            composers = X_scaled.loc[:, X.columns.str.contains('Composer')]
            cinematographers = X.loc[:, X.columns.str.contains('Cinematographer')]
            countries = X_scaled.loc[:, X.columns.str.contains('Country')]

            # Apply weights
            X_scaled[actors.columns] *= st.session_state["actors_weight"]
            X_scaled[directors.columns] *= st.session_state["directors_weight"]
            X_scaled[genres.columns] *= st.session_state["genres_weight"]
            X_scaled[studios.columns] *= st.session_state["studios_weight"]
            X_scaled[writers.columns] = 100 * X_scaled[writers.columns] / len(writers.columns)
            X_scaled[producers.columns] = 100 * X_scaled[producers.columns] / len(producers.columns)
            X_scaled[composers.columns] = 100 * X_scaled[composers.columns] / len(composers.columns)
            X_scaled[cinematographers.columns] = 1 * X_scaled[cinematographers.columns] / len(cinematographers.columns)
            X_scaled[countries.columns] = 100 * X_scaled[countries.columns] / len(countries.columns)

            # Text similarity
            selected_overview = dummy.loc[dummy["primaryTitle"] == selected_movie, "title_clean"].values[0]
            vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=100)
            vectorizer.fit([selected_overview])
            X_tfidf = vectorizer.transform(dummy["title_clean"].fillna(""))
            dummy_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
            dummy_tfidf = 50 * dummy_tfidf  # 20, 10, 5
            
            # Combine features
            X_model = pd.concat([X_scaled, dummy_tfidf], axis=1)
            X = X_model.select_dtypes(include='number')

            # Find nearest neighbors
            nn = NearestNeighbors(n_neighbors=31, metric="cosine")
            nn.fit(X)
            movie_index = dummy[dummy["primaryTitle"] == selected_movie].index[0]
            distances, indices = nn.kneighbors([X.iloc[movie_index]])
            recommended_indices = indices[0][1:]

            # List of recommendations
            df_recommendations = dummy.iloc[recommended_indices][["primaryTitle", "averageRating", "startYear", "poster_path"]].copy()

            # Get details for the selected movie
            selected_movie_data = dummy.iloc[movie_index]
            poster_url = f"https://image.tmdb.org/t/p/w500{selected_movie_data['poster_path']}"

            # Display selected movie details
            st.markdown(f"<h2>{selected_movie}</h2>", unsafe_allow_html=True)
            col1, col2 = st.columns(2, gap='large')
            with col1:
                st.image(poster_url, width=250, caption=f"{selected_movie} ({selected_movie_data['startYear']})")
            with col2:
                st.markdown(f"""
                    <div class='film-info'>
                    <p><b>Overview :</b> {selected_movie_data['overview']}</p>
                    <p><b>🎬 Réalisateur(s) :</b> {selected_movie_data['liste_director']}</p>
                    <p><b>🎞️ Genre(s) :</b> {selected_movie_data['genres']}</p>
                    <p><b>🎭 Acteurs :</b> {selected_movie_data['liste_actors']}</p>
                    <p><b>🎭 Actrices :</b> {selected_movie_data['liste_actress']}</p>
                    <p><b>📅 Année :</b> {selected_movie_data['startYear']}</p>
                    <p><b>⭐ Note Moyenne :</b> {selected_movie_data['averageRating']}</p>
                    
                    </div>
                """, unsafe_allow_html=True)

            # Display recommended movies in rows of 4
            st.markdown("### 🎥 Films Recommandés")
            rows = [df_recommendations.iloc[i:i+4] for i in range(0, len(df_recommendations), 4)]
            
            for row in rows:
                cols = st.columns(4, gap='large')
                for col, (_, movie) in zip(cols, row.iterrows()):
                    with col:
                        movie_poster = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
                        st.image(movie_poster, width=150, caption=f"{movie['primaryTitle']} ({movie['startYear']})")
                        st.write(f"⭐ {movie['averageRating']}")
                        
                        # Button to navigate to details page
                        if st.button(f"ℹ️ Voir Infos", key=movie['primaryTitle']):
                            st.session_state["selected_recommended_movie"] = movie['primaryTitle']
                            st.session_state["page"] = "details"
                            st.rerun()

elif st.session_state["page"] == "details":
    # Details page showing information about a selected recommended movie
    if "selected_recommended_movie" in st.session_state:
        recommended_movie = st.session_state["selected_recommended_movie"]
        
        # Fetch the movie data
        movie_data = dummy[dummy["primaryTitle"] == recommended_movie].iloc[0]
        poster_url = f"https://image.tmdb.org/t/p/w500{movie_data['poster_path']}"

        # Display movie details
        st.title("Movie Details")
        st.markdown(f"<h2>{recommended_movie}</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap='large')
        with col1:
            st.image(poster_url, width=250, caption=f"{recommended_movie} ({movie_data['startYear']})")
        with col2:
            st.markdown(f"""
                <div class='film-info'>
                <p><b>Overview :</b> {movie_data['overview']}</p>
                <p><b>🎬 Réalisateur(s) :</b> {movie_data['liste_director']}</p>
                <p><b>🎞️ Genre(s) :</b> {movie_data['genres']}</p>
                <p><b>🎭 Acteurs :</b> {movie_data['liste_actors']}</p>
                <p><b>🎭 Actrices :</b> {movie_data['liste_actress']}</p>
                <p><b>📅 Année :</b> {movie_data['startYear']}</p>
                <p><b>⭐ Note Moyenne :</b> {movie_data['averageRating']}</p>

                </div>
            """, unsafe_allow_html=True)

        # Button to return to recommendations
        if st.button("⬅️ Retour aux recommandations", key="back_button"):
            st.session_state["page"] = "main"
            st.rerun()
    else:
        # Fallback if no movie is selected
        st.error("No movie selected. Please return to the main page.")
        if st.button("⬅️ Return to Main Page"):
            st.session_state["page"] = "main"
            st.rerun()
