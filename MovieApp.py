import streamlit as st
import pandas as pd
import requests
from sklearn.neighbors import NearestNeighbors
from translate import Translator

# Image de fond
st.image('MovieApp.svg')


# Api tmdb - clef
url = "https://api.themoviedb.org/3/authentication"
headers = {
    "accept": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI1ZWMyMGJhM2JhMDY3Y2MwMWI3ZjQ2ZGVkZTViYTQ0OCIsInN1YiI6IjY1NjIyMmJmN2RmZGE2NTkzMDRiMzU0ZSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.ZhSey6voKWvmjdL3NlLcSeCBdPFgEH2ur2xUNBuiXlU"
}
response = requests.get(url, headers=headers)
print(response.text)


# Dictionnaire des genres
genre_mapping = {
    '28': 'Action',
    '12': 'Adventure',
    '16': 'Animation',
    '35': 'Comedy',
    '80': 'Crime',
    '99': 'Documentary',
    '18': 'Drama',
    '10751': 'Family',
    '14': 'Fantasy',
    '36': 'History',
    '27': 'Horror',
    '10402': 'Music',
    '9648': 'Mystery',
    '10749': 'Romance',
    '878': 'Science Fiction',
    '10770': 'TV Movie',
    '53': 'Thriller',
    '10752': 'War',
    '37': 'Western'
}

# Charger les DataFrames
import zipfile

# DataFrame ML
chemin_zip = 'df_ml.zip'
# Lisez le fichier CSV directement depuis le fichier ZIP
with zipfile.ZipFile(chemin_zip, 'r') as zip_ref:
    with zip_ref.open(zip_ref.namelist()[0]) as file:
        df_ml = pd.read_csv(file)

df_actors = pd.read_csv("actorsIdAndNames.csv", index_col=0)
df_actmov = pd.read_csv("actorsInFilmsTable.csv")
df_all = pd.read_csv("df_frenchComedies2000_all.csv", index_col=0)
df_directors = pd.read_csv("directorsTable.csv", index_col=0)
df_genres = pd.read_csv("tableGenresDesFilms.csv", index_col=0)
df_all_exploded = pd.read_csv("df_test1.csv")
df_test = pd.read_csv("df_test2_list.csv")

# Fusionner les tables en fonction des identifiants tconst, nconst, etc.
merged_df = pd.merge(df_all, df_actmov, on='tconst', how='inner')  # Fusionner avec les acteurs et films
merged_df = pd.merge(merged_df, df_actors, left_on='actor_id', right_on='nconst', how='inner')  # Fusionner avec les informations sur les acteurs
merged_df = pd.merge(merged_df, df_directors, on='tconst', how='inner')  # Fusionner avec les informations sur les réalisateurs
merged_df = pd.merge(merged_df, df_genres, on='tconst', how='inner')  # Fusionner avec les genres des films


def search_tab_1():
    st.title("Trouver un film")
    # Filter movies by release year and genre
    filtered_movies = df_all_exploded

    # User input for additional criteria
    selected_actor = st.selectbox("Select an actor:", ["None"] + list(df_actors['primaryName'].unique()))
    selected_director = st.selectbox("Select a director:", ["None"] + list(df_directors['primaryName'].unique()))
    selected_genre = st.selectbox("Select a genre:", ["None"] + list(df_genres['genres_x'].unique()))

    # Apply additional filters based on user input
    # Apply additional filters based on user input
    if selected_actor != "None":
        filtered_movies = filtered_movies[filtered_movies['Actors_Name'] == selected_actor]

    if selected_director != "None":
        filtered_movies = filtered_movies[filtered_movies['Directors_Name'] == selected_director]

    if selected_genre != "None":
        filtered_movies = filtered_movies[filtered_movies['Movie_Genres'] == selected_genre]

        # Optionally, you can retrieve additional information from TMDb for each movie in the filtered list
        # Here's an example for getting additional information for the first movie in the list
        if not filtered_movies.empty:
            movie_id = filtered_movies['ID']
            tmdb_url = f"https://api.themoviedb.org/3/find/{movie_id}?external_source=imdb_id"
            tmdb_response = requests.get(tmdb_url)

            if tmdb_response.status_code == 200:
                tmdb_data = tmdb_response.json()
                st.write("Additional information from TMDb:")
                st.write(f"Title: {tmdb_data['title']}")
                st.write(f"Overview: {tmdb_data['overview']}")
                st.write(f"Release Date: {tmdb_data['release_date']}")
                st.write(f"Genres: {', '.join([genre['name'] for genre in tmdb_data['genres']])}")
                st.write(f"Average Vote: {tmdb_data['vote_average']}")
                st.write(f"Popularity: {tmdb_data['popularity']}")
                st.write(f"Original Language: {tmdb_data['original_language']}")
                st.image(f"https://image.tmdb.org/t/p/w500/{tmdb_data['poster_path']}", caption='Movie Poster', use_column_width=True)
            else:
                st.warning("Unable to fetch additional information from TMDb.")



# Onget de recherche de films
def search_tab_2():
    st.title("Recherche de Films")
    
    # Ajoutez ici vos composants de recherche, résultats, etc.
    liste_films = df_test['Movie_Title']
    search_query = st.selectbox("J'aimerais voir un film similaire à:", liste_films, index=None, placeholder="Saisissez le titre d'un film que vous avez aimé")
    weight_option = st.selectbox("J'aimerais essentiellement retrouver:", ['Les réalisateurs', 'Les acteurs', 'Le genre', 'Un peu tout !'])
  
    
    if st.button("Rechercher"):
        # Logique de recherche et affichage des résultats
        st.write(f"Résultats de la recherche pour: {search_query}")

        # Utiliser le film choisi comme variable
        film = search_query

        from sklearn.preprocessing import MultiLabelBinarizer
        import numpy as np

        # factorisation des listes en une seule variable (de liste)
        if weight_option == 'Un peu tout !':
            mlb = MultiLabelBinarizer()
            genres_binarized= mlb.fit_transform(df_test['Movie_Genres'])*1.5
            actors_binarized= mlb.fit_transform(df_test['Actors_Name'])*1.1
            directors_binarized= mlb.fit_transform(df_test['Directors_Name'])*1.3
            year_np= np.array(df_test['Movie_Year']).reshape(-1, 1)

        if weight_option == 'Les réalisateurs':
            mlb = MultiLabelBinarizer()
            genres_binarized= mlb.fit_transform(df_test['Movie_Genres'])*1.3
            actors_binarized= mlb.fit_transform(df_test['Actors_Name'])*1.1
            directors_binarized= mlb.fit_transform(df_test['Directors_Name'])*1.5
            year_np= np.array(df_test['Movie_Year']).reshape(-1, 1)


        if weight_option == 'Les acteurs':
            mlb = MultiLabelBinarizer()
            genres_binarized= mlb.fit_transform(df_test['Movie_Genres'])*1.3
            actors_binarized= mlb.fit_transform(df_test['Actors_Name'])*1.5
            directors_binarized= mlb.fit_transform(df_test['Directors_Name'])*1.1
            year_np= np.array(df_test['Movie_Year']).reshape(-1, 1)

        if weight_option == 'Le genre':
            mlb = MultiLabelBinarizer()
            genres_binarized= mlb.fit_transform(df_test['Movie_Genres'])*1.7
            actors_binarized= mlb.fit_transform(df_test['Actors_Name'])*1.2
            directors_binarized= mlb.fit_transform(df_test['Directors_Name'])*1.2
            year_np= np.array(df_test['Movie_Year']).reshape(-1, 1)

        numerics_variable = np.hstack((genres_binarized, actors_binarized, directors_binarized, year_np))

        # initialiser et entrainer modèle de nearest neighbors
        from sklearn.neighbors import NearestNeighbors
        from sklearn.metrics.pairwise import cosine_similarity
        X = numerics_variable
        modelNN = NearestNeighbors(n_neighbors = 4, metric='cosine', algorithm='brute')
        modelNN.fit(X)

        index_film= df_test.index[df_test['Movie_Title'].str.contains(film)]

        distances, indices = modelNN.kneighbors(numerics_variable[index_film])
        #nearest_neighbors= element[0][1:]
        #nearest_films= df_test.iloc[nearest_neighbors]
        #nearest_films'''
                
        #df_without_movie = df_ml[df_ml['Movie_Title'] != film]
        #features_without_movie = df_without_movie.select_dtypes(exclude=['object'])

        # Initialiser le modèle Nearest Neighbors
        #nn_model = NearestNeighbors(n_neighbors=3, algorithm='ball_tree')

        # Entraîner le modèle sur le DataFrame sans le film
        #nn_model.fit(features_without_movie)

        # Rechercher les voisins les plus proches pour le film
        #movie_to_query = df_ml[df_ml['Movie_Title'] == film].select_dtypes(exclude=['object'])
        #distances, indices = nn_model.kneighbors(movie_to_query)

        # Afficher les informations des trois films les plus proches
        for i in range(3):
            nearest_movie_index = indices[0][i+1]
            nearest_movie = df_test.iloc[nearest_movie_index]

            # Obtenir l'ID IMDb du film
            imdb_id = nearest_movie['ID']

            # Appeler l'API TMDB pour obtenir les informations du film
            tmdb_url = f"https://api.themoviedb.org/3/find/{imdb_id}?external_source=imdb_id"
            tmdb_response = requests.get(tmdb_url, headers=headers)

            if tmdb_response.status_code == 200:
                tmdb_data = tmdb_response.json()
                
                # Traduire le résumé de l'anglais au français
                translator = Translator(to_lang='fr')
                english_summary = tmdb_data['movie_results'][0]['overview']

                # Split the text into chunks to avoid exceeding the character limit
                chunk_size = 400  # Adjust the chunk size as needed
                chunks = [english_summary[i:i + chunk_size] for i in range(0, len(english_summary), chunk_size)]

                # Translate each chunk and concatenate the results
                french_summary = " ".join(translator.translate(chunk) for chunk in chunks)

                # Afficher les informations du film à partir de l'API TMDB
                st.title(f"Informations pour le film : {nearest_movie['Movie_Title']}")

                # Afficher l'image et le résumé côte à côte
                col1, col2 = st.columns(2)
                col1.image(f"https://image.tmdb.org/t/p/w500/{tmdb_data['movie_results'][0]['poster_path']}", caption='Affiche', use_column_width=True)
                col2.write(f"Résumé : {french_summary}")

                # Afficher d'autres détails du film
                col2.subheader("Détails du Film:")
                col2.write(f"Date de sortie: {tmdb_data['movie_results'][0]['release_date']}")
                genre_ids = tmdb_data['movie_results'][0]['genre_ids']
                genre_names = [genre_mapping.get(str(genre_id), '') for genre_id in genre_ids]
                col2.write(f"Genres: {', '.join(genre_names)}")
                col2.write(f"Note moyenne: {tmdb_data['movie_results'][0]['vote_average']}")
                col2.write(f"Popularité: {tmdb_data['movie_results'][0]['popularity']}")


                # Ajoutez d'autres informations selon vos besoins
            else:
                st.warning(f"Impossible de récupérer les informations du film {nearest_movie['Movie_Title']} à partir de l'API TMDB.")


# Créer un DataFrame pour stocker les profils utilisateur
user_profiles = pd.DataFrame(columns=['User', 'Favorite_Actors', 'Favorite_Producers', 'Favorite_Movies'])

# Créer une fonction pour afficher la page de gestion des profils
def profile_tab():
    global user_profiles  # Déclarer user_profiles comme une variable globale

    st.title("Gestion des Profils Utilisateur")

    # Ajouter un composant de saisie pour le nom de l'utilisateur
    user_name = st.text_input("Nom du Profil:")

    # Ajouter des composants multiselect pour les acteurs, producteurs et films préférés
    favorite_actors = st.multiselect("Sélectionnez vos 5 acteurs préférés:", df_actors['primaryName'].unique(), key='actors')
    favorite_producers = st.multiselect("Sélectionnez vos 5 producteurs préférés:", df_directors['primaryName'].unique(), key='producers')
    favorite_movies = st.multiselect("Sélectionnez vos 5 films préférés:", df_ml['Movie_Title'].unique(), key='movies')

    # Ajouter un bouton pour enregistrer le profil utilisateur
    if st.button("Enregistrer le Profil"):
        # Vérifier si le profil existe déjà
        if user_name in user_profiles['User'].values:
            st.warning("Ce nom de profil existe déjà. Veuillez choisir un autre nom.")
        else:
            # Ajouter le profil utilisateur au DataFrame
            user_profiles = pd.concat([user_profiles, pd.DataFrame({
                'User': [user_name],
                'Favorite_Actors': [favorite_actors],
                'Favorite_Producers': [favorite_producers],
                'Favorite_Movies': [favorite_movies]
            })], ignore_index=True)
            st.success("Profil enregistré avec succès!")

    # Afficher les profils enregistrés
    st.title("Profils Utilisateur Enregistrés")
    st.table(user_profiles)

# Ajouter un onglet pour la gestion des profils
tabs = {"Trouver un film": search_tab_1, "Recherche de Films": search_tab_2, "Profils Utilisateur": profile_tab}

# Barre de navigation pour sélectionner les onglets
selected_tab = st.sidebar.radio("Navigation", list(tabs.keys()))

# Afficher l'onglet sélectionné
tabs[selected_tab]()
