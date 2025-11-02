#================================================================================================
#        pipeline Online : embeddings critiques lues + affichage des critiques similaires
# ================================================================================================
import os, re, html, json
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# recupération des fichiers utiles généré en mode offline
data_path = "data"
proc_path = os.path.join(data_path, "Processed Data")
cleaned_data_parquet = os.path.join(proc_path, "merged_clean.parquet")
embeddings_npz = os.path.join(proc_path, "embeddings.npz")
meta_path = os.path.join(proc_path, "film_index.json")
#index_path = os.path.join(proc_path, "index_by_film")

nom_modele = "intfloat/multilingual-e5-base"

# Chargement données
#print(" ======= Chargement des fichiers...  ")
data = pd.read_parquet(cleaned_data_parquet)
emb_all = np.load(embeddings_npz)["embeddings"]
model = SentenceTransformer(nom_modele)

# Ajout des embeddings au Ddata
data = data.copy()
data["embeddings"] = list(emb_all)

def charger_films(meta_path):
    with open(meta_path, "r", encoding="utf-8") as f:
        films = json.load(f)
    return set(films)

def formater_critiques_simailaires(text):

      """Cette fonction assure un affichage propre des critiques similaire pour la sortie utilisateur
      """"
      txt = BeautifulSoup(text, "html.parser").get_text(" ")
      txt = html.unescape(txt)
      txt = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", txt)
      txt = re.sub(r"[?!]{2,}", " ", txt)
      txt = re.sub(r"(\?\s*){2,}", " ", txt)
      #txt = re.sub(r"[^\w\s.,;:!?\'’”“€éèàùêôîïç]", " ", txt)
      txt = re.sub(r"\s+", " ", txt).strip()

      return txt


def recommander_critiques_similaires(critique_lue, film, top_n=5):

    """Cette fonction renvoie et affiche les critiques similaire de meme film
    """

    films_disponibles = charger_films(meta_path)

    if film in films_disponibles:
        data_film = data[data["film"] == film].reset_index(drop=True)
        embeddings_film = np.vstack(data_film["embeddings"].values)

        critique_emb = model.encode(["query: " + str(critique_lue)], normalize_embeddings=True)

        simalarity = cosine_similarity(critique_emb, embeddings_film)[0]
        top_idx = np.argsort(simalarity)[::-1][:top_n]

        cols = ["film", "rating", "username", "review_title", "review_content"]
        resultat = data_film.iloc[top_idx][cols]

        for _, cr in resultat.iterrows():
            print(formater_critiques_simailaires(cr["review_content"]))
    else:
        print(f"Aucune critique n'est disponible pour le film {film}  :/")
        return None


if __name__ == "__main__":
    # Demo
    top_n = 10
    #film = "Titanium"
    film = "Fight Club"
    critique_lue = "Je n'ai pas aimé, trop de bagarres à mains nues"

    #film = input("Film : ").strip()
    #critique_lue = input("La critique lue: ").strip()

    recommander_critiques_similaires(critique_lue, film, top_n=top_n)

