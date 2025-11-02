# ================================================================================================
#                 pipeline offline : nettoyage + entraînement + embeddings
# ================================================================================================

import os, re, html, json, unicodedata
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer


# ------------ Set up et configuration ------------
#configuration des chemin
data_path = "data"
proc_path = os.path.join(data_path, "Processed Data")
os.makedirs(proc_path, exist_ok=True)

#importation des données
FC_data = os.path.join(data_path, "fightclub_critiques.csv")
IT_data = os.path.join(data_path, "interstellar_critique.csv")

# sauvegarde des donnes pretraitées et les embeddings sous format numpy compressé
cleaned_data_parquet = os.path.join(proc_path, "merged_clean.parquet")
cleaned_data_csv = os.path.join(proc_path, "merged_clean.csv")
embeddings_npz = os.path.join(proc_path, "embeddings.npz")
index_films = os.path.join(proc_path, "film_index.json")
#index_dir = os.path.join(proc_path, "index_by_film")
#os.makedirs(index_dir, exist_ok=True)

# nom de modele E5
nom_modele = "intfloat/multilingual-e5-base"


# ========================================================
#                Fonctions utliles
# ========================================================

def importation_nettoyage_data():
    fc_df = pd.read_csv(FC_data)
    it_df = pd.read_csv(IT_data)
    fc_df["film"] = "Fight Club"
    it_df["film"] = "Interstellar"
    data = pd.concat([fc_df, it_df], ignore_index=True)

    # suppression des NaN et des doublons
    data = (data.dropna(subset=["review_title", "review_content"]).reset_index(drop=True))
    data = data.drop_duplicates().reset_index(drop=True)

    return data

def nettoyage_critiques(title, html):
    # suppression des balises HTML et de les ponctuations
    cleaned_review = BeautifulSoup(str(html), "html.parser").get_text(" ")
    cleaned_review = re.sub(r"[^\w\s]", " ", cleaned_review)

    # titre propre
    title = str(title).strip()
    title = re.sub(r"[^\w\s]", " ", title)
    title = re.sub(r"\s+$", "", title)

    # combiner et normaliser
    title_review = f"{title}. {cleaned_review}".strip()
    title_review = unicodedata.normalize("NFKC", title_review)
    title_review = re.sub(r"\s+", " ", title_review).strip().lower()

    return title_review

def generation_embeddings(texts, nom_modele):
    model = SentenceTransformer(nom_modele)
    review_texts = ["passage: " + t for t in texts]   # Préfixe pour E5 : "passage:" pour les reviews
    embeddings = model.encode(review_texts, normalize_embeddings=True, show_progress_bar=True)

    return embeddings



def main():

    print(" ======== Début de traitement Offline ...  ========")

    # data load -> data transformation -> generation de embeddings
    data = importation_nettoyage_data()
    data["review_clean"] = data.apply(lambda x: nettoyage_critiques(x["review_title"], x["review_content"]), axis=1)
    embeddings = generation_embeddings(data["review_clean"].astype(str).tolist(), nom_modele)

    # les sauvegardes
    data.to_parquet(cleaned_data_parquet, index=False)
    data.to_csv(cleaned_data_csv, index=False, encoding="utf-8")
    np.savez_compressed(embeddings_npz, embeddings=embeddings)

    # meta data : noms des films existant
    meta_data= sorted(data["film"].dropna().unique().tolist())
    with open(index_films, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=2)
    #print(meta_data)


    print(" ======== Traitement Offline terminé ! ========")



if __name__ == "__main__":
    main()








