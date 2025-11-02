# SensCritique : Recommandation des critiques similaires

## But de projet
Ce projet vise à recommander les critiques similaires du meme film, à partir d’une critique lue par l’utilisateur.  
Ce système utilise les embeddings sémantiques modernes (E5 multilingual) pour comprendre le sens des reviews, et calculer la similarité cosinus pour trouver les critiques les plus proches ou similaires.

---

## Architecture du système : 

### 1. Mode Offline 
Pipeline exécuté une seule fois pour nettoyer et vectoriser les données textuelles.

Étapes :
1. **Importation & nettoyage de données** :  
   - Fusion des deux fichies CSV (`Fight Club`, `Interstellar`)  
   - Suppression des valeurs NaN et des doublons  
   - Suppression des balises HTML, normalisation Unicode, mise en minuscule, et suppression de la ponctuation inutile  
   - Concaténation du texte sous la forme "titre.critique"
2. **Vectorisation (embedding)** :  
   - Génération d’embeddings avec le modèle E5 'intfloat/multilingual-e5-base', une version améliorée de SBERT 
   - Ce modèle est à la fois multilingue, accurate, gratuit, et fonctionne localement
3. **Persistance** :  
   - 'merged_clean.parquet' → données nettoyées sous format parquet 
   - 'embeddings.npz' → vecteurs compressés (embeddings)  
   - 'film_index.json' → liste des films disponibles dans les données 


---

### 2. Mode Online (recherche en direct)
Au moment de la requête :
- Le système filtre les critiques du film demandé  
- Encode la critique lue ("query: ..." pour E5)  
- Calcule la similarité cosinus avec les embeddings correspondants  
- Affiche les Top-N critiques les plus similaires, bbien formatées et nettoyées (balises HTML, ponctuation)

---

## Notebook d’accompagnement

Le fichier _sys_recommandation.ipynb_ détaille de manière approfondie :
- les choix techniques et méthodologiques du projet,  
- la préparation textuelle (regex, nettoyage HTML, normalisation, concat titre.critique),  
- la comparaison des modèles d’embeddings,  
- une exploration des données (qualité, distributions),  
- des commentaires explicatifs pour chaque étape du traitement et du modèle.

Ce notebook permet de comprendre et reproduire facilement chaque étape du système.

---

## Exécution

### Préparation de données (offline)
```bash
python batch_preparation.py

### Lancer le test Online
```bash
python recommandation_sys_Demo.py
```

---

## Amélioration :

## Utlisation de FAISS représente une solution plus optimale

**FAISS (Facebook AI Similarity Search)** permet de rechercher très rapidement les **vecteurs les plus proches** dans une grande base d’embeddings.

### 🔍 Avantages :
- **Vitesse** : évite de recalculer la similarité cosinus sur toutes les critiques  
- **Mémoire** : charge uniquement les vecteurs du film demandé  
- **Scalabilité** : chaque film a son propre index, facilement extensible  
- **Réactivité** : résultats quasi instantanés même avec une base volumineuse  

> ⚠️ **Note importante** :  
> FAISS n’a **pas été implémenté dans ce projet final** à cause de **conflits de bibliothèques** rencontrés lors des tests.  
> Cependant, son **intégration reste la solution la plus optimale** pour rendre le système rapide et scalable sur une grande base multi-films.


