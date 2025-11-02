# SensCritique : Recommandation des critiques similaires

## But de projet
Ce projet vise à recommander les critiques similaires du meme film, à partir d’une critique lue par l’utilisateur.  
Ce système utilise les embeddings sémantiques modernes (E5 multilingual) pour comprendre le sens des reviews, et calculer la similarité cosinus afin de trouver les critiques les plus proches ou similaires.

---

## Architecture du système : 

### 1. Mode Offline 
Pipeline exécuté une seule fois pour nettoyer et vectoriser les données textuelles.

Les étapes :
1. **Importation et nettoyage de données** :  
   - Fusion des deux fichies CSV (`Fight Club`, `Interstellar`)  
   - Suppression des valeurs NaN et des doublons  
   - Suppression des balises HTML, normalisation Unicode, mise en minuscule, et suppression de la ponctuation inutile  
   - Concaténation du texte sous la forme "titre.critique"
2. **Vectorisation (embedding)** :  
   - Génération d’embeddings avec le modèle E5 'intfloat/multilingual-e5-base', une version améliorée de SBERT 
   - Ce modèle est à la fois multilingue, accurate, gratuit, et fonctionne localement
3. **Persistance** :  
   - 'merged_clean.parquet' → données nettoyées sous format parquet 
   - 'embeddings.npz' → vecteurs compressés (embeddings) sous format numpy 
   - 'film_index.json' → liste des films disponibles dans les données 


---

### 2. Mode Online 
Au moment de la requête :
- Le système filtre les critiques du film demandé  
- Encode la critique lue ("query: ..." pour le syntaxe de E5)  
- Calcule la similarité cosinus avec les embeddings correspondants  
- Puis affiche les Top-N critiques les plus similaires, bbien formatées et nettoyées (balises HTML, ponctuation)

---

## Notebook d’accompagnement

Le notebook _sys_recommandation.ipynb_ détaille de manière approfondie :
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
```

### Lancer le test Online
```bash
python recommandation_sys_Demo.py
```

---

## Amélioration :

### - Utlisation de FAISS représente une solution plus optimale

FAISS permet de rechercher très rapidement les vecteurs les plus proches dans une grande base d’embeddings, car il permet :

- Eviter de recalculer la similarité cosinus sur toutes les critiques  
- De charger uniquement les vecteurs du film demandé  
- Garantir une meilleur scalabilité, chaque film a son propre index

> **Note :** :  
> FAISS n’a pas été implémenté dans ce projet à cause des conflits de bibliothèques dans mon venv

### Utlisation de Docker pour conteunariser le systeme, facilitera le deploiement du systeme en production
## Decoupage de la pipeline en des microservices independants (prettraitement, stockage, api) afin de maintenir une flexibilite en monté du deploiement
## Exposer le systeme via une API REST, testable depuis un navigateur ou un front-end

---

## Utilisation de l'IA dans ce projet :
L’IA a été utilisée dans ce projet à plusieurs niveaux :

- J'avais utlisé l'IA dans la partie prétraitement du texte pour trouver les bons regex,  pour supprimer les balises HTML, nettoyer les titres et uniformiser les critiques
- Aussi dans la partie amelioration de mon systeme, pour chercher la meilleure approche d’optimisation des calculs pour une grande base de données (FAISS) 
- L’IA a également été utilisée pour résoudre les conflits de libraries et ajuster la compatibilité entre sentence-transformers, et torch`  

