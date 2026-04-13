# MachineLearningCD8Cells
Classification des sous-types de lymphocytes T CD8+ par scRNA-seq

Ce projet est dédié à l'analyse bio-informatique et à la classification automatisée de données de séquençage d'ARN en cellule unique. L'objectif principal est de distinguer trois populations cellulaires biologiquement proches : les lymphocytes T CD8+ de type Tcm/Naïve, Tem/Trm et Tem/Temra. La difficulté majeure réside dans la haute dimensionnalité des données et la nécessité d'extraire un signal biologique pertinent au milieu du bruit technique.

Contenu du dépôt
Le présent dépôt rassemble les ressources nécessaires à la compréhension et à la reproduction de l'étude. Le fichier scriptComplet.ipynb contient l'intégralité de la démarche technique, allant du prétraitement des données avec Scanpy jusqu'à l'évaluation des modèles de Machine Learning. En complément, le document Rapport.pdf détaille le cadre théorique, l'analyse comparative des méthodes de réduction de dimension et l'interprétation biologique des résultats obtenus. Le fichier pipelineCellTypist.py est nécessaire si vous ne possédez pas de données annotées. 

Approche technologique et méthodologique
Le pipeline s'appuie sur le langage Python et des bibliothèques spécialisées telles que Scanpy pour la manipulation des objets AnnData. L'étude compare l'efficacité de la réduction de dimension linéaire (PCA) et non-linéaire (UMAP, t-SNE) pour la visualisation des clusters. La partie classification oppose des modèles d'apprentissage supervisé, notamment les forêts aléatoires et XGBoost, entraînés alternativement sur les composantes principales et sur l'ensemble des gènes bruts. Cette double approche permet d'évaluer l'impact de la compression de l'information sur la précision du diagnostic cellulaire.

Synthèse des résultats
Les analyses démontrent que les modèles entraînés sur les gènes bruts surpassent ceux basés sur la PCA, XGBoost atteignant un F1-score de 0,94. Cette performance souligne la capacité des algorithmes de gradient boosting à traiter efficacement la haute dimensionnalité sans perte d'information préalable. L'étude a également permis d'identifier des biomarqueurs clés tels que GZMA, GZMK et CCR7, dont l'importance dans la discrimination des sous-types cellulaires est confirmée par la littérature scientifique.

Installation
Pour reproduire les analyses présentées dans le notebook, l'installation des dépendances suivantes est requise :

pip install scanpy pandas numpy matplotlib seaborn scikit-learn xgboost

Licence
Ce projet est mis à disposition sous licence MIT. Les utilisateurs sont libres de consulter, modifier et distribuer le code dans le cadre de leurs propres travaux de recherche.
