"""
Pipeline scRNA-seq : CD8+ T cells of Healthy Donor 3 (10x Genomics) + CellTypist

Dépendances :
  pip install scanpy celltypist matplotlib seaborn
"""

import os
import scanpy as sc
import celltypist
from celltypist import models
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # backend sans interface graphique
import seaborn as sns
import pandas as pd
import numpy as np

# Paramètres globaux 
H5_PATH     = r"vdj_v1_hs_aggregated_donor3_filtered_feature_bc_matrix.h5"

RESULTS_DIR = "./results"
FIGURES_DIR = "./figures"

# Seuils QC
MIN_GENES        = 100    # gènes minimum par cellule
MAX_GENES        = 5000   # gènes maximum (filtre doublets)
MAX_PCT_MITO     = 20     # % mitochondrial maximum
MIN_CELLS        = 3      # cellules minimum par gène

# CellTypist
CELLTYPIST_MODEL = "Immune_All_Low.pkl"   # modèle immunitaire haute résolution
MAJORITY_VOTING  = True

sc.settings.verbosity = 2
sc.settings.figdir    = FIGURES_DIR

# ── Utilitaires ─────────────────────────────────────────────────────────────

def make_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ── Étape 1 : Vérification du fichier local ───────────────────────────────────

def step1_check_file():
    make_dirs(RESULTS_DIR, FIGURES_DIR)
    h5_path = os.path.abspath(H5_PATH)
    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"\n[ERREUR] Fichier HDF5 introuvable : {h5_path}\n"
            f"  → Modifiez la variable H5_PATH en haut du script."
        )
    size_mb = os.path.getsize(h5_path) / (1024 ** 2)
    print(f"[INFO] Fichier HDF5 trouvé : {h5_path}  ({size_mb:.1f} Mo)")
    return h5_path


# ── Étape 2 : Chargement & filtrage Feature Type ─────────────────────────────

def step2_load(h5_path: str) -> sc.AnnData:
    """
    Charge le fichier HDF5 multimodal (Gene Expression + Antibody Capture).
    Conserve uniquement les features Gene Expression pour CellTypist.
    """
    print("\n[ÉTAPE 2] Chargement du fichier HDF5 ...")
    adata = sc.read_10x_h5(h5_path, gex_only=False)
    adata.var_names_make_unique()

    print(f"  Données brutes  : {adata.shape[0]} cellules × {adata.shape[1]} features")
    print(f"  Types de features : {adata.var['feature_types'].value_counts().to_dict()}")

    # Conserver uniquement Gene Expression
    gex_mask = adata.var["feature_types"] == "Gene Expression"
    adata = adata[:, gex_mask].copy()
    print(f"  Après filtre GEX : {adata.shape[0]} cellules × {adata.shape[1]} gènes")
    return adata


# ── Étape 3 : Contrôle qualité ───────────────────────────────────────────────

def step3_qc(adata: sc.AnnData) -> sc.AnnData:
    """Calcul des métriques QC et filtrage cellules/gènes."""
    print("\n[ÉTAPE 3] Contrôle qualité ...")

    # Gènes mitochondriaux (préfixe MT-)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    # ── Figures QC ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(adata.obs["n_genes_by_counts"], bins=60, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Nombre de gènes / cellule")
    axes[0].set_ylabel("Nombre de cellules")
    axes[0].set_title("Distribution des gènes")
    axes[0].axvline(MIN_GENES, color="red",    linestyle="--", label=f"Min = {MIN_GENES}")
    axes[0].axvline(MAX_GENES, color="orange", linestyle="--", label=f"Max = {MAX_GENES}")
    axes[0].legend()

    axes[1].hist(adata.obs["total_counts"], bins=60, color="seagreen", edgecolor="white")
    axes[1].set_xlabel("UMI totaux / cellule")
    axes[1].set_title("Distribution des UMIs")

    axes[2].hist(adata.obs["pct_counts_mt"], bins=60, color="salmon", edgecolor="white")
    axes[2].set_xlabel("% mitochondrial / cellule")
    axes[2].set_title("% mitochondrial")
    axes[2].axvline(MAX_PCT_MITO, color="red", linestyle="--", label=f"Max = {MAX_PCT_MITO}%")
    axes[2].legend()

    plt.tight_layout()
    qc_fig_path = os.path.join(FIGURES_DIR, "qc_metrics.png")
    plt.savefig(qc_fig_path, dpi=150)
    plt.close()
    print(f"  Figure QC sauvegardée → {qc_fig_path}")

    # ── Filtrage ──
    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=MIN_GENES)
    sc.pp.filter_cells(adata, max_genes=MAX_GENES)
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS)
    adata = adata[adata.obs["pct_counts_mt"] < MAX_PCT_MITO].copy()
    n_after = adata.n_obs

    print(f"  Cellules avant filtrage : {n_before}")
    print(f"  Cellules après filtrage : {n_after} (retirées : {n_before - n_after})")
    print(f"  Gènes conservés         : {adata.n_vars}")
    return adata


# ── Étape 4 : Normalisation & log-transformation ──────────────────────────────

def step4_normalize(adata: sc.AnnData) -> sc.AnnData:
    """
    CellTypist requiert des données normalisées à 10 000 UMIs/cellule
    et log1p-transformées (log-normalized counts).
    On sauvegarde les counts bruts dans adata.layers['counts'].
    """
    print("\n[ÉTAPE 4] Normalisation ...")
    adata.layers["counts"] = adata.X.copy()          # sauvegarde counts bruts
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata                                 # sauvegarde pour visualisation
    print("  Normalisation 10k UMI + log1p effectuée.")
    return adata


# ── Étape 5 : HVG, PCA, voisins, UMAP ────────────────────────────────────────

def step5_embedding(adata: sc.AnnData) -> sc.AnnData:
    """Sélection de gènes variables, réduction dimensionnelle et clustering."""
    print("\n[ÉTAPE 5] Réduction dimensionnelle ...")

    # Gènes hautement variables
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
    print(f"  Gènes hautement variables sélectionnés : {adata.var['highly_variable'].sum()}")

    # PCA (sur les HVGs uniquement)
    sc.tl.pca(adata, svd_solver="arpack", use_highly_variable=True)

    # Graphe des k-plus-proches-voisins
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)

    # UMAP
    sc.tl.umap(adata)

    # Clustering de Leiden pour référence
    sc.tl.leiden(adata, resolution=0.5, key_added="leiden")
    print(f"  Clusters Leiden : {adata.obs['leiden'].nunique()}")

    # Figure UMAP (clustering Leiden)
    fig_leiden = os.path.join(FIGURES_DIR, "umap_leiden.png")
    sc.pl.umap(
        adata,
        color="leiden",
        legend_loc="on data",
        title="UMAP – Clusters Leiden",
        save=False,
        show=False,
    )
    plt.savefig(fig_leiden, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  UMAP Leiden sauvegardé → {fig_leiden}")

    return adata


# ── Étape 6 : Annotation CellTypist ──────────────────────────────────────────

def step6_celltypist(adata: sc.AnnData) -> sc.AnnData:
    """
    Annotation automatique avec CellTypist.
    Modèle : Immune_All_Low.pkl  (résolution fine, cellules immunitaires humaines).
    """
    print(f"\n[ÉTAPE 6] Annotation CellTypist (modèle : {CELLTYPIST_MODEL}) ...")

    # Téléchargement automatique du modèle si absent
    models.download_models(model=CELLTYPIST_MODEL, force_update=False)
    model = models.Model.load(model=CELLTYPIST_MODEL)

    # CellTypist attend des log-normalized counts (déjà appliqués à l'étape 4)
    predictions = celltypist.annotate(
        adata,
        model=model,
        majority_voting=MAJORITY_VOTING,
    )

    # Transfert des annotations vers adata
    adata = predictions.to_adata()

    # Résumé des prédictions
    print("\n  Distribution des types cellulaires (majority_voting) :")
    ct_counts = adata.obs["majority_voting"].value_counts()
    for ct, n in ct_counts.items():
        pct = 100 * n / adata.n_obs
        print(f"    {ct:<45} {n:>6} cellules ({pct:5.1f}%)")

    return adata


# ── Étape 7 : Visualisations & export ────────────────────────────────────────

def step7_visualize_and_export(adata: sc.AnnData):
    """Génère les figures finales et exporte les résultats."""
    print("\n[ÉTAPE 7] Visualisations & export ...")

    # ── UMAP coloré par type cellulaire ──
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    sc.pl.umap(
        adata,
        color="majority_voting",
        legend_loc="right margin",
        title="UMAP – CellTypist (majority voting)",
        ax=axes[0],
        show=False,
    )
    sc.pl.umap(
        adata,
        color="predicted_labels",
        legend_loc="right margin",
        title="UMAP – CellTypist (predicted labels)",
        ax=axes[1],
        show=False,
    )
    plt.tight_layout()
    umap_ct_path = os.path.join(FIGURES_DIR, "umap_celltypist.png")
    plt.savefig(umap_ct_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  UMAP CellTypist sauvegardé → {umap_ct_path}")

    # ── Score de confiance par type cellulaire ──
    # Récupération du score max pour chaque cellule (confiance de prédiction)
    if "conf_score" in adata.obs.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        order = (
            adata.obs.groupby("majority_voting")["conf_score"]
            .median()
            .sort_values(ascending=False)
            .index
        )
        sns.boxplot(
            data=adata.obs,
            x="majority_voting",
            y="conf_score",
            order=order,
            ax=ax,
            palette="Set2",
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.set_title("Score de confiance CellTypist par type cellulaire")
        ax.set_xlabel("")
        ax.set_ylabel("Score de confiance")
        plt.tight_layout()
        conf_path = os.path.join(FIGURES_DIR, "confidence_scores.png")
        plt.savefig(conf_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Scores de confiance sauvegardés → {conf_path}")

    # ── Heatmap : top gènes marqueurs par cluster ──
    sc.tl.rank_genes_groups(
        adata,
        groupby="majority_voting",
        method="wilcoxon",
        n_genes=5,
        use_raw=True,
    )
    fig_markers = os.path.join(FIGURES_DIR, "top_markers_dotplot.png")
    sc.pl.rank_genes_groups_dotplot(
        adata,
        n_genes=3,
        show=False,
        save=False,
    )
    plt.savefig(fig_markers, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Dotplot marqueurs sauvegardé → {fig_markers}")

    # ── UMAP : gènes CD8 signature ──
    cd8_genes = [g for g in ["CD8A", "CD8B", "CD3D", "CD3E", "GZMB", "PRF1",
                              "FOXP3", "PDCD1", "LAG3", "HAVCR2"] if g in adata.raw.var_names]
    if cd8_genes:
        fig_genes = os.path.join(FIGURES_DIR, "umap_cd8_markers.png")
        sc.pl.umap(
            adata,
            color=cd8_genes,
            use_raw=True,
            ncols=5,
            show=False,
            save=False,
        )
        plt.savefig(fig_genes, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  UMAP gènes CD8 sauvegardé → {fig_genes}")

    # ── Export CSV des annotations ──
    obs_cols = [c for c in [
        "n_genes_by_counts", "total_counts", "pct_counts_mt",
        "leiden", "predicted_labels", "majority_voting", "conf_score",
    ] if c in adata.obs.columns]

    csv_path = os.path.join(RESULTS_DIR, "cell_annotations.csv")
    adata.obs[obs_cols].to_csv(csv_path)
    print(f"  Annotations exportées → {csv_path}")

    # ── Résumé par type cellulaire ──
    summary = (
        adata.obs.groupby("majority_voting")
        .agg(
            n_cells=("majority_voting", "count"),
            mean_conf=("conf_score", "mean") if "conf_score" in adata.obs.columns else ("majority_voting", "count"),
            mean_genes=("n_genes_by_counts", "mean"),
            mean_umis=("total_counts", "mean"),
        )
        .sort_values("n_cells", ascending=False)
        .round(2)
    )
    summary_path = os.path.join(RESULTS_DIR, "celltypist_summary.csv")
    summary.to_csv(summary_path)
    print(f"  Résumé CellTypist exporté → {summary_path}")

    # ── Sauvegarde AnnData complet ──
    h5ad_path = os.path.join(RESULTS_DIR, "cd8_donor3_annotated.h5ad")
    adata.write_h5ad(h5ad_path)
    print(f"  AnnData annoté sauvegardé → {h5ad_path}")

    return adata


# ── Pipeline principale ───────────────────────────────────────────────────────

def run_pipeline():
    print("=" * 65)
    print("  Pipeline CD8+ T cells Healthy Donor 3 — CellTypist")
    print("=" * 65)

    h5_path = step1_check_file()
    adata   = step2_load(h5_path)
    adata   = step3_qc(adata)
    adata   = step4_normalize(adata)
    adata   = step5_embedding(adata)
    adata   = step6_celltypist(adata)
    adata   = step7_visualize_and_export(adata)

    print("\n" + "=" * 65)
    print("  Pipeline terminée avec succès !")
    print(f"  Figures   : {FIGURES_DIR}/")
    print(f"  Résultats : {RESULTS_DIR}/")
    print("=" * 65)
    return adata


if __name__ == "__main__":
    adata = run_pipeline()
