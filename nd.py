
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score


# ============================================================
# 1. CLASSE DES NUÉES DYNAMIQUES
# ============================================================

class NueesDynamiques:

    def __init__(
        self,
        n_clusters=3,
        max_iter=100,
        tolerance=1e-4,
        random_state=None,
        verbose=True
    ):

        if n_clusters < 1:
            raise ValueError(
                "Le nombre de clusters doit être supérieur à 0."
            )

        if max_iter < 1:
            raise ValueError(
                "Le nombre maximal d'itérations doit être supérieur à 0."
            )

        if tolerance <= 0:
            raise ValueError(
                "La tolérance doit être strictement positive."
            )

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state
        self.verbose = verbose

        # Attributs du modèle
        self.prototypes = None
        self.labels = None
        self.inertia_ = None
        self.n_iter_ = 0
        self.inertia_history_ = []
        self.prototype_history_ = []

    # ========================================================
    # 2. VALIDATION DES DONNÉES
    # ========================================================

    def _validate_data(self, X):

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "X doit être une matrice 2D."
            )

        if X.shape[0] < self.n_clusters:
            raise ValueError(
                "Le nombre de données doit être supérieur "
                "ou égal au nombre de clusters."
            )

        if not np.all(np.isfinite(X)):
            raise ValueError(
                "Les données ne doivent pas contenir "
                "de NaN ou de valeurs infinies."
            )

        return X

    # ========================================================
    # 3. INITIALISATION DES PROTOTYPES
    # ========================================================

    def _initialize_prototypes(self, X):

        rng = np.random.default_rng(
            self.random_state
        )

        indices = rng.choice(
            X.shape[0],
            size=self.n_clusters,
            replace=False
        )

        self.prototypes = X[indices].copy()

    # ========================================================
    # 4. CALCUL DES DISTANCES
    # ========================================================

    def _calculate_distances(self, X):

        """
        Calcule la distance euclidienne au carré
        entre chaque observation et chaque prototype.

        Résultat :
            matrice de dimension (n_observations, n_clusters)
        """

        differences = (
            X[:, np.newaxis, :] -
            self.prototypes[np.newaxis, :, :]
        )

        distances = np.sum(
            differences ** 2,
            axis=2
        )

        return distances

    # ========================================================
    # 5. AFFECTATION DES OBSERVATIONS
    # ========================================================

    def _assign_clusters(self, X):

        distances = self._calculate_distances(X)

        labels = np.argmin(
            distances,
            axis=1
        )

        return labels

    # ========================================================
    # 6. MISE À JOUR DES PROTOTYPES
    # ========================================================

    def _update_prototypes(self, X, labels):

        new_prototypes = np.zeros_like(
            self.prototypes
        )

        rng = np.random.default_rng(
            self.random_state
        )

        for k in range(self.n_clusters):

            cluster_points = X[
                labels == k
            ]

            # Si le cluster contient des observations
            if len(cluster_points) > 0:

                new_prototypes[k] = np.mean(
                    cluster_points,
                    axis=0
                )

            # Gestion d'un cluster vide
            else:

                random_index = rng.integers(
                    low=0,
                    high=X.shape[0]
                )

                new_prototypes[k] = X[
                    random_index
                ]

                if self.verbose:

                    print(
                        f"Cluster vide détecté : {k}. "
                        "Réinitialisation du prototype."
                    )

        return new_prototypes

    # ========================================================
    # 7. CALCUL DE L'INERTIE
    # ========================================================

    def _calculate_inertia(self, X, labels):

        distances = self._calculate_distances(X)

        # Sélectionne la distance du prototype attribué
        selected_distances = distances[
            np.arange(X.shape[0]),
            labels
        ]

        inertia = np.sum(
            selected_distances
        )

        return inertia

    # ========================================================
    # 8. ENTRAÎNEMENT DU MODÈLE
    # ========================================================

    def fit(self, X):

        X = self._validate_data(X)

        # Initialisation
        self._initialize_prototypes(X)

        self.inertia_history_ = []
        self.prototype_history = []

        for iteration in range(self.max_iter):

            # Sauvegarde des anciens prototypes
            old_prototypes = self.prototypes.copy()

            # Étape 1 : affectation
            labels = self._assign_clusters(X)

            # Étape 2 : mise à jour
            new_prototypes = self._update_prototypes(
                X,
                labels
            )

            # Calcul du déplacement des prototypes
            prototype_shift = np.linalg.norm(
                new_prototypes - old_prototypes
            )

            # Mise à jour des prototypes
            self.prototypes = new_prototypes

            # Calcul de l'inertie
            inertia = self._calculate_inertia(
                X,
                labels
            )

            self.inertia_history_.append(
                inertia
            )

            self.prototype_history.append(
                self.prototypes.copy()
            )

            if self.verbose:

                print(
                    f"Itération {iteration + 1:03d} | "
                    f"Inertie : {inertia:.4f} | "
                    f"Déplacement : {prototype_shift:.6f}"
                )

            # Vérification de la convergence
            if prototype_shift < self.tolerance:

                if self.verbose:

                    print(
                        "\nConvergence atteinte."
                    )

                self.n_iter_ = iteration + 1

                break

        else:

            self.n_iter_ = self.max_iter

            if self.verbose:

                print(
                    "\nNombre maximal d'itérations atteint."
                )

        # Affectation finale
        self.labels = self._assign_clusters(X)

        # Inertie finale
        self.inertia_ = self._calculate_inertia(
            X,
            self.labels
        )

        return self

    # ========================================================
    # 9. PRÉDICTION SUR DE NOUVELLES OBSERVATIONS
    # ========================================================

    def predict(self, X):

        if self.prototypes is None:

            raise ValueError(
                "Le modèle doit être entraîné avant predict()."
            )

        X = self._validate_data_for_predict(X)

        return self._assign_clusters(X)

    # ========================================================
    # 10. VALIDATION POUR PREDICT
    # ========================================================

    def _validate_data_for_predict(self, X):

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:

            raise ValueError(
                "X doit être une matrice 2D."
            )

        if X.shape[1] != self.prototypes.shape[1]:

            raise ValueError(
                "Le nombre de variables de X ne correspond "
                "pas aux prototypes."
            )

        if not np.all(np.isfinite(X)):

            raise ValueError(
                "Les données contiennent des valeurs invalides."
            )

        return X

    # ========================================================
    # 11. AFFICHAGE DES INFORMATIONS
    # ========================================================

    def summary(self):

        if self.labels is None:

            raise ValueError(
                "Le modèle doit être entraîné avant summary()."
            )

        print("\n" + "=" * 55)
        print("RÉSUMÉ DU MODÈLE DES NUÉES DYNAMIQUES")
        print("=" * 55)

        print(
            f"Nombre de clusters : {self.n_clusters}"
        )

        print(
            f"Nombre d'itérations : {self.n_iter_}"
        )

        print(
            f"Inertie finale : {self.inertia_:.4f}"
        )

        print("\nPrototypes finaux :")

        for k, prototype in enumerate(self.prototypes):

            print(
                f"Cluster {k} : {prototype}"
            )

        print("\nEffectifs des clusters :")

        for k in range(self.n_clusters):

            count = np.sum(
                self.labels == k
            )

            print(
                f"Cluster {k} : {count} observations"
            )

        print("=" * 55)


# ============================================================
# 12. GÉNÉRATION D'UN JEU DE DONNÉES SYNTHÉTIQUE
# ============================================================

def generate_data(random_state=42):

    rng = np.random.default_rng(
        random_state
    )

    cluster_1 = rng.normal(
        loc=[2, 2],
        scale=1.0,
        size=(100, 2)
    )

    cluster_2 = rng.normal(
        loc=[8, 3],
        scale=1.0,
        size=(100, 2)
    )

    cluster_3 = rng.normal(
        loc=[5, 8],
        scale=1.0,
        size=(100, 2)
    )

    X = np.vstack(
        [
            cluster_1,
            cluster_2,
            cluster_3
        ]
    )

    return X


# ============================================================
# 13. VISUALISATION DES DONNÉES INITIALES
# ============================================================

def plot_initial_data(X):

    plt.figure(figsize=(8, 6))

    plt.scatter(
        X[:, 0],
        X[:, 1],
        s=35,
        alpha=0.7
    )

    plt.title(
        "Données initiales"
    )

    plt.xlabel(
        "Variable 1"
    )

    plt.ylabel(
        "Variable 2"
    )

    plt.grid(
        alpha=0.3
    )

    plt.show()


# ============================================================
# 14. VISUALISATION DES CLUSTERS
# ============================================================

def plot_clusters(X, model):

    plt.figure(figsize=(9, 7))

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=model.labels,
        s=40,
        alpha=0.7
    )

    plt.scatter(
        model.prototypes[:, 0],
        model.prototypes[:, 1],
        marker="X",
        s=300,
        edgecolors="black",
        linewidths=1.5,
        label="Prototypes"
    )

    plt.title(
        "Résultat du clustering par nuées dynamiques"
    )

    plt.xlabel(
        "Variable 1"
    )

    plt.ylabel(
        "Variable 2"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.show()


# ============================================================
# 15. VISUALISATION DE L'ÉVOLUTION DE L'INERTIE
# ============================================================

def plot_inertia_history(model):

    plt.figure(figsize=(9, 5))

    plt.plot(
        range(
            1,
            len(model.inertia_history_) + 1
        ),
        model.inertia_history_,
        marker="o"
    )

    plt.title(
        "Évolution de l'inertie au cours des itérations"
    )

    plt.xlabel(
        "Itération"
    )

    plt.ylabel(
        "Inertie"
    )

    plt.grid(
        alpha=0.3
    )

    plt.show()


# ============================================================
# 16. VISUALISATION DU DÉPLACEMENT DES PROTOTYPES
# ============================================================

def plot_prototype_evolution(X, model):

    plt.figure(figsize=(9, 7))

    plt.scatter(
        X[:, 0],
        X[:, 1],
        s=25,
        alpha=0.3
    )

    history = np.array(
        model.prototype_history
    )

    for k in range(model.n_clusters):

        plt.plot(
            history[:, k, 0],
            history[:, k, 1],
            marker="o",
            label=f"Prototype {k}"
        )

        plt.scatter(
            history[0, k, 0],
            history[0, k, 1],
            marker="s",
            s=100
        )

        plt.scatter(
            history[-1, k, 0],
            history[-1, k, 1],
            marker="X",
            s=200
        )

    plt.title(
        "Évolution des prototypes"
    )

    plt.xlabel(
        "Variable 1"
    )

    plt.ylabel(
        "Variable 2"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.show()


# ============================================================
# 17. ÉVALUATION AVEC LA SILHOUETTE
# ============================================================

def evaluate_model(X, model):

    labels = model.labels

    unique_labels = np.unique(
        labels
    )

    if len(unique_labels) < 2:

        print(
            "Impossible de calculer la silhouette : "
            "une seule classe détectée."
        )

        return None

    score = silhouette_score(
        X,
        labels
    )

    print(
        f"\nScore de silhouette : {score:.4f}"
    )

    return score


# ============================================================
# 18. RECHERCHE DU NOMBRE DE CLUSTERS
# ============================================================

def test_different_k(
    X,
    k_values=range(2, 8),
    random_state=42
):

    inertias = []
    silhouette_scores = []

    for k in k_values:

        print(
            f"\nTest avec K = {k}"
        )

        model = NueesDynamiques(
            n_clusters=k,
            max_iter=100,
            tolerance=1e-4,
            random_state=random_state,
            verbose=False
        )

        model.fit(X)

        inertias.append(
            model.inertia_
        )

        score = silhouette_score(
            X,
            model.labels
        )

        silhouette_scores.append(
            score
        )

        print(
            f"Inertie : {model.inertia_:.4f}"
        )

        print(
            f"Silhouette : {score:.4f}"
        )

    # Graphique de l'inertie
    plt.figure(figsize=(8, 5))

    plt.plot(
        list(k_values),
        inertias,
        marker="o"
    )

    plt.title(
        "Méthode du coude"
    )

    plt.xlabel(
        "Nombre de clusters K"
    )

    plt.ylabel(
        "Inertie"
    )

    plt.grid(
        alpha=0.3
    )

    plt.show()

    # Graphique de la silhouette
    plt.figure(figsize=(8, 5))

    plt.plot(
        list(k_values),
        silhouette_scores,
        marker="o"
    )

    plt.title(
        "Score de silhouette selon K"
    )

    plt.xlabel(
        "Nombre de clusters K"
    )

    plt.ylabel(
        "Score de silhouette"
    )

    plt.grid(
        alpha=0.3
    )

    plt.show()

    return inertias, silhouette_scores


# ============================================================
# 19. PROGRAMME PRINCIPAL
# ============================================================

if __name__ == "__main__":

    # Génération des données
    X = generate_data(
        random_state=42
    )

    print(
        "Dimensions des données :",
        X.shape
    )

    # Visualisation initiale
    plot_initial_data(
        X
    )

    # Création du modèle
    model = NueesDynamiques(
        n_clusters=3,
        max_iter=100,
        tolerance=1e-4,
        random_state=42,
        verbose=True
    )

    # Entraînement
    model.fit(
        X
    )

    # Résumé
    model.summary()

    # Évaluation
    evaluate_model(
        X,
        model
    )

    # Visualisation des clusters
    plot_clusters(
        X,
        model
    )

    # Évolution de l'inertie
    plot_inertia_history(
        model
    )

    # Évolution des prototypes
    plot_prototype_evolution(
        X,
        model
    )

    # Recherche du nombre de clusters
    test_different_k(
        X,
        k_values=range(2, 8),
        random_state=42
    )

    # Exemple de prédiction
    new_data = np.array(
        [
            [2, 3],
            [8, 4],
            [5, 7]
        ]
    )

    predictions = model.predict(
        new_data
    )

    print(
        "\nPrédictions pour les nouvelles observations :"
    )

    for observation, prediction in zip(
        new_data,
        predictions
    ):

        print(
            f"Observation {observation} "
            f"→ Cluster {prediction}"
        )