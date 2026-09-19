
import numpy as np
import matplotlib.pyplot as plt

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
        verbose=True,
        representation="points_representatifs"
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
        self.representation = self._normalize_representation(
            representation
        )

        # Attributs du modèle
        self.prototypes = None
        self.labels = None
        self.inertia_ = None
        self.n_iter_ = 0
        self.inertia_history_ = []
        self.prototype_history_ = []
        self.n_features_in_ = None

    @staticmethod
    def _normalize_representation(representation):

        aliases = {
            "points_representatifs": "points_representatifs",
            "points représentatifs": "points_representatifs",
            "axes_factoriels": "axes_factoriels",
            "axes factoriels": "axes_factoriels",
            "distribution": "distribution",
            "structure_representative": "structure_representative",
            "structure représentative": "structure_representative",
            "point": "point",
        }

        key = str(representation).strip().lower()

        if key not in aliases:
            raise ValueError(
                "Représentation inconnue. Choisissez parmi : "
                "points_representatifs, axes_factoriels, distribution, "
                "structure_representative ou point."
            )

        return aliases[key]

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

                if self.representation == "structure_representative":
                    distances = np.sum(
                        (cluster_points[:, np.newaxis, :] -
                         cluster_points[np.newaxis, :, :]) ** 2,
                        axis=2
                    )
                    new_prototypes[k] = cluster_points[
                        np.argmin(np.sum(distances, axis=1))
                    ]
                else:
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

    def _fit_factorial_axes(self, X):

        self.pca_mean_ = np.mean(X, axis=0)
        centered = X - self.pca_mean_
        _, singular_values, right_vectors = np.linalg.svd(
            centered,
            full_matrices=False
        )
        n_components = min(2, X.shape[0], X.shape[1])
        self.pca_components_ = right_vectors[:n_components]
        self.pca_values_ = singular_values[:n_components]

        return centered @ self.pca_components_.T

    def _transform_factorial_axes(self, X):

        return (X - self.pca_mean_) @ self.pca_components_.T

    def _fit_distribution(self, X):

        rng = np.random.default_rng(self.random_state)
        means = X[rng.choice(
            X.shape[0],
            size=self.n_clusters,
            replace=False
        )].copy()
        covariances = np.array([
            np.cov(X, rowvar=False) +
            np.eye(X.shape[1]) * 1e-6
            for _ in range(self.n_clusters)
        ])
        weights = np.full(self.n_clusters, 1 / self.n_clusters)
        previous_log_likelihood = -np.inf

        for iteration in range(self.max_iter):

            probabilities = np.zeros((X.shape[0], self.n_clusters))

            for cluster in range(self.n_clusters):
                covariance = covariances[cluster]
                inverse = np.linalg.pinv(covariance)
                determinant = max(np.linalg.det(covariance), 1e-12)
                differences = X - means[cluster]
                exponent = -0.5 * np.sum(
                    (differences @ inverse) * differences,
                    axis=1
                )
                coefficient = 1 / np.sqrt(
                    (2 * np.pi) ** X.shape[1] * determinant
                )
                probabilities[:, cluster] = (
                    weights[cluster] * coefficient * np.exp(exponent)
                )

            totals = np.maximum(
                np.sum(probabilities, axis=1),
                1e-300
            )
            responsibilities = probabilities / totals[:, np.newaxis]
            log_likelihood = np.sum(np.log(totals))

            effective_sizes = np.sum(responsibilities, axis=0)
            weights = effective_sizes / X.shape[0]

            for cluster in range(self.n_clusters):
                if effective_sizes[cluster] < 1e-12:
                    means[cluster] = X[rng.integers(X.shape[0])]
                    covariances[cluster] = np.cov(
                        X,
                        rowvar=False
                    ) + np.eye(X.shape[1]) * 1e-6
                    weights[cluster] = 1 / X.shape[0]
                    continue

                means[cluster] = np.sum(
                    responsibilities[:, cluster, np.newaxis] * X,
                    axis=0
                ) / effective_sizes[cluster]
                differences = X - means[cluster]
                covariances[cluster] = (
                    (responsibilities[:, cluster, np.newaxis] * differences).T
                    @ differences / effective_sizes[cluster]
                    + np.eye(X.shape[1]) * 1e-6
                )

            if abs(log_likelihood - previous_log_likelihood) < self.tolerance:
                self.n_iter_ = iteration + 1
                break

            previous_log_likelihood = log_likelihood

        else:
            self.n_iter_ = self.max_iter

        self.distribution_means_ = means
        self.distribution_covariances_ = covariances
        self.distribution_weights_ = weights

        return self._predict_distribution(X)

    def _predict_distribution(self, X):

        probabilities = np.zeros((X.shape[0], self.n_clusters))

        for cluster in range(self.n_clusters):
            covariance = self.distribution_covariances_[cluster]
            inverse = np.linalg.pinv(covariance)
            determinant = max(np.linalg.det(covariance), 1e-12)
            differences = X - self.distribution_means_[cluster]
            exponent = -0.5 * np.sum(
                (differences @ inverse) * differences,
                axis=1
            )
            coefficient = 1 / np.sqrt(
                (2 * np.pi) ** X.shape[1] * determinant
            )
            probabilities[:, cluster] = (
                self.distribution_weights_[cluster] *
                coefficient * np.exp(exponent)
            )

        return np.argmax(probabilities, axis=1)

    # ========================================================
    # 8. ENTRAÎNEMENT DU MODÈLE
    # ========================================================

    def fit(self, X):

        X = self._validate_data(X)
        self.n_features_in_ = X.shape[1]

        if self.representation == "axes_factoriels":
            X = self._fit_factorial_axes(X)

        if self.representation == "distribution":
            self.labels = self._fit_distribution(X)
            self.prototypes = self.distribution_means_.copy()
            self.inertia_ = self._calculate_inertia(X, self.labels)
            self.inertia_history_ = [self.inertia_]
            self.prototype_history = [self.prototypes.copy()]
            return self

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

        if self.representation == "axes_factoriels":
            X = self._transform_factorial_axes(X)

        if self.representation == "distribution":
            return self._predict_distribution(X)

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

        if X.shape[1] != self.n_features_in_:

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



# 15. VISUALISATION DE L'ÉVOLUTION DE L'INERTIE


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



# 16. VISUALISATION DU DÉPLACEMENT DES PROTOTYPES


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



# 17. ÉVALUATION AVEC LA SILHOUETTE


def calculate_silhouette_score(X, labels):

    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    distances = np.sqrt(
        np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=2)
    )
    scores = np.zeros(X.shape[0])

    for index in range(X.shape[0]):
        same_cluster = labels == labels[index]
        same_cluster[index] = False

        if not np.any(same_cluster):
            continue

        mean_intra_distance = np.mean(distances[index, same_cluster])
        mean_inter_distances = []

        for other_label in np.unique(labels):
            if other_label == labels[index]:
                continue

            other_cluster = labels == other_label
            mean_inter_distances.append(
                np.mean(distances[index, other_cluster])
            )

        nearest_cluster_distance = min(mean_inter_distances)
        denominator = max(
            mean_intra_distance,
            nearest_cluster_distance
        )

        if denominator > 0:
            scores[index] = (
                nearest_cluster_distance - mean_intra_distance
            ) / denominator

    return float(np.mean(scores))

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

    score = calculate_silhouette_score(
        X,
        labels
    )

    print(
        f"\nScore de silhouette : {score:.4f}"
    )

    return score



# 18. RECHERCHE DU NOMBRE DE CLUSTERS


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

        score = calculate_silhouette_score(
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



# 19. PROGRAMME PRINCIPAL


def choose_representation():

    choices = {
        "1": "points_representatifs",
        "2": "axes_factoriels",
        "3": "distribution",
        "4": "structure_representative",
        "5": "point",
    }

    print("\nChoisissez la représentation de la nuée dynamique :")
    print("1. Ensemble des points représentatifs")
    print("2. Axes factoriels")
    print("3. Distribution")
    print("4. Structure représentative")
    print("5. Point (K-means)")

    choice = input("Votre choix [1-5] : ").strip()

    if choice not in choices:
        raise ValueError("Le choix doit être compris entre 1 et 5.")

    return choices[choice]

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

    representation = choose_representation()

    # Création du modèle
    model = NueesDynamiques(
        n_clusters=3,
        max_iter=100,
        tolerance=1e-4,
        random_state=42,
        verbose=True,
        representation=representation
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