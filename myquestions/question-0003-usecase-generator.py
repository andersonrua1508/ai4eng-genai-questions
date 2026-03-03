import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

def generar_caso_de_uso_clusterizar_dbscan():
    """
    Genera un caso de uso aleatorio (input/output esperado)
    para clusterizar_dbscan(X, eps, min_samples)
    """
    rng = np.random.default_rng()

    n_clusters_true = random.randint(2, 5)
    n_features = random.randint(2, 6)

    blobs = []
    for _ in range(n_clusters_true):
        center = rng.normal(loc=0.0, scale=4.0, size=n_features)
        m = random.randint(25, 70)
        blobs.append(rng.normal(loc=center, scale=rng.uniform(0.3, 1.2), size=(m, n_features)))

    X = np.vstack(blobs)

    # agregar ruido
    n_noise = random.randint(10, 40)
    noise = rng.uniform(low=-10, high=10, size=(n_noise, n_features))
    X = np.vstack([X, noise])

    eps = float(random.choice([0.3, 0.5, 0.7, 1.0]))
    min_samples = int(random.choice([3, 4, 5, 6]))

    input_data = {"X": X.copy(), "eps": eps, "min_samples": min_samples}

    # ---- Ground truth ----
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(Xs)

    labels = model.labels_
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    n_noise2 = int(np.sum(labels == -1))

    output_data = {"labels": labels, "n_clusters": n_clusters, "n_noise": n_noise2}
    return input_data, output_data
