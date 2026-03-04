import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def generar_caso_de_uso_curva_k_distances():
    """
    Genera un caso de uso aleatorio (input/output esperado)
    para curva_k_distances(X, k=5)
    """
    rng = np.random.default_rng()

    n_samples = random.randint(120, 320)
    n_features = random.randint(2, 8)

    # Datos con "blobs" para que la curva tenga estructura
    n_blobs = random.randint(2, 5)
    blobs = []
    for _ in range(n_blobs):
        center = rng.normal(loc=0.0, scale=4.0, size=n_features)
        m = random.randint(25, 90)
        blobs.append(rng.normal(loc=center, scale=rng.uniform(0.3, 1.3), size=(m, n_features)))
    X = np.vstack(blobs)

    # Agregar algo de ruido uniforme
    n_noise = random.randint(10, 50)
    noise = rng.uniform(low=-10, high=10, size=(n_noise, n_features))
    X = np.vstack([X, noise])

    k = int(random.choice([3, 4, 5, 6, 8, 10]))
    if k < 2:
        k = 2

    input_data = {"X": X.copy(), "k": k}

    # ----- Ground truth -----
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(Xs)
    distances, _ = nn.kneighbors(Xs)  # shape (n_samples, k)
    kdist = distances[:, k - 1]       # k-ésimo vecino (index k-1)
    kdist_sorted = np.sort(kdist.astype(float))

    output_data = {
        "k": int(k),
        "k_distances_sorted": np.round(kdist_sorted, 6)
    }
    return input_data, output_data
