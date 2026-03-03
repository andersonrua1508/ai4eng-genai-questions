import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

def generar_caso_de_uso_top_features_permutation_importance():
    """
    Genera un caso de uso aleatorio (input/output esperado)
    para top_features_permutation_importance(X_train, y_train, X_test, y_test, n_top=5, random_state=0)
    """
    rng = np.random.default_rng()

    n_train = random.randint(120, 260)
    n_test = random.randint(60, 140)
    n_features = random.randint(5, 14)

    X_train = rng.normal(size=(n_train, n_features))
    X_test = rng.normal(size=(n_test, n_features))

    rel = rng.choice(np.arange(n_features), size=random.randint(2, 4), replace=False)
    w = rng.normal(loc=0.0, scale=3.0, size=n_features)
    nonrel = np.setdiff1d(np.arange(n_features), rel)
    w[nonrel] *= 0.2

    y_train = X_train @ w + rng.normal(scale=rng.uniform(0.5, 1.5), size=n_train)
    y_test = X_test @ w + rng.normal(scale=rng.uniform(0.5, 1.5), size=n_test)

    n_top = int(random.choice([3, 5, 7]))
    random_state = int(random.randint(0, 50))

    input_data = {
        "X_train": X_train.copy(),
        "y_train": y_train.copy(),
        "X_test": X_test.copy(),
        "y_test": y_test.copy(),
        "n_top": n_top,
        "random_state": random_state,
    }

    # ---- Ground truth ----
    model = RandomForestRegressor(random_state=random_state)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)

    pi = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        random_state=random_state,
        scoring="r2"
    )

    imp = pi.importances_mean.astype(float)
    order = sorted(range(n_features), key=lambda i: (-imp[i], i))
    top_idx = order[:n_top]
    top_imps = np.round(imp[top_idx], 6)

    output_data = {
        "top_idx": [int(i) for i in top_idx],
        "top_importances": top_imps,
        "r2_test": float(np.round(r2, 6)),
    }
    return input_data, output_data
