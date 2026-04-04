import random
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score

def generar_caso_de_uso_evaluar_con_baseline_kfold():
    """
    Genera un caso de uso aleatorio (input/output esperado)
    para evaluar_con_baseline_kfold(X, y, k=5, random_state=0)
    """
    rng = np.random.default_rng()

    n_samples = random.randint(140, 360)
    n_features = random.randint(4, 14)

    X = rng.normal(size=(n_samples, n_features))

    # señal lineal + ruido -> clasificación binaria
    w = rng.normal(size=n_features)
    logits = X @ w + rng.normal(scale=0.6, size=n_samples)

    # umbral para crear desbalance moderado a veces
    thr = np.quantile(logits, random.choice([0.45, 0.5, 0.55]))
    y = (logits > thr).astype(int)

    k = int(random.choice([3, 4, 5, 6]))
    if k < 2:
        k = 2
    random_state = int(random.randint(0, 50))

    input_data = {
        "X": X.copy(),
        "y": y.copy(),
        "k": k,
        "random_state": random_state,
    }

    # ----- Ground truth -----
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)

    lift_acc = []
    lift_f1 = []
    acc_lr = []
    acc_dummy = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        lr = LogisticRegression(max_iter=1000, random_state=random_state)
        lr.fit(X_train_s, y_train)
        pred_lr = lr.predict(X_test_s)

        dum = DummyClassifier(strategy="most_frequent", random_state=random_state)
        dum.fit(X_train_s, y_train)
        pred_d = dum.predict(X_test_s)

        a_lr = accuracy_score(y_test, pred_lr)
        a_d = accuracy_score(y_test, pred_d)

        f_lr = f1_score(y_test, pred_lr, average="binary", zero_division=0)
        f_d = f1_score(y_test, pred_d, average="binary", zero_division=0)

        acc_lr.append(a_lr)
        acc_dummy.append(a_d)
        lift_acc.append(a_lr - a_d)
        lift_f1.append(f_lr - f_d)

    output_data = {
        "lift_accuracy_mean": float(np.round(np.mean(lift_acc), 6)),
        "lift_f1_mean": float(np.round(np.mean(lift_f1), 6)),
        "logreg_accuracy_mean": float(np.round(np.mean(acc_lr), 6)),
        "dummy_accuracy_mean": float(np.round(np.mean(acc_dummy), 6)),
    }

    return input_data, output_data
