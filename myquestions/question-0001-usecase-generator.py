import random
import json
import numpy as np
import pandas as pd

def generar_caso_de_uso_expandir_json_y_resumir():
    """
    Genera un caso de uso aleatorio (input/output esperado)
    para expandir_json_y_resumir(df, json_col, user_col)
    """
    rng = np.random.default_rng()

    n_users = random.randint(3, 10)
    users = [f"u{i}" for i in range(n_users)]
    n = random.randint(40, 140)

    keys = [f"k{i}" for i in range(random.randint(4, 9))]

    rows = []
    for _ in range(n):
        u = rng.choice(users)
        kk = rng.choice(keys, size=int(rng.integers(0, len(keys) + 1)), replace=False).tolist()
        d = {k: float(rng.integers(0, 20)) for k in kk}

        p = rng.random()
        if p < 0.55:
            payload = json.dumps(d)
        elif p < 0.80:
            payload = d
        elif p < 0.92:
            payload = "{json_invalido"
        else:
            payload = None

        rows.append((u, payload))

    df = pd.DataFrame(rows, columns=["user", "payload"])

    # meter algunos user NaN
    for idx in rng.choice(df.index.to_numpy(), size=max(1, n // 25), replace=False):
        df.loc[idx, "user"] = np.nan

    input_data = {"df": df.copy(), "json_col": "payload", "user_col": "user"}

    # ---- Ground truth ----
    dfx = df.copy()
    dfx = dfx.dropna(subset=["user"]).copy()

    dicts = []
    keycount = []
    for x in dfx["payload"].tolist():
        d = {}
        if isinstance(x, dict):
            d = x
        elif isinstance(x, str):
            try:
                d = json.loads(x)
                if not isinstance(d, dict):
                    d = {}
            except Exception:
                d = {}
        else:
            d = {}

        keycount.append(len(d))
        dicts.append(d)

    expanded = pd.DataFrame(dicts).fillna(0.0)

    # asegurar numérico
    for c in expanded.columns:
        expanded[c] = pd.to_numeric(expanded[c], errors="coerce").fillna(0.0)

    tmp = pd.concat([dfx[["user"]].reset_index(drop=True), expanded.reset_index(drop=True)], axis=1)
    tmp["keys_per_row"] = keycount
    tmp["row_sum"] = expanded.sum(axis=1) if expanded.shape[1] > 0 else 0.0

    out = (
        tmp.groupby("user", as_index=False)
           .agg(
               num_rows=("user", "size"),
               sum_all_keys=("row_sum", "sum"),
               mean_keys_per_row=("keys_per_row", "mean"),
           )
           .sort_values("user")
           .reset_index(drop=True)
    )

    out["sum_all_keys"] = out["sum_all_keys"].astype(float).round(6)
    out["mean_keys_per_row"] = out["mean_keys_per_row"].astype(float).round(6)

    output_data = out
    return input_data, output_data
