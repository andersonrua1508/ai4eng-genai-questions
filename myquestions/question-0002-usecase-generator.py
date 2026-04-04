import random
import numpy as np
import pandas as pd

def generar_caso_de_uso_alinear_sensor_eventos():
    """
    Genera un caso de uso aleatorio (input/output esperado)
    para alinear_sensor_eventos(sensor_df, eventos_df, time_col, value_col, event_time_col, direction)
    """
    rng = np.random.default_rng()

    base = pd.Timestamp("2024-01-01") + pd.Timedelta(days=random.randint(0, 250))

    n_sensor = random.randint(80, 200)
    n_events = random.randint(20, 60)

    step = int(rng.integers(1, 6))
    sensor_times = [base + pd.Timedelta(minutes=step * i) for i in range(n_sensor)]
    sensor_vals = (rng.normal(loc=0.0, scale=1.0, size=n_sensor) + np.linspace(0, 2, n_sensor)).astype(float)
    sensor_df = pd.DataFrame({"ts": sensor_times, "val": sensor_vals})

    event_times = [base + pd.Timedelta(minutes=int(rng.integers(0, step * (n_sensor - 1)))) for _ in range(n_events)]
    eventos_df = pd.DataFrame({"event_ts": event_times})

    # meter invalidos
    sensor_df['ts'] = sensor_df['ts'].astype(object)
    for idx in rng.choice(sensor_df.index.to_numpy(), size=max(1, n_sensor // 30), replace=False):
        sensor_df.loc[idx, "ts"] = "fecha_mala"

    for idx in rng.choice(eventos_df.index.to_numpy(), size=max(1, n_events // 20), replace=False):
        eventos_df.loc[idx, "event_ts"] = None

    direction = random.choice(["backward", "forward", "nearest"])

    input_data = {
        "sensor_df": sensor_df.copy(),
        "eventos_df": eventos_df.copy(),
        "time_col": "ts",
        "value_col": "val",
        "event_time_col": "event_ts",
        "direction": direction,
    }

    # ---- Ground truth ----
    s = sensor_df.copy()
    e = eventos_df.copy()
    s["ts"] = pd.to_datetime(s["ts"], errors="coerce")
    e["event_ts"] = pd.to_datetime(e["event_ts"], errors="coerce")
    s = s.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    e = e.dropna(subset=["event_ts"]).sort_values("event_ts").reset_index(drop=True)

    merged = pd.merge_asof(
        e.rename(columns={"event_ts": "event_time"}),
        s.rename(columns={"ts": "sensor_time", "val": "sensor_value"}),
        left_on="event_time",
        right_on="sensor_time",
        direction=direction,
    )

    merged["lag_seconds"] = (merged["event_time"] - merged["sensor_time"]).dt.total_seconds()

    out = merged[["event_time", "sensor_time", "sensor_value", "lag_seconds"]].sort_values("event_time").reset_index(drop=True)
    out["sensor_value"] = out["sensor_value"].astype(float)
    out["lag_seconds"] = out["lag_seconds"].astype(float)

    out["sensor_value"] = out["sensor_value"].round(6)
    out["lag_seconds"] = out["lag_seconds"].round(6)

    output_data = out
    return input_data, output_data
