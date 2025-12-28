import pandas as pd

X_train, y_train, X_val, y_val = prepare_train_val(
    cfg["data"]["train_csv"],
    cfg["data"].get("validation_csv", ""),
    cfg["data"]["target"],
    stratify=cfg["data"].get("stratify", True),
    seed=cfg.get("seed", 42),
    header=cfg["data"].get("header", None),
    target_candidates=cfg["data"].get("target_candidates", []),
)

# (opcjonalnie) ograniczenie rozmiaru danych do szybkich testów
try:
    X_train, y_train, X_val, y_val = limit_dataset(X_train, X_val, y_train, y_val)
except Exception:
    pass



test_csv = cfg["data"].get("test_csv", "")
X_test, y_test = None, None
if test_csv:
    # test.csv: brak nagłówka, ostatnia kolumna = label
    df_test = pd.read_csv(test_csv, header=None)
    arr = df_test.to_numpy()

    if arr.shape[1] < 2:
        raise ValueError(f"test_csv={test_csv} ma za mało kolumn: {arr.shape[1]}")

    X_test = arr[:, :-1]
    y_test = arr[:, -1]

    log.info(
        f"Loaded external TEST set from {test_csv}: "
        f"{X_test.shape[0]} rows, {X_test.shape[1]} features, last col as label"
    )
else:
    log.warning("No external test_csv provided in config; external test will be skipped.")
