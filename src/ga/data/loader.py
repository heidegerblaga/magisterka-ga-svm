import csv
import pandas as pd
from sklearn.model_selection import train_test_split


def _detect_sep(path: str, default=","):
    try:
        with open(path, "r", encoding="utf-8") as f:
            sample = f.read(4096)
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        return default


def load_dataset(path: str, header: bool | None = None) -> pd.DataFrame:
    """
    Load CSV dataset. If header is None, try with header first; if it fails, fallback to header=None.
    """
    sep = _detect_sep(path)
    if header is True:
        return pd.read_csv(path, sep=sep, header=0)
    if header is False:
        return pd.read_csv(path, sep=sep, header=None)

    # header auto-try
    try:
        return pd.read_csv(path, sep=sep, header=0)
    except Exception:
        return pd.read_csv(path, sep=sep, header=None)


def _guess_target_name(df: pd.DataFrame) -> str | None:
    # proste heurystyki: typowe nazwy
    candidates = ["label", "target", "y", "class", "Label", "Target", "Class"]
    for c in candidates:
        if c in df.columns:
            return c
    # jeśli brak nazw (kolumny numeryczne 0..N-1), spróbuj ostatnią kolumnę,
    # ale tylko jeśli wygląda jak kategoria/binaria (niedużo unikalnych)
    last = df.columns[-1]
    nunq = df[last].nunique(dropna=True)
    if nunq <= max(50, int(0.05 * len(df))):  # etykiety zwykle mają mało klas
        return last
    return None


def split_features_labels(df: pd.DataFrame, target_col: str):
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Available: {list(df.columns)[:10]}...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def prepare_train_val(
    train_path: str,
    val_path: str,
    target_col: str,
    stratify: bool = True,
    seed: int = 42,
    header: bool | None = None,
    target_candidates: list[str] | None = None,
):
    """
    Load and prepare training and validation sets.
    - header: True (ma nagłówek), False (bez nagłówka), None (spróbuj wykryć)
    - target_candidates: lista alternatywnych nazw celu do sprawdzenia
    """
    df_train = load_dataset(train_path, header=header)

    # ustalenie targetu
    tgt = target_col
    if tgt not in df_train.columns:
        if target_candidates:
            for c in target_candidates:
                if c in df_train.columns:
                    tgt = c
                    break
        if tgt not in df_train.columns:
            g = _guess_target_name(df_train)
            if g is not None:
                tgt = g

    if tgt not in df_train.columns:
        raise ValueError(
            f"Nie mogę znaleźć kolumny celu. Podaj poprawną nazwę w configu "
            f"lub ustaw header/target_candidates. Kolumny: {df_train.columns.tolist()[:20]} ..."
        )

    if val_path:
        df_val = load_dataset(val_path, header=header)
        # upewnij się, że target jest w walidacji; jeśli nie – spróbuj zgadnąć tak samo
        if tgt not in df_val.columns:
            g2 = _guess_target_name(df_val)
            if g2:
                tgt = g2
            else:
                raise ValueError(f"Kolumna celu '{tgt}' nie występuje w walidacji.")
    else:
        df_train, df_val = train_test_split(
            df_train,
            test_size=0.2,
            random_state=seed,
            stratify=df_train[tgt] if stratify else None,
        )

    X_train, y_train = split_features_labels(df_train, tgt)
    X_val, y_val = split_features_labels(df_val, tgt)
    return X_train, y_train, X_val, y_val
