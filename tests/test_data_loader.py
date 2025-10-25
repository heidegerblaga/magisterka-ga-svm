import pandas as pd
from src.ga.data.loader import load_dataset, split_features_labels, prepare_train_val


def test_split_features_labels(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "label": [0, 1]})
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)

    X_train, y_train, X_val, y_val = prepare_train_val(str(path), "", "label", stratify=False)
    assert len(X_train) > 0 and len(y_train) > 0
    assert set(X_train.columns) == {"a", "b"}
    assert y_train.nunique() <= 2
