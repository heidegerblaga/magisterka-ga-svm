# src/ga/profiling.py
from dataclasses import dataclass


@dataclass
class SVMProfile:
    n_train_eval: int = 0      # ile razy wywołano train_eval_svm (uczenie + walidacja)
    n_fixed_eval: int = 0      # ile razy wywołano eval_fixed_svm (tylko predykcja na walidacji)
    val_points_evaluated: int = 0  # suma |X_val| ze wszystkich ewaluacji SVM

    def reset(self) -> None:
        self.n_train_eval = 0
        self.n_fixed_eval = 0
        self.val_points_evaluated = 0

    @property
    def approx_cost(self) -> int:
        """
        Przybliżony koszt: suma rozmiarów walidacji użytych w ewaluacjach.
        To jest de facto:
            ≈ (liczba ewaluacji) × (średni rozmiar walidacji)
        """
        return self.val_points_evaluated


_PROFILE = SVMProfile()


def get_profile() -> SVMProfile:
    """
    Zwraca globalny licznik. Nie komplikujemy przekazywaniem przez config,
    bo i tak wszystko chodzi w jednym procesie.
    """
    return _PROFILE
