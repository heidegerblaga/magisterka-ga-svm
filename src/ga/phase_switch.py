"""DAGASVM z jawnym automatem fazowym T' ↔ V' (``algorithm_mode: dagasvm_phase_switch``).

Tryb jest NOWY i niezależny od ``ga.phase_engine`` (``dagasvm_phase``), który pozostaje bez zmian, aby badanie
followup_stop_rule było odtwarzalne. Rozdział ról zbiorów jest ten sam: sterują wyłącznie metryki liczone na V'
(i ΔV'); pełne V jest pulą kandydatów i źródłem OBSERWACJI, test — wyłącznie obserwacją.

AUTOMAT (jeden krok = jedno działanie; kolumna ``phase`` w cycles.csv)
---------------------------------------------------------------------
``optimize_T_prime`` — jedno uruchomienie GA dla T' przy stałym K na pełnym, skumulowanym V'. Po kroku:
  * BA(V') >= próg (T' opanowało V')       -> ``optimize_V_prime`` (K bez zmian), a gdy V' nie da się
                                              rozszerzyć (limit rozmiaru / brak kandydatów) -> STOP
                                              ``stop_perfect_on_max_vprime``;
  * BA(V') <  próg, brak zastoju           -> zwiększ K (co najwyżej raz) i kolejny krok T';
  * zastój T' (``max_no_improvement_k_increases`` kolejnych zwiększeń K bez poprawy najlepszego fitnessu
    o ``fitness_min_delta`` albo K nie może rosnąć)
                                           -> jednorazowo (na poziom K) ``optimize_V_prime`` jako „próba V'”,
                                              a gdy próba niedostępna -> STOP (``stop_tprime_no_progress``,
                                              ``stop_k_limit`` lub ``stop_no_k_growth_signal``);
  * po próbie V' (``retry``) T' dostaje szansę „także po zwiększeniu K”: dopóki poziom odniesienia nie jest osiągnięty,
    K rośnie (najwyżej raz na krok, najwyżej ``max_no_improvement_k_increases`` razy), a dopiero potem STOP
    ``stop_no_progress_after_vprime_retry`` (``retry_resolution: first_step`` = zachowanie v1: STOP już po 1. kroku).
``optimize_V_prime`` — jedno rozszerzenie V' o przyrost ΔV' (strategia doboru: ewolucyjna/heurystyczna/losowa)
  przy USTALONYM modelu T'. Oceniany jest wyłącznie ΔV': trudność d = błąd klasyfikacji ΔV' (``error_rate``, domyślnie)
  albo 1 − metryka_V'(ΔV') (``one_minus_metric``). Faza trwa, dopóki trudność (średnia z okna ostatnich przyrostów)
  rośnie co najmniej o ``vprime_difficulty_min_delta``; po ``vprime_difficulty_patience`` kolejnych przyrostach bez
  wzrostu, po osiągnięciu limitu |V'|, liczby rozszerzeń, braku kandydatów lub ``max_vprime_expansions_per_phase``
  -> powrót do ``optimize_T_prime`` (K bez zmian).

Niezmienniki kroku: K rośnie najwyżej raz, V' rozszerzane najwyżej raz, nigdy oba naraz; BA(V') >= próg nigdy
nie zwiększa K; pełne V i test nie są czytane przez logikę decyzyjną.

Porównanie „postępu po próbie V'” (odstępstwo od dosłownej specyfikacji, uzasadnione w AUDYT_v2.md): fitness po
próbie liczony jest na ROZSZERZONYM V', więc domyślnie porównujemy go z najlepszym T' sprzed próby ocenionym na TYM
SAMYM rozszerzonym V' (``retry_reference: same_vprime``); postęp = poprawa o co najmniej ``fitness_min_delta`` albo
osiągnięcie progu. ``retry_reference: literal`` porównuje z zapamiętanym fitnessem sprzed próby (na starym V', „co
najmniej”). Drugie porównanie jest zawsze zapisywane jako diagnostyka.

Koszt: każde wywołanie modelu w fazie V' i przy ocenach T' przechodzi przez ``CountingModel`` (liczniki wywołań
``predict`` / ``decision_function`` i wierszy). Ewaluacje wewnątrz GA dla T' (jedna ``predict`` na osobnika, patrz
``evaluate_individual_fixed_train``) są liczone z historii GA; zgodność liczników sprawdza test z podsłuchem ``SVC.predict``.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ga.encoding import KPerClassConfig, repair_to_k_per_class
from ga.fitness import FitnessConfig
from ga.loop.engine import EarlyStopping, build_ga_config, run_ga_k_per_class_with_trace
from ga.models.svm import SVMConfig, evaluate_fitted_svm, fit_svm_model
from ga.phase_engine import next_k_from_sv_ratio
from ga.utils.artifacts import save_mask, summarize_history, summarize_split, write_json
from ga.vprime import FullValidationObserver, VPrimeConfig, VPrimeSet, class_balanced_ids, expand_vprime

MODE = "dagasvm_phase_switch"
PHASE_T = "optimize_T_prime"
PHASE_V = "optimize_V_prime"
PHASE_STOP = "stop"

STOP_CATEGORY = {
    "stop_perfect_on_max_vprime": "perfect_on_max_vprime",
    "stop_no_progress_after_vprime_retry": "no_progress_after_retry",
    "stop_tprime_no_progress": "tprime_no_progress",
    "stop_k_limit": "k_limit",
    "stop_no_k_growth_signal": "no_k_growth_signal",
    "stop_max_cycles": "max_cycles",
    "stop_replay_end": "replay_end",
}
# metryki, które dobór V' (Val-GA liczy metrykę z predykcji) obsługuje bez cichego zastąpienia przez BA
VPRIME_SELECTION_METRICS = ("balanced_accuracy", "accuracy", "f1")
TPRIME_METRICS = ("balanced_accuracy", "accuracy", "f1", "f1_macro", "precision", "recall", "roc_auc", "pr_auc")
SCORE_METRICS = ("roc_auc", "pr_auc")  # wymagają svm.probability=true (inaczej evaluate_fitted_svm zwraca BA)
DIFFICULTY_METRICS = ("error_rate", "one_minus_metric")
RETRY_RESOLUTIONS = ("after_k_increases", "first_step")
RETRY_REFERENCES = ("same_vprime", "literal")

# Kolumny cycles.csv. „Cykl” = jeden krok automatu (jedno uruchomienie GA dla T' ALBO jedno rozszerzenie V').
PHASE_SWITCH_COLUMNS = [
    "cycle", "phase", "prev_phase", "vprime_phase_id", "phase_entry_reason", "retry_phase",
    "k_per_class", "k_after", "k_increased", "k_increases_total", "vprime_expanded", "vprime_expansions_total",
    "vprime_rows_before", "vprime_rows_after", "full_validation_rows", "vprime_cap_rows", "vprime_fraction",
    "vprime_limit_reason",
    # --- model oceniany na V' w tym kroku (T': nowy najlepszy T'; V': model ustalony na czas fazy) ---
    "tprime_metric", "model_metric_on_vprime",
    # --- krok T' ---
    "tprime_fitness", "tprime_perfect", "tprime_incumbent_fitness", "tprime_improved",
    "no_improvement_count", "sv_ratio", "sv_signal", "k_next_candidate",
    "ga_generations", "ga_svm_evaluations", "ga_warm_start", "ga_reached_target", "train_rows", "num_support_vectors",
    "retry_check", "retry_pending", "retry_fail_count", "retry_reference_raw", "retry_reference_same_vprime",
    "retry_progress", "retry_progress_raw",
    # --- krok V' (ocena wyłącznie przyrostu ΔV') ---
    "vprime_metric", "delta_rows", "delta_per_class", "delta_metric", "delta_error_rate", "delta_difficulty",
    "difficulty_change_vs_prev", "difficulty_window_mean", "difficulty_gain", "difficulty_no_gain_count",
    "difficulty_plateau", "expansions_in_phase",
    "delta_eval_rows", "selection_predict_rows", "selection_decision_rows", "val_ga_population_evaluations",
    "n_candidates", "n_scored_pool",
    # --- decyzja ---
    "decision", "next_phase", "transition_reason",
    # --- koszt kroku (wywołania modelu wykonane przez algorytm; obserwacja pełnego V i testu osobno) ---
    "svm_trainings", "predict_calls", "predict_rows", "decision_function_calls", "decision_function_rows",
    "tprime_eval_rows_nominal", "tprime_eval_rows_actual", "val_ga_nominal_rows", "observation_rows",
    # --- obserwacja (nie steruje) ---
    "observed_full_validation_metric", "observed_test_metric", "observed_test_accuracy",
    "cycle_duration_sec", "stop_decision", "stop_category",
]


# ======================================================================================
# konfiguracja
# ======================================================================================
@dataclass
class PhaseSwitchSettings:
    k_start: int = 10
    tsv: float = 0.25
    ga_patience: int = 3
    ga_min_delta: float = 0.0005
    perfect_threshold: float = 1.0
    tprime_metric: str = "balanced_accuracy"
    vprime_metric: str = "balanced_accuracy"
    # --- rozmiar V' ---
    initial_vprime_per_class: int = 10
    vprime_increment_per_class: int = 10
    max_vprime_fraction: float = 0.25
    max_vprime_rows: int | None = None
    max_vprime_expansions: int | None = None  # globalny limit liczby rozszerzeń V' w całym przebiegu
    # --- faza V' (trudność przyrostu) ---
    vprime_difficulty_metric: str = "error_rate"
    vprime_difficulty_min_delta: float = 0.01
    vprime_difficulty_patience: int = 2
    vprime_difficulty_window: int = 2
    vprime_min_difficulty: float = 0.0
    max_vprime_expansions_per_phase: int = 10
    # --- zastój T' ---
    max_no_improvement_k_increases: int = 3
    fitness_min_delta: float = 0.0005
    allow_one_vprime_retry_after_t_stall: bool = True
    retry_resolution: str = "after_k_increases"  # first_step = zachowanie v1 (STOP po 1. kroku T' bez postępu)
    retry_reference: str = "same_vprime"         # literal = zapamiętany fitness sprzed próby (na starym V')
    k_growth_requires_sv_signal: bool = True  # istniejąca polityka K: wzrost tylko przy ρ_SV >= TSV
    # --- pozostałe ---
    warm_start_tprime: bool = True
    ga_stop_at_perfect: bool = True
    representative: str = "best_on_final_vprime"
    max_cycles: int = 300
    vprime: VPrimeConfig = field(default_factory=VPrimeConfig)
    observe_full_validation: bool = True
    observe_test: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < float(self.max_vprime_fraction) <= 1.0:
            raise ValueError("phase_switch.max_vprime_fraction musi należeć do (0, 1].")
        if self.max_vprime_rows is not None and int(self.max_vprime_rows) <= 0:
            raise ValueError("phase_switch.max_vprime_rows musi być dodatnie (albo null).")
        if self.max_vprime_expansions is not None and int(self.max_vprime_expansions) < 1:
            raise ValueError("phase_switch.max_vprime_expansions musi być >= 1 (albo null).")
        if self.vprime_difficulty_metric not in DIFFICULTY_METRICS:
            raise ValueError(f"phase_switch.vprime_difficulty_metric={self.vprime_difficulty_metric!r}; "
                             f"obsługiwane: {DIFFICULTY_METRICS}.")
        if self.retry_resolution not in RETRY_RESOLUTIONS:
            raise ValueError(f"phase_switch.retry_resolution={self.retry_resolution!r}; obsługiwane: {RETRY_RESOLUTIONS}.")
        if self.retry_reference not in RETRY_REFERENCES:
            raise ValueError(f"phase_switch.retry_reference={self.retry_reference!r}; obsługiwane: {RETRY_REFERENCES}.")
        for name in ("vprime_increment_per_class", "initial_vprime_per_class", "vprime_difficulty_patience",
                     "vprime_difficulty_window", "max_vprime_expansions_per_phase", "max_no_improvement_k_increases",
                     "max_cycles"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"phase_switch.{name} musi być >= 1.")
        if self.tprime_metric not in TPRIME_METRICS:
            raise ValueError(f"fitness.tprime_metric={self.tprime_metric!r}; obsługiwane: {TPRIME_METRICS}.")
        if self.vprime_metric not in VPRIME_SELECTION_METRICS:
            raise ValueError(
                f"fitness.vprime_metric={self.vprime_metric!r} nie jest jeszcze obsługiwana przy doborze V' "
                f"(Val-GA zastąpiłby ją po cichu BA); obsługiwane: {VPRIME_SELECTION_METRICS}."
            )
        if self.vprime.mode != "growing" or not self.vprime.memory:
            raise ValueError("Tryb dagasvm_phase_switch wymaga vprime.mode=growing i vprime.memory=true (ΔV' = przyrost).")
        if self.representative not in ("best_on_final_vprime", "last_tprime_step"):
            raise ValueError("phase_switch.representative: best_on_final_vprime | last_tprime_step.")


def parse_phase_switch_settings(cfg: dict) -> PhaseSwitchSettings:
    daga = cfg.get("daga", {}) or {}
    ps = cfg.get("phase_switch", {}) or {}
    fit = cfg.get("fitness", {}) or {}
    vp = cfg.get("vprime", {}) or {}
    obs = cfg.get("observation", {}) or {}
    k_start = int(daga.get("k_start", cfg.get("encoding", {}).get("k_per_class", 10)))
    tprime_metric = str(fit.get("tprime_metric", fit.get("metric", "balanced_accuracy")))
    vprime = VPrimeConfig(
        selection=str(vp.get("selection", "evolutionary")),
        memory=bool(vp.get("memory", True)),
        mode=str(vp.get("mode", "growing")),
        # diagnostyka błędności CAŁEJ puli nie steruje niczym i w doborze losowym kosztowałaby osobne `predict` po
        # całej puli — domyślnie wyłączona, żeby koszt wariantu losowego i ewolucyjnego był porównywalny
        score_candidate_pool=bool(vp.get("score_candidate_pool", False)),
        pool_budget_factor=None if vp.get("pool_budget_factor") is None else float(vp["pool_budget_factor"]),
    )
    defaults = PhaseSwitchSettings.__dataclass_fields__

    def g(name):
        return ps.get(name, defaults[name].default)

    return PhaseSwitchSettings(
        k_start=k_start,
        tsv=float(daga.get("TSV", daga.get("tsv", 0.25))),
        ga_patience=int(daga.get("Ts", 3)),
        ga_min_delta=float(daga.get("phase_min_delta", 0.0005)),
        perfect_threshold=float(ps.get("perfect_threshold", daga.get("perfect_val_threshold", 1.0))),
        tprime_metric=tprime_metric,
        vprime_metric=str(fit.get("vprime_metric", tprime_metric)),
        initial_vprime_per_class=int(ps.get("initial_vprime_per_class") or k_start),
        vprime_increment_per_class=int(g("vprime_increment_per_class")),
        max_vprime_fraction=float(g("max_vprime_fraction")),
        max_vprime_rows=None if ps.get("max_vprime_rows") is None else int(ps["max_vprime_rows"]),
        max_vprime_expansions=None if ps.get("max_vprime_expansions") is None else int(ps["max_vprime_expansions"]),
        vprime_difficulty_metric=str(g("vprime_difficulty_metric")),
        vprime_difficulty_min_delta=float(g("vprime_difficulty_min_delta")),
        vprime_difficulty_patience=int(g("vprime_difficulty_patience")),
        vprime_difficulty_window=int(g("vprime_difficulty_window")),
        vprime_min_difficulty=float(g("vprime_min_difficulty")),
        max_vprime_expansions_per_phase=int(g("max_vprime_expansions_per_phase")),
        max_no_improvement_k_increases=int(g("max_no_improvement_k_increases")),
        fitness_min_delta=float(g("fitness_min_delta")),
        allow_one_vprime_retry_after_t_stall=bool(g("allow_one_vprime_retry_after_t_stall")),
        retry_resolution=str(g("retry_resolution")),
        retry_reference=str(g("retry_reference")),
        k_growth_requires_sv_signal=bool(g("k_growth_requires_sv_signal")),
        warm_start_tprime=bool(g("warm_start_tprime")),
        ga_stop_at_perfect=bool(g("ga_stop_at_perfect")),
        representative=str(g("representative")),
        max_cycles=int(g("max_cycles")),
        vprime=vprime,
        observe_full_validation=bool(obs.get("full_validation", True)),
        observe_test=bool(obs.get("test", True)),
    )


# ======================================================================================
# rozmiar V'
# ======================================================================================
def vprime_cap_rows(n_val_full: int, max_fraction: float, max_rows: int | None) -> int:
    """Maksymalny dopuszczalny |V'|: min(⌊ułamek·|V|⌋, max_rows, |V|)."""
    cap = int(np.floor(float(max_fraction) * int(n_val_full)))
    if max_rows is not None:
        cap = min(cap, int(max_rows))
    return max(0, min(cap, int(n_val_full)))


def expansion_block_reason(*, vprime_rows: int, cap_rows: int, n_candidates: int, n_classes: int,
                           expansions_left: int | None = None) -> str | None:
    """Dlaczego V' nie może zostać rozszerzone (None = może). Zależy wyłącznie od rozmiarów i liczników."""
    if expansions_left is not None and int(expansions_left) <= 0:
        return "expansion_limit"
    if int(cap_rows) - int(vprime_rows) < int(n_classes):
        return "size_limit"
    if int(n_candidates) < int(n_classes):
        return "no_candidates"
    return None


def plan_increment_per_class(*, vprime_rows: int, cap_rows: int, n_candidates: int, n_classes: int,
                             increment_per_class: int, expansions_left: int | None = None) -> int:
    """Liczba próbek/klasę w kolejnym przyrostcie V' (0 = V' nie da się już rozszerzyć).

    Przyrost nie przekracza limitu |V'|, liczby kandydatów ani limitu liczby rozszerzeń; zawsze jest wielokrotnością
    liczby klas. Nie zależy od K ani od ρ_SV."""
    if expansion_block_reason(vprime_rows=vprime_rows, cap_rows=cap_rows, n_candidates=n_candidates,
                              n_classes=n_classes, expansions_left=expansions_left) is not None:
        return 0
    remaining = min(int(cap_rows) - int(vprime_rows), int(n_candidates))
    return int(min(int(increment_per_class), remaining // int(n_classes)))


# ======================================================================================
# licznik wywołań modelu
# ======================================================================================
class CountingModel:
    """Osłona na wytrenowany model: zlicza wywołania ``predict`` / ``decision_function`` i przetworzone wiersze.

    Pozostałe atrybuty (``predict_proba``, ``n_support_`` …) są delegowane do modelu, więc ``hasattr`` i ``try`` w kodzie
    doboru V' zachowują się tak samo jak dla modelu bez osłony."""

    def __init__(self, clf):
        self._clf = clf
        self.predict_calls = 0
        self.predict_rows = 0
        self.decision_function_calls = 0
        self.decision_function_rows = 0

    def predict(self, X):
        self.predict_calls += 1
        self.predict_rows += int(len(X))
        return self._clf.predict(X)

    def decision_function(self, X):
        self.decision_function_calls += 1
        self.decision_function_rows += int(len(X))
        return self._clf.decision_function(X)

    def __getattr__(self, name):
        return getattr(self._clf, name)

    def snapshot(self) -> dict:
        return {"predict_calls": self.predict_calls, "predict_rows": self.predict_rows,
                "decision_function_calls": self.decision_function_calls,
                "decision_function_rows": self.decision_function_rows}


# ======================================================================================
# trudność przyrostu ΔV'
# ======================================================================================
@dataclass
class DifficultyUpdate:
    difficulty: float
    change_vs_prev: float | None
    window_mean: float
    gain: float | None
    no_gain_count: int
    plateau: bool


class DifficultyTracker:
    """Śledzi trudność kolejnych przyrostów ΔV' w JEDNEJ fazie V' (model T' ustalony w fazie).

    d_i = trudność i-tego przyrostu; D_i = średnia z ostatnich ``window`` przyrostów; zysk g_i = D_i − D_{i−1}.
    Przyrost „bez wzrostu trudności”: g_i < ``min_delta`` albo d_i <= ``min_difficulty`` (np. zero błędów modelu
    na ΔV'). Plateau = ``patience`` kolejnych przyrostów bez wzrostu."""

    def __init__(self, *, min_delta: float, patience: int, window: int, min_difficulty: float = 0.0):
        self.min_delta = float(min_delta)
        self.patience = int(patience)
        self.window = int(window)
        self.min_difficulty = float(min_difficulty)
        self.history: list[float] = []
        self.means: list[float] = []
        self.no_gain = 0

    def update(self, difficulty: float) -> DifficultyUpdate:
        d = float(difficulty)
        prev = self.history[-1] if self.history else None
        self.history.append(d)
        mean = float(np.mean(self.history[-self.window:]))
        gain = None if not self.means else mean - self.means[-1]
        self.means.append(mean)
        no_gain_step = d <= self.min_difficulty or (gain is not None and gain < self.min_delta)
        self.no_gain = self.no_gain + 1 if no_gain_step else 0
        return DifficultyUpdate(
            difficulty=d,
            change_vs_prev=None if prev is None else d - prev,
            window_mean=mean,
            gain=gain,
            no_gain_count=self.no_gain,
            plateau=self.no_gain >= self.patience,
        )


# ======================================================================================
# czyste funkcje decyzyjne automatu
# ======================================================================================
@dataclass
class TDecision:
    next_phase: str
    increase_k: bool = False
    retry: bool = False
    stop_code: str | None = None
    reason: str = ""


def decide_after_tprime(
    *,
    perfect: bool,
    vprime_expandable: bool,
    no_improvement_count: int,
    max_no_improvement: int,
    k_can_grow: bool,
    sv_signal: bool,
    k_growth_requires_sv_signal: bool,
    allow_retry: bool,
    retry_used_at_current_k: bool,
    retry_check: bool = False,
    retry_progress: bool = True,
    retry_resolution: str = "after_k_increases",
) -> TDecision:
    """Decyzja po kroku T'. Argumenty zależą wyłącznie od V', T', K i puli kandydatów (nie od pełnego V/testu).

    ``retry_check`` = to jest krok T' po próbie V' (zastój); ``retry_progress`` = T' osiągnęło poziom odniesienia.
    W tym przypadku ``no_improvement_count`` oznacza liczbę zwiększeń K od próby V', które poziomu nie osiągnęły."""
    if retry_check and not retry_progress and not perfect:
        k_blocked = (not k_can_grow) or (k_growth_requires_sv_signal and not sv_signal)
        if retry_resolution == "first_step" or k_blocked or no_improvement_count >= max_no_improvement:
            why = ("zwiększenie K jest niemożliwe (limit klasy albo ρ_SV < TSV)" if k_blocked and
                   retry_resolution != "first_step" else
                   f"{no_improvement_count} zwiększeń K od próby V' bez osiągnięcia poziomu odniesienia"
                   if retry_resolution != "first_step" else "pierwszy krok T' po próbie V' bez postępu")
            return TDecision(PHASE_STOP, stop_code="stop_no_progress_after_vprime_retry",
                             reason=f"po dodatkowej próbie V' T' nie osiągnęło poziomu odniesienia: {why}")
        return TDecision(PHASE_T, increase_k=True,
                         reason="po próbie V' T' nie osiągnęło poziomu odniesienia — zwiększam K (raz) i ponawiam T'")
    if perfect:
        if vprime_expandable:
            return TDecision(PHASE_V, reason="T' opanowało V' (metryka >= próg) — utrudniam V'; K bez zmian")
        return TDecision(PHASE_STOP, stop_code="stop_perfect_on_max_vprime",
                         reason="T' opanowało V', a V' nie da się już rozszerzyć (limit |V'| lub brak kandydatów)")
    stalled_no_progress = no_improvement_count >= max_no_improvement
    blocked_by_signal = k_growth_requires_sv_signal and not sv_signal
    if not (stalled_no_progress or not k_can_grow or blocked_by_signal):
        return TDecision(PHASE_T, increase_k=True,
                         reason="T' nie opanowało V' — zwiększam K (raz) i ponownie optymalizuję T'")
    if stalled_no_progress:
        cause, code = f"{no_improvement_count} kolejnych zwiększeń K bez poprawy fitnessu", "stop_tprime_no_progress"
    elif not k_can_grow:
        cause, code = "K nie może dalej rosnąć (limit klasy)", "stop_k_limit"
    else:
        cause, code = "brak sygnału wzrostu K (ρ_SV < TSV)", "stop_no_k_growth_signal"
    if allow_retry and not retry_used_at_current_k and vprime_expandable:
        return TDecision(PHASE_V, retry=True, reason=f"zastój T' ({cause}) — jednorazowa próba rozszerzenia V'")
    why = ("próba V' przy tym K już wykonana" if retry_used_at_current_k
           else "próby V' wyłączone" if not allow_retry else "V' nie da się rozszerzyć")
    return TDecision(PHASE_STOP, stop_code=code, reason=f"zastój T' ({cause}); {why}")


def decide_after_vprime(*, n_new: int, plateau: bool, can_expand_more: bool, expansions_in_phase: int,
                        max_expansions_per_phase: int) -> tuple[str, str]:
    """Decyzja po kroku V': (następna faza, powód)."""
    if n_new <= 0:
        return PHASE_T, "no_new_elements"
    if plateau:
        return PHASE_T, "difficulty_plateau"
    if not can_expand_more:
        return PHASE_T, "vprime_limit_or_no_candidates"
    if expansions_in_phase >= max_expansions_per_phase:
        return PHASE_T, "max_expansions_per_phase"
    return PHASE_V, "difficulty_rising"


# ======================================================================================
# kontroler (stan automatu) — bez SVM, testowalny skryptowanymi wartościami
# ======================================================================================
class PhaseSwitchController:
    def __init__(self, st: PhaseSwitchSettings, k_start: int):
        self.st = st
        self.phase = PHASE_T
        self.k = int(k_start)
        self.incumbent: float | None = None   # najlepszy fitness T' na BIEŻĄCYM V'
        self.no_improve = 0
        self.retry_used_k: set[int] = set()
        self.retry: dict | None = None        # próba V' w toku: {"k", "reference_raw"}
        self.vprime_phase_id = 0
        self.phase_entry_reason = "start"
        self.retry_phase = False
        self.tracker: DifficultyTracker | None = None
        self.expansions_in_phase = 0
        self.k_increases = 0

    # ---------------- krok T' ----------------
    def retry_needs_reference(self) -> bool:
        """Czy kolejny krok T' jest PIERWSZYM po próbie V' (wtedy trzeba ocenić T' sprzed próby na nowym V')."""
        return self.retry is not None and not self.retry["checked"]

    def on_tprime(self, *, fitness: float, k_next: int, sv_signal: bool, vprime_expandable: bool,
                  retry_reference_same: float | None = None) -> dict:
        assert self.phase == PHASE_T, "krok T' poza fazą optimize_T_prime"
        st = self.st
        f = float(fitness)
        perfect = f >= st.perfect_threshold
        info = {"retry_check": False, "retry_reference_raw": None, "retry_reference_same_vprime": None,
                "retry_progress": None, "retry_progress_raw": None, "retry_first_check": False}
        retry_progress = True
        retry_fail = 0
        if self.retry is not None:
            first_check = not self.retry["checked"]
            if first_check:
                # najlepsze T' sprzed próby, ocenione na ROZSZERZONYM V', jest punktem odniesienia także dla licznika zastoju
                self.retry["ref_same"] = float(retry_reference_same)
                self.incumbent, self.no_improve = self.retry["ref_same"], 0
            ref_same = float(self.retry["ref_same"])
            ref_raw = float(self.retry["reference_raw"])
            if st.retry_reference == "same_vprime":
                retry_progress = perfect or f >= ref_same + st.fitness_min_delta
            else:
                retry_progress = perfect or f >= ref_raw
            info.update(retry_check=True, retry_first_check=first_check, retry_reference_raw=ref_raw,
                        retry_reference_same_vprime=ref_same, retry_progress=bool(retry_progress),
                        retry_progress_raw=bool(perfect or f >= ref_raw + st.fitness_min_delta))
            if retry_progress:
                self.retry = None
                improved = True
                self.incumbent, self.no_improve = f, 0
            else:
                self.retry["checked"] = True
                if not first_check:
                    self.retry["fail"] += 1  # to był krok po zwiększeniu K, a poziom odniesienia nadal nieosiągnięty
                retry_fail = self.retry["fail"]
                improved = f > self.incumbent + st.fitness_min_delta
                if improved:
                    self.incumbent = f
        elif self.incumbent is None:  # pierwszy krok T' na nowym V' — punkt odniesienia
            improved = None
            self.incumbent, self.no_improve = f, 0
        else:
            improved = f > self.incumbent + st.fitness_min_delta
            if improved:
                self.incumbent, self.no_improve = f, 0
            else:
                self.no_improve += 1
        retry_pending_now = self.retry is not None  # po kroku: próba V' trwa (poziom odniesienia nieosiągnięty)
        stall_count = retry_fail if retry_pending_now else self.no_improve
        decision = decide_after_tprime(
            perfect=perfect,
            vprime_expandable=vprime_expandable,
            no_improvement_count=stall_count,
            max_no_improvement=st.max_no_improvement_k_increases,
            k_can_grow=int(k_next) > self.k,
            sv_signal=sv_signal,
            k_growth_requires_sv_signal=st.k_growth_requires_sv_signal,
            allow_retry=st.allow_one_vprime_retry_after_t_stall,
            retry_used_at_current_k=self.k in self.retry_used_k,
            retry_check=info["retry_check"],
            retry_progress=retry_progress,
            retry_resolution=st.retry_resolution,
        )
        k_before = self.k
        incumbent_after_step = self.incumbent
        if decision.increase_k:
            assert not perfect, "K nie może rosnąć, gdy T' opanowało V'"
            self.k = int(k_next)
            self.k_increases += 1
        elif decision.next_phase == PHASE_V:
            if decision.retry:
                self.retry_used_k.add(self.k)
                self.retry = {"k": self.k, "reference_raw": float(self.incumbent), "checked": False, "fail": 0,
                              "ref_same": None}
            self.vprime_phase_id += 1
            self.phase_entry_reason = "tprime_stall_retry" if decision.retry else "tprime_mastered_vprime"
            self.retry_phase = bool(decision.retry)
            self.tracker = DifficultyTracker(
                min_delta=st.vprime_difficulty_min_delta, patience=st.vprime_difficulty_patience,
                window=st.vprime_difficulty_window, min_difficulty=st.vprime_min_difficulty,
            )
            self.expansions_in_phase = 0
            self.incumbent, self.no_improve = None, 0  # V' się zmieni: fitness przestaje być porównywalny
        self.phase = decision.next_phase
        return {
            "decision": decision,
            "perfect": perfect,
            "improved": improved,
            "no_improvement_count": 0 if decision.next_phase == PHASE_V else stall_count,
            "retry_pending": self.retry is not None,
            "retry_fail_count": 0 if self.retry is None else int(self.retry["fail"]),
            "incumbent_fitness": incumbent_after_step,
            "k_before": k_before,
            "k_after": self.k,
            **info,
        }

    # ---------------- krok V' ----------------
    def on_vprime(self, *, difficulty: float, n_new: int, can_expand_more: bool) -> dict:
        assert self.phase == PHASE_V and self.tracker is not None
        self.expansions_in_phase += 1
        upd = self.tracker.update(difficulty)
        nxt, reason = decide_after_vprime(
            n_new=n_new, plateau=upd.plateau, can_expand_more=can_expand_more,
            expansions_in_phase=self.expansions_in_phase,
            max_expansions_per_phase=self.st.max_vprime_expansions_per_phase,
        )
        if nxt == PHASE_T:
            self.phase = PHASE_T
            self.phase_entry_reason = f"vprime_phase_end:{reason}"
        return {"update": upd, "next_phase": nxt, "reason": reason, "expansions_in_phase": self.expansions_in_phase}


# ======================================================================================
# pętla
# ======================================================================================
def _per_class_counts(y: np.ndarray, classes: np.ndarray) -> dict:
    return {str(c): int((y == c).sum()) for c in classes}


def _metric(clf, X, y, metric: str) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # BA na ΔV' z jedną klasą: ostrzeżenie sklearn, wynik poprawny (recall klasy)
        return evaluate_fitted_svm(clf, X, y, metric=metric)


def run_phase_switch_experiment(
    X_train, y_train, X_val_full, y_val_full, X_test, y_test,
    svm_cfg: SVMConfig, cfg: dict, log, artifact_layout,
    *, observer: FullValidationObserver | None = None, ga_runner=run_ga_k_per_class_with_trace,
) -> dict:
    st = parse_phase_switch_settings(cfg)
    if st.tprime_metric in SCORE_METRICS and not svm_cfg.probability:
        raise ValueError(f"fitness.tprime_metric={st.tprime_metric} wymaga svm.probability=true "
                         "(w przeciwnym razie evaluate_fitted_svm po cichu zwraca BA).")
    seed = int(cfg.get("seed", 42))
    vcfg = st.vprime
    X_train, y_train = np.asarray(X_train, dtype=float), np.asarray(y_train)
    X_val_full, y_val_full = np.asarray(X_val_full, dtype=float), np.asarray(y_val_full)
    X_test = None if X_test is None else np.asarray(X_test, dtype=float)
    y_test = None if y_test is None else np.asarray(y_test)
    if observer is None:
        observer = FullValidationObserver(X_val_full, y_val_full, st.tprime_metric, evaluate_fitted_svm,
                                          enabled=st.observe_full_validation)
    val_ga_gcfg = build_ga_config(cfg.get("val_ga", {})) if vcfg.selection == "evolutionary" else None
    fcfg = FitnessConfig(metric=st.tprime_metric, class_imbalance_penalty=0.0)

    classes = np.unique(y_train)
    n_classes = int(classes.size)
    if n_classes > 2 and st.tprime_metric == "pr_auc":
        raise ValueError("fitness.tprime_metric=pr_auc nie jest zdefiniowana dla >2 klas "
                         "(compute_prediction_metrics podstawiłaby po cichu BA).")
    if n_classes > 2 and st.vprime_metric == "f1":
        raise ValueError("fitness.vprime_metric=f1 w Val-GA jest binarne (average='binary'); dla >2 klas użyj "
                         "balanced_accuracy albo accuracy.")
    train_class_cap = int(np.unique(y_train, return_counts=True)[1].min())
    n_val = int(X_val_full.shape[0])
    cap_rows = vprime_cap_rows(n_val, st.max_vprime_fraction, st.max_vprime_rows)
    metrics_dir, checkpoints_dir = Path(artifact_layout["metrics"]), Path(artifact_layout["checkpoints"])

    vset = VPrimeSet(n_val)
    vset.replace(class_balanced_ids(y_val_full, st.initial_vprime_per_class, np.random.default_rng(seed)))
    if vset.size > cap_rows:
        log.warning("Startowe V' (%s) przekracza limit |V'| (%s) — V' nie będzie rozszerzane.", vset.size, cap_rows)
    log.info("[phase_switch] V' start: %s/%s (limit |V'|=%s; ułamek=%s, max_rows=%s) | selection=%s | metryki T'=%s V'=%s",
             vset.size, n_val, cap_rows, st.max_vprime_fraction, st.max_vprime_rows, vcfg.selection,
             st.tprime_metric, st.vprime_metric)

    ctl = PhaseSwitchController(st, st.k_start)
    replay = load_schedule_replay(cfg)
    if replay is not None:
        # wariant „ten sam harmonogram”: fazy, K i rozmiary ΔV' z przebiegu referencyjnego; inny jest tylko dobór ΔV'
        assert replay[0]["vprime_rows_before"] == vset.size, "replay: inne V' startowe niż w przebiegu referencyjnym"
        assert replay[0]["full_validation_rows"] == n_val, "replay: inne pełne V niż w przebiegu referencyjnym"
        log.info("[phase_switch] REPLAY harmonogramu z %s (%s kroków)", cfg["phase_switch"]["schedule_replay"], len(replay))
    replay_tracker: DifficultyTracker | None = None
    rows: list[dict] = []
    counter = {"model_predictions": 0, "diagnostic_predictions": 0}
    cost = {"delta_eval_rows": 0, "retry_reference_rows": 0, "train_ga_svm_evaluations": 0, "train_ga_fit_rows": 0,
            "train_ga_vprime_rows_scored": 0}
    increments: dict[str, np.ndarray] = {}
    last_t: dict | None = None        # ostatni krok T' (model, maska, metryki)
    best_t: dict | None = None        # najlepszy T' na BIEŻĄCYM V' (incumbent)
    phase_model: dict | None = None   # model ustalony na czas fazy V'
    retry_incumbent: dict | None = None
    final_stop = None
    cycle = 0
    n_expansions_total = 0            # rozszerzenia V' w całym przebiegu (limit: max_vprime_expansions)
    k_increases_total = 0

    def expansions_left() -> int | None:
        return None if st.max_vprime_expansions is None else st.max_vprime_expansions - n_expansions_total

    def plan_per_class() -> int:
        return plan_increment_per_class(vprime_rows=vset.size, cap_rows=cap_rows, n_candidates=vset.candidates().size,
                                        n_classes=n_classes, increment_per_class=st.vprime_increment_per_class,
                                        expansions_left=expansions_left())

    while True:
        cycle += 1
        t0 = time.perf_counter()
        ref = replay[cycle - 1] if replay is not None else None
        phase = ref["phase"] if ref else ctl.phase
        vp_ids = vset.ids
        X_vp, y_vp = X_val_full[vp_ids], y_val_full[vp_ids]
        meters: list[CountingModel] = []   # osłony modeli użyte w tym kroku (koszt wywołań)
        obs_rows_before = int(observer.rows_scored)
        row = {c: None for c in PHASE_SWITCH_COLUMNS}
        row.update(cycle=cycle, phase=phase, prev_phase=rows[-1]["phase"] if rows else None,
                   vprime_phase_id=ref["vprime_phase_id"] if ref else ctl.vprime_phase_id,
                   phase_entry_reason=ref["phase_entry_reason"] if ref else ctl.phase_entry_reason,
                   retry_phase=(bool(ref["retry_phase"]) if ref else bool(ctl.retry_phase)) if phase == PHASE_V else False,
                   k_per_class=ref["k_per_class"] if ref else ctl.k, vprime_rows_before=int(vset.size),
                   full_validation_rows=n_val,
                   vprime_cap_rows=cap_rows, tprime_metric=st.tprime_metric, vprime_metric=st.vprime_metric,
                   k_increased=False, vprime_expanded=False)

        if phase == PHASE_T:
            k = ref["k_per_class"] if ref else ctl.k
            kcfg = KPerClassConfig(k_per_class=int(k), random_state=seed + cycle - 1)
            gcfg = build_ga_config(cfg.get("ga", {}))
            gcfg.early_stopping = EarlyStopping(patience=max(1, st.ga_patience), min_delta=st.ga_min_delta)
            seeds = []
            if st.warm_start_tprime:
                wrng = np.random.default_rng(np.random.SeedSequence([seed, cycle, 404]))
                for src in (best_t, last_t, retry_incumbent):
                    if src is None:
                        continue
                    cand = repair_to_k_per_class(src["mask"], y_train, int(k), wrng)
                    if not any(np.array_equal(cand, s0) for s0 in seeds):
                        seeds.append(cand)
            history, best_mask, trace, _ = ga_runner(
                X_pool=X_train, y_pool=y_train, X_val=X_vp, y_val=y_vp, kcfg=kcfg, svm_cfg=svm_cfg, fcfg=fcfg,
                gcfg=gcfg, seed=seed + cycle - 1, seed_masks=seeds or None,
                target_fitness=st.perfect_threshold if st.ga_stop_at_perfect else None,
            )
            hs = summarize_history(history, gcfg.population_size, evaluation_label="svm_evaluations")
            mask_path = checkpoints_dir / f"step_{cycle:03d}_train_mask.npy"
            save_mask(mask_path, best_mask)
            clf, fit_meta = fit_svm_model(X_train[best_mask], y_train[best_mask], svm_cfg, return_metadata=True)
            clf_counted = CountingModel(clf)
            meters.append(clf_counted)
            ev = _metric(clf_counted, X_vp, y_vp, st.tprime_metric)
            f = float(ev["metric"])
            n_tr = int(best_mask.sum())
            sv_ratio = float(fit_meta["num_support_vectors"] / n_tr) if n_tr else None
            sv_signal = sv_ratio is not None and sv_ratio >= st.tsv
            k_next = min(next_k_from_sv_ratio(int(k), sv_ratio, st.tsv), train_class_cap)
            per_class = plan_per_class()
            ref_same, ref_eval = None, None
            if ref is None and ctl.retry_needs_reference():
                # najlepsze T' sprzed próby V' ocenione na rozszerzonym V' (jednorazowo, przy pierwszym kroku po próbie)
                ref_counted = CountingModel(retry_incumbent["clf"])
                meters.append(ref_counted)
                ref_eval = _metric(ref_counted, X_vp, y_vp, st.tprime_metric)
                ref_same = float(ref_eval["metric"])
                cost["retry_reference_rows"] += int(vp_ids.size)
            if ref is None:
                res = ctl.on_tprime(fitness=f, k_next=k_next, sv_signal=sv_signal, vprime_expandable=per_class > 0,
                                    retry_reference_same=ref_same)
            else:
                res = _replay_tprime(ref, replay, cycle, f, best_t, st)
            dec: TDecision = res["decision"]
            # obserwacja (po decyzji; nie steruje)
            full_obs = observer.observe(clf)
            test_obs = _metric(clf, X_test, y_test, st.tprime_metric) if (X_test is not None and st.observe_test) else None
            cur = {"cycle": cycle, "k": int(k), "mask": best_mask.copy(), "mask_path": str(mask_path), "clf": clf,
                   "fitness": f, "vprime_rows": int(vp_ids.size), "full": full_obs, "test": test_obs,
                   "train_summary": summarize_split(X_train[best_mask], y_train[best_mask]), "metrics_val": ev}
            if res["improved"] is None or res["improved"] or best_t is None:
                best_t = cur
            last_t = cur
            if res["retry_check"] and res["retry_first_check"] and ref_same > f:
                # na rozszerzonym V' lepszy jest T' sprzed próby (np. bez warm startu) — pozostaje najlepszym
                best_t = {**retry_incumbent, "fitness": ref_same, "vprime_rows": int(vp_ids.size),
                          "metrics_val": ref_eval}
            if dec.next_phase == PHASE_V:
                phase_model = cur if res["perfect"] else best_t
                retry_incumbent = best_t if dec.retry else None
                best_t = None  # V' się zmieni
            elif res["retry_check"] and not res["retry_pending"]:
                retry_incumbent = None  # próba V' rozstrzygnięta pozytywnie; do tego czasu T' sprzed próby jest ziarnem warm startu
            cost["train_ga_svm_evaluations"] += int(hs["svm_evaluations"])
            cost["train_ga_fit_rows"] += int(hs["svm_evaluations"]) * n_tr
            cost["train_ga_vprime_rows_scored"] += int(hs["svm_evaluations"]) * int(vp_ids.size)
            # koszt: ewaluacje wewnątrz GA = po jednym `predict` na osobnika (na całym V'), + osłonięte oceny bezpośrednie
            row.update(
                svm_trainings=int(hs["svm_evaluations"]) + 1,
                tprime_eval_rows_actual=int(hs["svm_evaluations"]) * int(vp_ids.size),
                tprime_eval_rows_nominal=int(gcfg.population_size) * (int(gcfg.generations) + 1) * int(vp_ids.size),
                model_metric_on_vprime=f,
            )
            row.update(
                k_after=ref["k_after"] if ref else ctl.k, k_increased=bool(dec.increase_k), tprime_fitness=f, tprime_perfect=res["perfect"],
                tprime_incumbent_fitness=res["incumbent_fitness"], tprime_improved=res["improved"],
                no_improvement_count=res["no_improvement_count"], sv_ratio=sv_ratio, sv_signal=bool(sv_signal),
                k_next_candidate=int(k_next), ga_generations=hs["generations_executed"],
                ga_svm_evaluations=hs["svm_evaluations"], ga_warm_start=len(seeds),
                ga_reached_target=bool(st.ga_stop_at_perfect and max(history) >= st.perfect_threshold),
                train_rows=n_tr, num_support_vectors=int(fit_meta["num_support_vectors"]),
                retry_check=res["retry_check"], retry_pending=res["retry_pending"], retry_fail_count=res["retry_fail_count"],
                retry_reference_raw=res["retry_reference_raw"],
                retry_reference_same_vprime=res["retry_reference_same_vprime"], retry_progress=res["retry_progress"],
                retry_progress_raw=res["retry_progress_raw"],
                decision=("increase_k" if dec.increase_k else "switch_to_V_prime" if dec.next_phase == PHASE_V
                          else dec.stop_code if dec.next_phase == PHASE_STOP else "continue_T_prime"),
                next_phase=dec.next_phase, transition_reason=dec.reason,
                observed_full_validation_metric=None if full_obs is None else float(full_obs["metric"]),
                observed_test_metric=None if test_obs is None else float(test_obs["metric"]),
                observed_test_accuracy=None if test_obs is None else test_obs["all"].get("accuracy"),
            )
            log.info("[krok %s | T'] K=%s |V'|=%s %s(V')=%.4f incumbent=%s brak_poprawy=%s ρ_SV=%s -> %s (%s)%s",
                     cycle, k, vp_ids.size, st.tprime_metric, f, res["incumbent_fitness"], res["no_improvement_count"],
                     None if sv_ratio is None else round(sv_ratio, 4), row["decision"], dec.reason,
                     "" if test_obs is None else f" | obserwacja test={test_obs['metric']:.4f}")
            if dec.next_phase == PHASE_STOP:
                final_stop = dec.stop_code
        else:  # PHASE_V
            assert phase_model is not None
            clf = CountingModel(phase_model["clf"])
            meters.append(clf)
            if ref is None:
                per_class = plan_per_class()
                n_add = per_class * n_classes
            else:
                n_add = int(ref["delta_rows"])
                per_class = int(np.ceil(n_add / n_classes))
            assert per_class > 0, "faza V' rozpoczęta bez możliwości rozszerzenia"
            # dobór ΔV' WYŁĄCZNIE z puli V \ V' (expand_vprime widzi tylko kandydatów); stare próbki V' nie są oceniane
            sel_rng = np.random.default_rng(np.random.SeedSequence([seed, cycle, 101]))
            exp = expand_vprime(
                vset, memory=True, strategy=vcfg.selection, n_add=n_add, per_class_add=per_class,
                n_classes=n_classes, X_val=X_val_full, y_val=y_val_full, clf=clf, metric=st.vprime_metric,
                val_ga_cfg=val_ga_gcfg, seed=seed + cycle - 1, rng=sel_rng, counter=counter,
                score_candidate_pool=vcfg.score_candidate_pool, pool_budget_factor=vcfg.pool_budget_factor,
            )
            delta = exp.selected_ids
            increments[f"step_{cycle:03d}"] = delta
            if delta.size:
                dev = _metric(clf, X_val_full[delta], y_val_full[delta], st.vprime_metric)
                delta_metric = float(dev["metric"])
                delta_err = float(1.0 - dev["all"]["accuracy"])
                cost["delta_eval_rows"] += int(delta.size)
            else:
                delta_metric, delta_err = None, None
            if delta_metric is None:
                difficulty = 0.0
            elif st.vprime_difficulty_metric == "error_rate":
                difficulty = delta_err
            else:
                difficulty = 1.0 - delta_metric
            if exp.n_new > 0:
                n_expansions_total += 1
            next_per_class = plan_per_class()
            if ref is None:
                res = ctl.on_vprime(difficulty=difficulty, n_new=exp.n_new, can_expand_more=next_per_class > 0)
            else:
                if rows and rows[-1]["phase"] == PHASE_T:
                    replay_tracker = DifficultyTracker(
                        min_delta=st.vprime_difficulty_min_delta, patience=st.vprime_difficulty_patience,
                        window=st.vprime_difficulty_window, min_difficulty=st.vprime_min_difficulty)
                nxt = replay[cycle]["phase"] if cycle < len(replay) else PHASE_STOP
                res = {"update": replay_tracker.update(difficulty), "next_phase": nxt, "reason": "replay",
                       "expansions_in_phase": ref["expansions_in_phase"]}
            upd: DifficultyUpdate = res["update"]
            d = exp.diagnostics
            row.update(
                k_after=ref["k_after"] if ref else ctl.k, vprime_expanded=bool(exp.n_new > 0), delta_rows=int(delta.size),
                delta_per_class=_per_class_counts(y_val_full[delta], classes), delta_metric=delta_metric,
                delta_error_rate=delta_err, delta_difficulty=difficulty, difficulty_change_vs_prev=upd.change_vs_prev,
                difficulty_window_mean=upd.window_mean, difficulty_gain=upd.gain,
                difficulty_no_gain_count=upd.no_gain_count, difficulty_plateau=upd.plateau,
                expansions_in_phase=res["expansions_in_phase"], delta_eval_rows=int(delta.size),
                selection_predict_rows=d.get("cost_selection_predict_rows"),
                selection_decision_rows=d.get("cost_selection_decision_rows"),
                val_ga_population_evaluations=d.get("val_ga_population_evaluations"),
                n_candidates=d.get("n_candidates"), n_scored_pool=d.get("n_scored_pool"),
                decision="expand_V_prime", next_phase=res["next_phase"], transition_reason=res["reason"],
                svm_trainings=0, val_ga_nominal_rows=d.get("val_ga_nominal_subset_rows_scored"),
                model_metric_on_vprime=phase_model["fitness"],  # z oceny na V' na początku fazy; stare próbki nie są oceniane ponownie
            )
            log.info("[krok %s | V'] faza#%s (%s) |V'| %s->%s ΔV'=%s %s(ΔV')=%s trudność=%.4f okno=%.4f zysk=%s "
                     "bez_wzrostu=%s -> %s (%s)", cycle, ctl.vprime_phase_id, row["phase_entry_reason"], exp.size_before,
                     exp.size_after, delta.size, st.vprime_metric, None if delta_metric is None else round(delta_metric, 4),
                     difficulty, upd.window_mean, None if upd.gain is None else round(upd.gain, 4), upd.no_gain_count,
                     res["next_phase"], res["reason"])
            if res["next_phase"] == PHASE_T:
                phase_model = None

        # ---------------- niezmienniki kroku ----------------
        assert int(row["k_increased"]) + int(row["vprime_expanded"]) <= 1, "K i V' zmienione w jednym kroku"
        assert row["k_after"] - row["k_per_class"] >= 0
        row["vprime_rows_after"] = int(vset.size)
        assert row["vprime_rows_after"] <= max(cap_rows, row["vprime_rows_before"]), "przekroczony limit |V'|"
        row["vprime_fraction"] = float(vset.size / n_val)
        k_increases_total += int(bool(row["k_increased"]))
        row["k_increases_total"] = k_increases_total
        row["vprime_expansions_total"] = n_expansions_total
        row["vprime_limit_reason"] = expansion_block_reason(
            vprime_rows=vset.size, cap_rows=cap_rows, n_candidates=vset.candidates().size, n_classes=n_classes,
            expansions_left=expansions_left())
        # koszt kroku: wywołania modelu przez algorytm (obserwacja pełnego V i testu liczona osobno)
        meter_sum = {k: sum(m.snapshot()[k] for m in meters) for k in
                     ("predict_calls", "predict_rows", "decision_function_calls", "decision_function_rows")}
        ga_evals = int(row["ga_svm_evaluations"] or 0)
        row.update(
            predict_calls=meter_sum["predict_calls"] + ga_evals,
            predict_rows=meter_sum["predict_rows"] + int(row["tprime_eval_rows_actual"] or 0),
            decision_function_calls=meter_sum["decision_function_calls"],
            decision_function_rows=meter_sum["decision_function_rows"],
            observation_rows=int(observer.rows_scored) - obs_rows_before
            + (int(len(X_test)) if (phase == PHASE_T and X_test is not None and st.observe_test) else 0),
        )
        row["cycle_duration_sec"] = time.perf_counter() - t0
        rows.append(row)

        if replay is not None and cycle >= len(replay):
            final_stop = "stop_replay_end"
        if final_stop is None and cycle >= st.max_cycles:
            final_stop = "stop_max_cycles"
            log.warning("STOP: limit %s kroków (zabezpieczenie)", st.max_cycles)
        if final_stop is not None:
            break

    rows[-1]["stop_decision"] = final_stop
    rows[-1]["stop_category"] = STOP_CATEGORY.get(final_stop, "other")
    np.save(checkpoints_dir / "final_vprime_ids.npy", vset.ids)
    save_mask(checkpoints_dir / "final_global_val_mask.npy", vset.mask())
    if increments:
        np.savez(checkpoints_dir / "vprime_increments.npz", **increments)

    # model reprezentatywny: wybór wyłącznie na podstawie V' (nigdy pełnego V ani testu)
    rep = last_t if (st.representative == "last_tprime_step" or best_t is None) else best_t
    rep_mask_path = checkpoints_dir / "global_best_train_mask.npy"
    save_mask(rep_mask_path, rep["mask"])
    global_best = {
        "cycle": rep["cycle"], "k_per_class": rep["k"], "metrics_val": rep["metrics_val"],
        "metrics_full_validation": rep["full"], "metrics_test": rep["test"], "train_summary": rep["train_summary"],
        "train_mask_path": str(rep_mask_path), "validation_rows": rep["vprime_rows"],
        "selection_rule": st.representative,
    }
    t_rows = [r for r in rows if r["phase"] == PHASE_T]
    v_rows = [r for r in rows if r["phase"] == PHASE_V]
    phase_study = {
        "algorithm_mode": MODE,
        "stop_decision": final_stop,
        "stop_category": STOP_CATEGORY.get(final_stop, "other"),
        "vprime_rows": int(vset.size),
        "full_validation_rows": n_val,
        "vprime_fraction": float(vset.size / n_val),
        "vprime_cap_rows": cap_rows,
        "vprime_cap_reached": bool(vset.size >= cap_rows),
        "steps": len(rows),
        "tprime_steps": len(t_rows),
        "vprime_steps": len(v_rows),
        "vprime_phases": int(sum(1 for r in t_rows if r["next_phase"] == PHASE_V)),
        "vprime_phases_mastered": int(sum(1 for r in t_rows if r["next_phase"] == PHASE_V and r["tprime_perfect"])),
        "vprime_retries": int(sum(1 for r in t_rows if r["next_phase"] == PHASE_V and not r["tprime_perfect"])),
        "retry_k_levels": sorted({int(r["k_per_class"]) for r in t_rows if r["next_phase"] == PHASE_V
                                  and not r["tprime_perfect"]}),
        "k_increases_total": int(sum(1 for r in rows if r["k_increased"])),
        "vprime_expansions": int(n_expansions_total),
        "vprime_limit_reason": rows[-1]["vprime_limit_reason"],
        "max_vprime_expansions": st.max_vprime_expansions,
        "retry_outcomes": {"started": int(sum(1 for r in t_rows if r["next_phase"] == PHASE_V and not r["tprime_perfect"])),
                           "resolved_with_progress": int(sum(1 for r in t_rows if r["retry_check"] and r["retry_progress"])),
                           "failed_checks": int(sum(1 for r in t_rows if r["retry_check"] and not r["retry_progress"]))},
        "final_k": int(rows[-1]["k_after"]),
        "schedule_replay": (cfg.get("phase_switch") or {}).get("schedule_replay"),
        "reference_stop_decision": None if replay is None else replay[-1]["stop_decision"],
        "perfect_tprime_steps": int(sum(1 for r in t_rows if r["tprime_perfect"])),
        "vprime_phase_end_reasons": _count([r["transition_reason"] for r in v_rows if r["next_phase"] == PHASE_T]),
        "mean_delta_difficulty": _nanmean([r["delta_difficulty"] for r in v_rows]),
        "mean_delta_error_rate": _nanmean([r["delta_error_rate"] for r in v_rows]),
        "settings": {k: v for k, v in st.__dict__.items() if k != "vprime"},
        "vprime": {"selection": vcfg.selection, "memory": vcfg.memory, "pool_budget_factor": vcfg.pool_budget_factor},
        "cost": {
            **cost,
            "selection_predict_rows": int(counter.get("selection_predict_rows", 0)),
            "selection_decision_rows": int(counter.get("selection_decision_rows", 0)),
            "diagnostic_scored_rows": int(counter.get("diagnostic_scored_rows", 0)),
            "observation_full_validation_rows_scored": int(observer.rows_scored),
            # --- koszt łączny i w podziale na fazy (wywołania modelu wykonane przez algorytm; bez obserwacji) ---
            "total": _cost_totals(rows),
            "tprime_phase": _cost_totals(t_rows),
            "vprime_phase": _cost_totals(v_rows),
            "observation_rows": int(sum(r["observation_rows"] or 0 for r in rows)),
        },
        "used_full_validation_for_control": False,
        "used_test_for_control": False,
    }
    write_json(metrics_dir / "phase_switch_study.json", phase_study)
    log.info("STOP %s | |V'|=%s/%s (%.3f) | kroki T'=%s V'=%s | fazy V'=%s (próby po zastoju: %s) | K=%s",
             final_stop, vset.size, n_val, vset.size / n_val, len(t_rows), len(v_rows), ctl.vprime_phase_id,
             len(ctl.retry_used_k), ctl.k)
    return {
        "cycles": rows,
        "cycle_table_rows": rows,
        "global_best": global_best,
        "final_global_val": {**summarize_split(X_val_full[vset.ids], y_val_full[vset.ids]),
                             "final_global_val_mask_path": str(checkpoints_dir / "final_global_val_mask.npy")},
        "representative_model": {
            "algorithm_mode": MODE, "training_rows": int(rep["mask"].sum()), "validation_rows": int(rep["vprime_rows"]),
            "train_mask_path": str(rep_mask_path), "cycle": rep["cycle"],
        },
        "phase_study": phase_study,
    }


def load_schedule_replay(cfg: dict) -> list[dict] | None:
    """Harmonogram z cycles.csv przebiegu referencyjnego (``phase_switch.schedule_replay``) albo None."""
    path = (cfg.get("phase_switch") or {}).get("schedule_replay")
    if not path:
        return None
    import pandas as pd

    frame = pd.read_csv(path)
    keep = ["phase", "k_per_class", "k_after", "delta_rows", "vprime_rows_before", "full_validation_rows",
            "vprime_phase_id", "phase_entry_reason", "retry_phase", "expansions_in_phase", "stop_decision"]
    out = []
    for rec in frame[keep].to_dict("records"):
        rec = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in rec.items()}
        for k in ("k_per_class", "k_after", "vprime_rows_before", "full_validation_rows", "vprime_phase_id"):
            rec[k] = int(rec[k])
        rec["delta_rows"] = None if rec["delta_rows"] is None else int(rec["delta_rows"])
        rec["retry_phase"] = str(rec["retry_phase"]).lower() == "true"
        out.append(rec)
    return out


def _replay_tprime(ref: dict, replay: list[dict], cycle: int, f: float, best_t: dict | None,
                   st: PhaseSwitchSettings) -> dict:
    """Krok T' w trybie replay: decyzja (następna faza, wzrost K) pochodzi z harmonogramu, nie z kontrolera."""
    nxt = replay[cycle]["phase"] if cycle < len(replay) else PHASE_STOP
    perfect = f >= st.perfect_threshold
    improved = None if best_t is None else bool(f > best_t["fitness"])
    dec = TDecision(nxt, increase_k=ref["k_after"] > ref["k_per_class"],
                    stop_code="stop_replay_end" if nxt == PHASE_STOP else None,
                    reason="replay harmonogramu przebiegu referencyjnego")
    return {"decision": dec, "perfect": perfect, "improved": improved, "no_improvement_count": None,
            "incumbent_fitness": None if best_t is None else best_t["fitness"], "retry_check": False,
            "retry_first_check": False, "retry_pending": False, "retry_fail_count": 0,
            "retry_reference_raw": None, "retry_reference_same_vprime": None, "retry_progress": None,
            "retry_progress_raw": None}


COST_COLUMNS = ("svm_trainings", "predict_calls", "predict_rows", "decision_function_calls", "decision_function_rows",
                "tprime_eval_rows_nominal", "tprime_eval_rows_actual", "val_ga_nominal_rows", "delta_eval_rows",
                "n_candidates")


def _cost_totals(rows: list[dict]) -> dict:
    """Suma kolumn kosztu po krokach (``n_candidates`` = suma liczb kandydatów z kolejnych doborów ΔV')."""
    return {c: int(sum((r.get(c) or 0) for r in rows)) for c in COST_COLUMNS}


def _count(items) -> dict:
    out: dict = {}
    for it in items:
        out[it] = out.get(it, 0) + 1
    return out


def _nanmean(values) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return float(np.mean(vals)) if vals else None
