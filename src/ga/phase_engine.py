"""Fazowy DAGASVM: naprzemienna optymalizacja T' (przy stałym K) i rozszerzanie V'.

ROZDZIAŁ RÓL ZBIORÓW
--------------------
* **V'** (``VPrimeSet``) — aktualny podzbiór walidacyjny. Wyłącznie metryki liczone na V'
  (oraz ρ_SV modelu) STERUJĄ algorytmem: fitness T', decyzja o rozszerzeniu V', wzroście K
  i zatrzymaniu.
* **V** (pełny zbiór walidacyjny) — (a) pula kandydatów, z której wybierane są nowe
  elementy V' (``ga.vprime``); (b) źródło metryk OBSERWACYJNYCH (``FullValidationObserver``),
  liczonych po podjęciu decyzji w cyklu i nigdy nieodczytywanych przez logikę decyzyjną.
* **Test** — wyłącznie obserwacja zewnętrzna.

LOGIKA K
--------
K (``k_per_class``) jest stałe podczas pojedynczego cyklu optymalizacji T'. Decyzja
``increase_k`` zapada co najwyżej raz na cykl (wymuszone asercją). Gdy aktualne T' osiąga
próg ``perfect_val_threshold`` na V', K NIE rośnie — algorytm najpierw utrudnia V',
a gdy V' było już utrudnione dla tego K — kończy.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ga.encoding import KPerClassConfig
from ga.fitness import FitnessConfig
from ga.loop.engine import (
    EarlyStopping,
    build_ga_config,
    run_ga_k_per_class_with_trace,
)
from ga.models.svm import SVMConfig, evaluate_fitted_svm, fit_svm_model
from ga.utils.artifacts import (
    class_distribution,
    save_mask,
    summarize_history,
    summarize_split,
    write_json,
)
from ga.vprime import (
    FullValidationObserver,
    VPrimeConfig,
    VPrimeSet,
    class_balanced_ids,
    expand_vprime,
    stratified_random_ids,
)

# Kody decyzji cyklu (kolumna ``decision`` w cycles.csv).
DECISION_EXPAND = "expand_validation_only"
DECISION_INCREASE_K = "increase_k"
STOP_DECISIONS = {
    "stop_perfect_on_current_val",
    "stop_no_validation_candidates",
    "stop_no_k_growth",
    "stop_no_growth_signal",
    "stop_max_cycles",
}


@dataclass
class PhaseSettings:
    k_start: int
    tsv: float
    phase_patience: int
    phase_min_delta: float
    sv_ratio_stability_epsilon: float
    sv_ratio_saturation_margin: float
    perfect_val_threshold: float
    max_cycles: int
    vprime: VPrimeConfig
    observe_full_validation: bool
    observe_test: bool


def parse_phase_settings(cfg: dict) -> PhaseSettings:
    """Czyta konfigurację fazowego DAGASVM (zgodnie wstecz z ``daga.val_selection_strategy``)."""
    daga_cfg = cfg.get("daga", {})
    ks_cfg = cfg.get("k_schedule", {})
    k_start = int(
        daga_cfg.get(
            "k_start",
            ks_cfg.get("k_start", cfg.get("encoding", {}).get("k_per_class", 20)),
        )
    )
    vp = cfg.get("vprime", {}) or {}
    selection = str(vp.get("selection", daga_cfg.get("val_selection_strategy", "heuristic")))
    vprime = VPrimeConfig(
        selection=selection,
        memory=bool(vp.get("memory", True)),
        mode=str(vp.get("mode", "growing")),
        fixed_size=None if vp.get("fixed_size") is None else int(vp["fixed_size"]),
        score_candidate_pool=bool(vp.get("score_candidate_pool", True)),
    )
    obs = cfg.get("observation", {}) or {}
    return PhaseSettings(
        k_start=k_start,
        tsv=float(daga_cfg.get("TSV", daga_cfg.get("tsv", 0.25))),
        phase_patience=int(daga_cfg.get("Ts", daga_cfg.get("stable_chromosome_size_counter", 3))),
        phase_min_delta=float(
            daga_cfg.get("phase_min_delta", cfg.get("ga", {}).get("early_stopping", {}).get("min_delta", 0.0))
        ),
        sv_ratio_stability_epsilon=float(daga_cfg.get("sv_ratio_stability_epsilon", 0.02)),
        sv_ratio_saturation_margin=float(daga_cfg.get("sv_ratio_saturation_margin", 0.15)),
        perfect_val_threshold=float(daga_cfg.get("perfect_val_threshold", 1.0)),
        max_cycles=int(daga_cfg.get("max_cycles", 500)),
        vprime=vprime,
        observe_full_validation=bool(obs.get("full_validation", True)),
        observe_test=bool(obs.get("test", True)),
    )


def next_k_from_sv_ratio(current_k: int, sv_ratio: float | None, tsv: float) -> int:
    if sv_ratio is None:
        return current_k + 1
    clipped_tsv = float(np.clip(tsv, 0.0, 0.999999))
    delta_k = 1.0 + abs(float(sv_ratio) - clipped_tsv) / (1.0 - clipped_tsv)
    next_k = int(np.ceil(float(current_k) * delta_k))
    return max(current_k + 1, next_k)


@dataclass
class CycleDecision:
    decision: str
    reason: str

    @property
    def expands_vprime(self) -> bool:
        return self.decision == DECISION_EXPAND

    @property
    def increases_k(self) -> bool:
        return self.decision == DECISION_INCREASE_K

    @property
    def stops(self) -> bool:
        return self.decision in STOP_DECISIONS


def decide_cycle(
    *,
    perfect_on_current_val: bool,
    has_growth_signal: bool,
    expansion_possible: bool,
    vprime_mode: str,
    validation_hardened_for_current_k: bool,
    sv_ratio_stable_since_hardening: bool,
    can_increase_k: bool,
    forced_growth_after_patience: bool = False,
) -> CycleDecision:
    """Czysta funkcja decyzyjna cyklu. Argumenty zależą WYŁĄCZNIE od V', T' i puli kandydatów
    (czy w ogóle istnieją jeszcze elementy do dodania) — nigdy od metryk pełnego V."""
    if perfect_on_current_val:
        if expansion_possible and not validation_hardened_for_current_k:
            return CycleDecision(
                DECISION_EXPAND,
                "T' osiąga próg 100% na V'; K nie rośnie automatycznie — najpierw utrudniam V'",
            )
        return CycleDecision(
            "stop_perfect_on_current_val",
            "T' osiąga próg na V' mimo wcześniejszego utrudnienia V' (lub V' nie da się rozszerzyć); K bez zmian",
        )
    if not has_growth_signal:
        return CycleDecision(
            "stop_no_growth_signal",
            "ρ_SV poniżej TSV (lub K osiągnęło limit klasy): brak przesłanek do dalszego wzrostu K",
        )
    if vprime_mode == "fixed":
        if can_increase_k:
            return CycleDecision(
                DECISION_INCREASE_K,
                "V' jest stałe (wariant bazowy, brak kroku rozszerzenia); sygnał wzrostu K z ρ_SV",
            )
        return CycleDecision("stop_no_k_growth", "nie da się zwiększyć K ponad bieżącą wartość")
    if not validation_hardened_for_current_k:
        if expansion_possible:
            return CycleDecision(
                DECISION_EXPAND,
                "jest sygnał wzrostu K, ale V' nie było jeszcze utrudnione przy tym K — najpierw rozszerzam V'",
            )
        return CycleDecision("stop_no_validation_candidates", "brak nowych próbek do rozszerzenia V'")
    if sv_ratio_stable_since_hardening:
        if expansion_possible:
            return CycleDecision(
                DECISION_EXPAND,
                "ρ_SV stabilne po utrudnieniu V' (to samo K może wystarczać) — rozszerzam V' zamiast zwiększać K",
            )
        return CycleDecision("stop_no_validation_candidates", "brak nowych próbek do rozszerzenia V'")
    if can_increase_k:
        reason = "V' było już utrudnione dla tego K, a ρ_SV się zmieniło — zwiększam K"
        if forced_growth_after_patience:
            reason = "wyczerpano cierpliwość Ts przy stabilnym ρ_SV — wymuszony wzrost K"
        return CycleDecision(DECISION_INCREASE_K, reason)
    return CycleDecision("stop_no_k_growth", "nie da się zwiększyć K ponad bieżącą wartość")


def _eval_dict(clf, fit_meta: dict, X, y, metric: str) -> dict:
    res = evaluate_fitted_svm(clf, X, y, metric=metric)
    return {
        "metric": res["metric"],
        "all": res["all"],
        "fit_time": fit_meta["fit_time"],
        "score_time": res["score_time"],
        "num_support_vectors": fit_meta["num_support_vectors"],
    }


def run_phase_experiment(
    X_train,
    y_train,
    X_val_full,
    y_val_full,
    X_test,
    y_test,
    svm_cfg: SVMConfig,
    cfg: dict,
    log,
    artifact_layout,
    *,
    observer: FullValidationObserver | None = None,
    ga_runner=run_ga_k_per_class_with_trace,
) -> dict:
    """Pętla fazowego DAGASVM. Zwraca słownik zgodny z ``run_dagasvm_phase_mode``."""
    metric = cfg.get("fitness", {}).get("metric", "balanced_accuracy")
    fcfg = FitnessConfig(metric=metric, class_imbalance_penalty=0.0)
    st = parse_phase_settings(cfg)
    seed = int(cfg.get("seed", 42))
    vcfg = st.vprime

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_val_full = np.asarray(X_val_full, dtype=float)
    y_val_full = np.asarray(y_val_full)
    X_test = None if X_test is None else np.asarray(X_test, dtype=float)
    y_test = None if y_test is None else np.asarray(y_test)

    if observer is None:
        observer = FullValidationObserver(
            X_val_full, y_val_full, metric, evaluate_fitted_svm, enabled=st.observe_full_validation
        )
    val_ga_gcfg = (
        build_ga_config(cfg.get("val_ga", {})) if vcfg.selection == "evolutionary" else None
    )

    classes = np.unique(y_train)
    n_classes = int(classes.size)
    train_class_cap = int(np.unique(y_train, return_counts=True)[1].min())
    n_val_full = int(X_val_full.shape[0])

    # --- inicjalizacja V' ---
    rng = np.random.default_rng(seed)
    vset = VPrimeSet(n_val_full)
    if vcfg.mode == "fixed":
        init_rng = np.random.default_rng(np.random.SeedSequence([seed, 202]))
        vset.replace(stratified_random_ids(y_val_full, int(vcfg.fixed_size), init_rng))
        log.info(
            "V' (wariant bazowy: losowe, stałe): %s / %s próbek (stratyfikowane, rozmiar zadany z konfiguracji)",
            vset.size,
            n_val_full,
        )
    else:
        vset.replace(class_balanced_ids(y_val_full, st.k_start, rng))
        log.info(
            "Początkowa walidacja V': %s / %s próbek (class-balanced, per_class=%s)",
            vset.size,
            n_val_full,
            st.k_start,
        )
    log.info(
        "Konfiguracja V': selection=%s memory=%s mode=%s | obserwacja pełnego V: %s",
        vcfg.selection,
        vcfg.memory,
        vcfg.mode,
        "włączona" if observer.enabled else "wyłączona",
    )

    current_k = st.k_start
    validation_hardened_for_current_k = False
    sv_ratio_at_hardening: float | None = None
    stable_cycles_at_current_k = 0
    best_observed_full_val = -1.0  # wyłącznie raportowanie
    total_k_increases = 0

    metrics_dir = Path(artifact_layout["metrics"])
    checkpoints_dir = Path(artifact_layout["checkpoints"])
    cycle_summaries: list[dict] = []
    cycle_table_rows: list[dict] = []
    cycle_candidates: list[dict] = []
    counter = {"model_predictions": 0, "diagnostic_predictions": 0}
    cycle = 0
    final_stop_decision = None

    while True:
        cycle_start = time.perf_counter()
        cycle += 1
        log.info("=" * 60)
        log.info("FAZA T' (cykl %s), K=%s", cycle, int(current_k))

        vp_ids = vset.ids
        X_vp = X_val_full[vp_ids]
        y_vp = y_val_full[vp_ids]
        log.info("[cykl %s] rozmiar V': %s / %s próbek", cycle, vp_ids.size, n_val_full)

        kcfg = KPerClassConfig(k_per_class=int(current_k), random_state=seed + cycle - 1)
        gcfg_train = build_ga_config(cfg.get("ga", {}))
        # patience/min_delta GA w fazie T' pochodzą z daga.Ts / daga.phase_min_delta
        gcfg_train.early_stopping = EarlyStopping(
            patience=max(1, st.phase_patience), min_delta=st.phase_min_delta
        )

        history_train, best_train_mask, train_trace, best_masks_per_gen = ga_runner(
            X_pool=X_train,
            y_pool=y_train,
            X_val=X_vp,
            y_val=y_vp,
            kcfg=kcfg,
            svm_cfg=svm_cfg,
            fcfg=fcfg,
            gcfg=gcfg_train,
            seed=seed + cycle - 1,
        )
        train_history_summary = summarize_history(
            history_train, gcfg_train.population_size, evaluation_label="svm_evaluations"
        )
        train_history_summary["trace"] = train_trace
        train_history_path = metrics_dir / f"cycle_{cycle:02d}_train_ga_history.json"
        write_json(train_history_path, train_history_summary)
        best_masks_path = checkpoints_dir / f"cycle_{cycle:02d}_best_masks_per_generation.npy"
        np.save(best_masks_path, best_masks_per_gen.astype(bool, copy=False))
        best_train_mask_path = checkpoints_dir / f"cycle_{cycle:02d}_best_train_mask.npy"
        save_mask(best_train_mask_path, best_train_mask)

        X_tr_best = X_train[best_train_mask]
        y_tr_best = y_train[best_train_mask]
        clf, fit_meta = fit_svm_model(X_tr_best, y_tr_best, svm_cfg, return_metadata=True)

        # ================= METRYKI STERUJĄCE (tylko V') =================
        best_res_val = _eval_dict(clf, fit_meta, X_vp, y_vp, metric)
        log.info(
            "[T' cykl %s] metryki sterujące, V' %s=%.4f | SV=%s | fit=%.3fs",
            cycle,
            metric,
            best_res_val["metric"],
            best_res_val["num_support_vectors"],
            best_res_val["fit_time"],
        )
        best_sv_ratio = (
            float(best_res_val["num_support_vectors"] / len(X_tr_best)) if len(X_tr_best) > 0 else None
        )
        validation_hardened_before_cycle = bool(validation_hardened_for_current_k)
        candidate_ids = vset.candidates()
        prev_k = int(current_k)
        perfect_on_current_val = bool(best_res_val["metric"] >= st.perfect_val_threshold)
        has_growth_signal = (
            best_sv_ratio is not None and best_sv_ratio >= st.tsv and prev_k < train_class_cap
        )
        candidate_next_k = next_k_from_sv_ratio(prev_k, best_sv_ratio, st.tsv) if has_growth_signal else prev_k
        candidate_next_k = min(candidate_next_k, train_class_cap)
        candidate_add_per_class = max(0, candidate_next_k - prev_k)

        sv_ratio_delta_since_hardening = (
            None
            if sv_ratio_at_hardening is None or best_sv_ratio is None
            else float(best_sv_ratio - sv_ratio_at_hardening)
        )
        sv_ratio_near_threshold = (
            best_sv_ratio is not None and best_sv_ratio < (st.tsv + st.sv_ratio_saturation_margin)
        )
        sv_ratio_stable_raw = (
            sv_ratio_delta_since_hardening is not None
            and abs(sv_ratio_delta_since_hardening) < st.sv_ratio_stability_epsilon
            and sv_ratio_near_threshold
        )
        sv_ratio_stable_since_hardening = sv_ratio_stable_raw and stable_cycles_at_current_k < st.phase_patience
        forced_growth_after_patience = sv_ratio_stable_raw and not sv_ratio_stable_since_hardening

        decision = decide_cycle(
            perfect_on_current_val=perfect_on_current_val,
            has_growth_signal=has_growth_signal,
            expansion_possible=(vcfg.mode == "growing" and candidate_ids.size > 0),
            vprime_mode=vcfg.mode,
            validation_hardened_for_current_k=validation_hardened_for_current_k,
            sv_ratio_stable_since_hardening=sv_ratio_stable_since_hardening,
            can_increase_k=candidate_next_k > prev_k,
            forced_growth_after_patience=forced_growth_after_patience,
        )
        if cycle >= st.max_cycles and not decision.stops:
            decision = CycleDecision("stop_max_cycles", f"osiągnięto limit {st.max_cycles} cykli (zabezpieczenie)")
        # inwarianty polityki K
        assert not (perfect_on_current_val and decision.increases_k), "K nie może rosnąć przy 100% na V'"
        k_increases_this_cycle = 1 if decision.increases_k else 0
        assert k_increases_this_cycle <= 1
        next_k = candidate_next_k if decision.increases_k else prev_k

        log.info(
            "[cykl %s] DECYZJA=%s | powód: %s | ρ_SV=%s (TSV=%s) | V' perfect=%s",
            cycle,
            decision.decision,
            decision.reason,
            None if best_sv_ratio is None else round(best_sv_ratio, 4),
            st.tsv,
            perfect_on_current_val,
        )

        # ================= DOBÓR V' (V jako pula kandydatów) =================
        expansion = None
        add_per_class = 0
        if decision.expands_vprime:
            add_per_class = max(1, candidate_add_per_class or 1)
            log.info(
                "FAZA V' (cykl %s, strategia=%s, pamięć=%s): dodanie %s próbek/klasę przy stałym K=%s",
                cycle,
                vcfg.selection,
                "tak" if vcfg.memory else "nie",
                add_per_class,
                prev_k,
            )
            sel_rng = np.random.default_rng(np.random.SeedSequence([seed, cycle, 101]))
            expansion = expand_vprime(
                vset,
                memory=vcfg.memory,
                strategy=vcfg.selection,
                n_add=add_per_class * n_classes,
                per_class_add=add_per_class,
                n_classes=n_classes,
                X_val=X_val_full,
                y_val=y_val_full,
                clf=clf,
                metric=metric,
                val_ga_cfg=val_ga_gcfg,
                seed=seed + cycle - 1,
                rng=sel_rng,
                counter=counter,
                score_candidate_pool=vcfg.score_candidate_pool,
            )
        expanded_ok = bool(expansion is not None and expansion.n_new > 0)
        size_before = int(vp_ids.size)
        size_after = int(vset.size)

        val_history_summary = {
            "selection_strategy": vcfg.selection,
            "memory": bool(vcfg.memory),
            "planned_add_per_class": int(add_per_class),
            "selected_rows": 0 if expansion is None else int(expansion.selected_ids.size),
            "candidate_rows": int(candidate_ids.size),
            "best_sv_ratio": best_sv_ratio,
            "cycle_decision": decision.decision,
            "decision_reason": decision.reason,
            "validation_hardened_for_current_k_before_cycle": bool(validation_hardened_before_cycle),
            **({} if expansion is None else expansion.diagnostics),
        }
        val_history_path = metrics_dir / f"cycle_{cycle:02d}_val_ga_history.json"
        write_json(val_history_path, val_history_summary)
        selected_ids = np.empty(0, dtype=np.int64) if expansion is None else expansion.selected_ids
        adv_val_mask = np.zeros(n_val_full, dtype=bool)
        adv_val_mask[selected_ids] = True
        adv_val_mask_path = checkpoints_dir / f"cycle_{cycle:02d}_adv_val_mask.npy"
        save_mask(adv_val_mask_path, adv_val_mask)
        global_val_mask_path = checkpoints_dir / f"cycle_{cycle:02d}_global_val_mask.npy"
        save_mask(global_val_mask_path, vset.mask())
        np.save(checkpoints_dir / f"cycle_{cycle:02d}_vprime_ids.npy", vset.ids)

        adv_metric = None
        if expansion is not None and selected_ids.size:
            adv_metric = float(
                evaluate_fitted_svm(clf, X_val_full[selected_ids], y_val_full[selected_ids], metric=metric)["metric"]
            )
            log.info(
                "[V' cykl %s] przyrost: %s %s=%.4f (V' przed: %.4f), błędy modelu w przyroście: %s/%s",
                cycle,
                "wybrano",
                metric,
                adv_metric,
                best_res_val["metric"],
                expansion.diagnostics.get("selected_errors"),
                expansion.diagnostics.get("selected_rows"),
            )
        if expansion is not None:
            log.info(
                "[V' cykl %s] V': %s -> %s próbek (nowych: %s, usuniętych: %s)",
                cycle,
                expansion.size_before,
                expansion.size_after,
                expansion.n_new,
                expansion.n_dropped,
            )

        # ================= AKTUALIZACJA K / STANU FAZ =================
        if decision.expands_vprime and expanded_ok:
            current_k = prev_k
            if not validation_hardened_before_cycle:
                sv_ratio_at_hardening = best_sv_ratio
            validation_hardened_for_current_k = True
            if sv_ratio_stable_since_hardening:
                stable_cycles_at_current_k += 1
            log.info(
                "Przejście V' -> T': K=%s bez zmian (wzrost K w tym cyklu: nie); stable_cycles_at_current_k=%s/%s",
                current_k,
                stable_cycles_at_current_k,
                st.phase_patience,
            )
        elif decision.increases_k and next_k > prev_k:
            current_k = next_k
            total_k_increases += 1
            validation_hardened_for_current_k = False
            sv_ratio_at_hardening = None
            stable_cycles_at_current_k = 0
            log.info(
                "Przejście V' -> T': wzrost K %s -> %s (jedyny wzrost K w tym cyklu)%s",
                prev_k,
                current_k,
                " [wymuszony po wyczerpaniu cierpliwości Ts]" if forced_growth_after_patience else "",
            )
        else:
            current_k = prev_k
        assert (int(current_k) > prev_k) == (decision.increases_k and next_k > prev_k), "niespójna aktualizacja K"

        # ================= OBSERWACJA ZEWNĘTRZNA (po decyzji; nie steruje) =================
        best_res_full_val = observer.observe(clf)
        if best_res_full_val is not None:
            best_observed_full_val = max(best_observed_full_val, best_res_full_val["metric"])
            log.info(
                "[T' cykl %s] OBSERWACJA pełne V (nie steruje): %s=%.4f",
                cycle,
                metric,
                best_res_full_val["metric"],
            )
        best_res_test = None
        if X_test is not None and st.observe_test:
            best_res_test = _eval_dict(clf, fit_meta, X_test, y_test, metric)
            log.info("[T' cykl %s] OBSERWACJA test: %s=%.4f", cycle, metric, best_res_test["metric"])

        cycle_duration_sec = time.perf_counter() - cycle_start
        cycle_candidates.append(
            {
                "cycle": cycle,
                "k_per_class": prev_k,
                "train_mask": best_train_mask.copy(),
                "metrics_val": best_res_val,
                "metrics_full_validation": best_res_full_val,
                "metrics_test": best_res_test,
                "train_mask_path": str(best_train_mask_path),
                "train_summary": summarize_split(X_tr_best, y_tr_best),
                "validation_rows": int(vp_ids.size),
            }
        )

        diag = {} if expansion is None else expansion.diagnostics
        cycle_summary = {
            "cycle": cycle,
            "k_per_class": prev_k,
            "k_after": int(current_k),
            "decision": decision.decision,
            "decision_reason": decision.reason,
            "perfect_on_current_val": perfect_on_current_val,
            "validation_hardened_for_current_k_before_cycle": validation_hardened_before_cycle,
            "k_increased_this_cycle": bool(decision.increases_k),
            "k_increases_this_cycle": k_increases_this_cycle,
            "best_sv_ratio": best_sv_ratio,
            "sv_ratio_at_hardening": sv_ratio_at_hardening,
            "sv_ratio_delta_since_hardening": sv_ratio_delta_since_hardening,
            "sv_ratio_near_threshold": sv_ratio_near_threshold,
            "sv_ratio_stable_since_hardening": sv_ratio_stable_since_hardening,
            "stable_cycles_at_current_k": stable_cycles_at_current_k,
            "forced_growth_after_patience": forced_growth_after_patience,
            "used_full_validation_for_control": False,
            "control_metrics_source": "V_prime",
            "observation_metrics_source": ["full_V", "test"],
            "cycle_duration_sec": cycle_duration_sec,
            "train_pool": summarize_split(X_train, y_train),
            "current_val": summarize_split(X_vp, y_vp),
            "best_train": summarize_split(X_tr_best, y_tr_best),
            "train_ga": {
                "history_path": str(train_history_path),
                "best_train_mask_path": str(best_train_mask_path),
                "best_masks_per_generation_path": str(best_masks_path),
                "history_summary": train_history_summary,
                "best_val": best_res_val,
                "best_full_validation": best_res_full_val,
                "best_test": best_res_test,
            },
            "val_ga": {
                "candidate_rows": int(candidate_ids.size),
                "history_path": str(val_history_path),
                "adv_val_mask_path": str(adv_val_mask_path),
                "history_summary": val_history_summary,
                "adv_increment_metric": adv_metric,
                "adv_selected_rows": int(selected_ids.size),
                "selection_strategy": vcfg.selection,
                "adv_selected_class_distribution": (
                    class_distribution(y_val_full[selected_ids]) if selected_ids.size else {}
                ),
                "global_val_mask_path": str(global_val_mask_path),
                "global_val_rows_before": size_before,
                "global_val_rows_after": size_after,
                "global_val_rows_added": size_after - size_before,
            },
        }
        cycle_summaries.append(cycle_summary)
        cycle_table_rows.append(
            {
                "cycle": cycle,
                "k_per_class": prev_k,
                "k_after": int(current_k),
                "cycle_duration_sec": cycle_duration_sec,
                "train_ga_best_fitness": train_history_summary["best_final_fitness"],
                "train_ga_generations_executed": train_history_summary["generations_executed"],
                "train_ga_svm_evaluations": train_history_summary["svm_evaluations"],
                "train_ga_best_val_metric": float(best_res_val["metric"]),
                "train_ga_test_metric": None if best_res_test is None else float(best_res_test["metric"]),
                "train_ga_best_full_validation_metric": (
                    None if best_res_full_val is None else float(best_res_full_val["metric"])
                ),
                "val_ga_best_fitness": None,
                "val_ga_generations_executed": None,
                "val_ga_population_evaluations": diag.get("val_ga_population_evaluations"),
                "val_ga_adv_increment_metric": adv_metric,
                "decision": decision.decision,
                "decision_reason": decision.reason,
                "perfect_on_current_val": perfect_on_current_val,
                "k_increased_this_cycle": bool(decision.increases_k),
                "k_increases_this_cycle": k_increases_this_cycle,
                "best_sv_ratio": best_sv_ratio,
                "sv_ratio_at_hardening": sv_ratio_at_hardening,
                "sv_ratio_delta_since_hardening": sv_ratio_delta_since_hardening,
                "sv_ratio_near_threshold": sv_ratio_near_threshold,
                "sv_ratio_stable_since_hardening": sv_ratio_stable_since_hardening,
                "stable_cycles_at_current_k": stable_cycles_at_current_k,
                "forced_growth_after_patience": forced_growth_after_patience,
                "val_rows_before": size_before,
                "val_rows_after": size_after,
                "val_rows_added": size_after - size_before,
                "vprime_new_elements": None if expansion is None else expansion.n_new,
                "vprime_dropped_elements": None if expansion is None else expansion.n_dropped,
                "increment_errors": diag.get("selected_errors"),
                "increment_error_rate": diag.get("selected_error_rate"),
                "candidate_pool_error_rate": diag.get("candidate_pool_error_rate"),
                "increment_mean_abs_margin": diag.get("selected_mean_abs_margin"),
                "num_support_vectors": int(best_res_val["num_support_vectors"]),
                "train_rows": int(len(X_tr_best)),
                "train_ga_test_accuracy": (
                    None if best_res_test is None else best_res_test["all"].get("accuracy")
                ),
                "train_ga_full_validation_accuracy": (
                    None if best_res_full_val is None else best_res_full_val["all"].get("accuracy")
                ),
                "control_model_predictions_cum": int(counter["model_predictions"]),
            }
        )

        # ================= WARUNKI STOPU (tylko V', T', K) =================
        if decision.stops:
            final_stop_decision = decision.decision
            log.info("STOP (%s): %s", decision.decision, decision.reason)
            break
        if current_k >= train_class_cap:
            final_stop_decision = "stop_k_class_cap"
            log.info("STOP: osiągnięto naturalny limit danych treningowych dla K=%s.", current_k)
            break
        if decision.expands_vprime and not expanded_ok:
            final_stop_decision = "stop_no_new_vprime_elements"
            log.info("STOP: zaplanowano rozszerzenie V', ale nie wybrano żadnych nowych próbek.")
            break

    final_mask_path = checkpoints_dir / "final_global_val_mask.npy"
    save_mask(final_mask_path, vset.mask())
    np.save(checkpoints_dir / "final_vprime_ids.npy", vset.ids)

    # Reprezentatywny model = ostatni ukończony cykl (nie wybieramy go po pełnym V).
    representative_info = cycle_candidates[-1]
    global_best_mask_path = checkpoints_dir / "global_best_train_mask.npy"
    save_mask(global_best_mask_path, representative_info["train_mask"])
    full_val_final = representative_info["metrics_full_validation"]
    log.info(
        "REPREZENTATYWNY model (ostatni cykl %s, K=%s): V' %s=%.4f | pełne V (obserwacja)=%s | test (obserwacja)=%s",
        representative_info["cycle"],
        representative_info["k_per_class"],
        metric,
        representative_info["metrics_val"]["metric"],
        None if full_val_final is None else round(full_val_final["metric"], 4),
        None
        if representative_info["metrics_test"] is None
        else round(representative_info["metrics_test"]["metric"], 4),
    )
    global_best_summary = {
        "cycle": representative_info["cycle"],
        "k_per_class": representative_info["k_per_class"],
        "metrics_val": representative_info["metrics_val"],
        "metrics_full_validation": full_val_final,
        "metrics_test": representative_info["metrics_test"],
        "train_summary": representative_info["train_summary"],
        "train_mask_path": str(global_best_mask_path),
        "validation_rows": representative_info["validation_rows"],
        "selection_rule": "last_completed_cycle",
        "best_observed_full_validation_metric": float(best_observed_full_val),
    }
    representative_model = {
        "algorithm_mode": "dagasvm_phase",
        "training_rows": int(representative_info["train_summary"]["rows"]),
        "validation_rows": int(representative_info["validation_rows"]),
        "train_mask_path": str(global_best_mask_path),
        "cycle": representative_info["cycle"],
    }
    n_expansions = sum(1 for c in cycle_summaries if c["decision"] == DECISION_EXPAND)
    phase_study = {
        "vprime": {
            "selection": vcfg.selection,
            "memory": bool(vcfg.memory),
            "mode": vcfg.mode,
            "fixed_size": vcfg.fixed_size,
        },
        "stop_decision": final_stop_decision,
        "cycles": len(cycle_summaries),
        "k_increases_total": total_k_increases,
        "vprime_expansions": n_expansions,
        "max_k_increases_in_one_cycle": max((c["k_increases_this_cycle"] for c in cycle_summaries), default=0),
        "cost": {
            "control_model_predictions": int(counter["model_predictions"]),
            "diagnostic_predictions": int(counter["diagnostic_predictions"]),
            "observation_full_validation_rows_scored": int(observer.rows_scored),
            "observation_full_validation_calls": int(observer.calls),
            "train_ga_svm_evaluations": int(
                sum(c["train_ga"]["history_summary"]["svm_evaluations"] for c in cycle_summaries)
            ),
            "val_ga_population_evaluations": int(
                sum(
                    (c["val_ga"]["history_summary"].get("val_ga_population_evaluations") or 0)
                    for c in cycle_summaries
                )
            ),
        },
        "observation_full_validation_enabled": bool(observer.enabled),
        "used_full_validation_for_control": False,
    }

    return {
        "cycles": cycle_summaries,
        "cycle_table_rows": cycle_table_rows,
        "global_best": global_best_summary,
        "final_global_val": {
            **summarize_split(X_val_full[vset.ids], y_val_full[vset.ids]),
            "final_global_val_mask_path": str(final_mask_path),
        },
        "representative_model": representative_model,
        "phase_study": phase_study,
    }
