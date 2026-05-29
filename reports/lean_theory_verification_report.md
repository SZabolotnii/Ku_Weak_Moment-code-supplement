# Lean theory verification report — WeakMoment backbone

**Status:** `lake build WeakMoment` completes successfully — **0 `sorry`, 0 `axiom`**
(verified 2026-05-21, Lean `v4.26.0`, Mathlib `v4.26.0`).

The `WeakMoment` Lean library (`Lean/WeakMoment/`) machine-checks the
*deterministic* (analytic / algebraic) core of the weak-moment regression
estimators. The asymptotic statements (consistency, asymptotic normality,
breakdown) remain classical and are written in `paper/main.tex` §"Asymptotic
theory", leaning on the weak CLT of Labouriau (2026b) and standard GMM/M-estimation
theory — Mathlib 4.26 has no M-/GMM-estimation CLT, so those are deliberately
out of formalization scope.

## Lean lemma ↔ paper ↔ Python check

| # | Lean lemma (`WeakMoment.*`) | Statement | Paper | Python verification |
|---|------------------------------|-----------|-------|---------------------|
| i | `Moments.weak_moment_integrable` | `φ ∈ 𝒮(ℝ) ⇒ x ↦ xʲφ(x)` integrable ∀ j (weak moments of all orders finite) | §4.3, §6.4(i) | `experiments/weak_moment_sanity` — windowed `m₂` concentrates at `1/n` for Schwartz/compact windows; diverges for the non-Schwartz Cauchy-like window |
| ii | `Score.abs_cfScore_le_one`, `Score.cfScore_odd` | `|sin(ur)| ≤ 1` and `sin(u(−r)) = −sin(ur)` (bounded, odd CF score) | §4.4, §6.4(ii) | `tests/test_invariants.py` (weak-CF score boundedness / symmetry sanity) |
| iii | `Score.cauchyWeight_mul_eq_mle` | `w_γ(r)·r = γ² · r/(γ²+r²)` (degree-1 ↔ Cauchy-MLE) | Prop. (§4.1), §6.4(iii) | `tests/test_invariants.py` — degree-1 weak PMM ≡ Cauchy MLE to `1e-9` |
| iv | `Foundations.foc_kappaW` | `κ^w = (m₆−3m₂m₄)/(m₄−3m₂²)` solves the variance first-order condition `foc(κ^w)=0` | Prop. `prop:kappa` (§4.3), §6.4(iv) | `src/ku_weak_moment/estimators.py::_weak_pmm3_kappa` uses exactly this formula; numerical FOC check in `tests/test_invariants.py` |
| v | `Foundations.ident_of_unit_det` | invertible Jacobian `G ⇒ (Gδ=0 ⇒ δ=0)` (local-identifiability core) | Prop. `prop:idrank` (§6.3) | empirical root-multiplicity diagnostic (Phase 2) |

## Build

```bash
cd /Users/docua/Project/Research/Ku_Weak_Moment
lake build WeakMoment      # 0 sorry, 0 axiom
```

The Mathlib build cache lives under `.lake/` (gitignored); `lake-manifest.json`
pins the exact Mathlib commit. First build fetches the prebuilt Mathlib oleans
via `lake exe cache get`.
