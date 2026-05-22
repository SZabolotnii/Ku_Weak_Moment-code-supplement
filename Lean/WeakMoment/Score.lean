/-
# Bounded / redescending scores and the bounded CF sine-score

Deterministic-backbone lemmas for the weak-moment estimators.

* The characteristic-function (CF) score `r ↦ sin(u r)` is bounded by 1 and odd —
  the boundedness that makes the weak-CF estimator free of moment-order fragility
  (paper §4.4: `sin(ur)` as the bounded, all-order limit of the odd-polynomial
  hierarchy).
* The degree-one weak-PMM score with the Cauchy-like window equals (up to the
  constant `γ²`) the known-scale Cauchy-MLE score — the algebraic half of the
  degree-1 ↔ M-estimator correspondence (paper §4.1, Proposition; numerically
  checked to `1e-9` in `tests/test_invariants.py`).
-/
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Tactic

namespace WeakMoment

noncomputable section

/-- CF score used by the weak-CF estimator. -/
def cfScore (u r : ℝ) : ℝ := Real.sin (u * r)

/-- The CF score is bounded by 1, uniformly in `u, r` — no tail amplification. -/
theorem abs_cfScore_le_one (u r : ℝ) : |cfScore u r| ≤ 1 := by
  unfold cfScore
  exact abs_le.2 ⟨Real.neg_one_le_sin _, Real.sin_le_one _⟩

/-- The CF score is odd in the residual — its expectation vanishes for symmetric
noise at the true parameter (paper §4.4). -/
theorem cfScore_odd (u r : ℝ) : cfScore u (-r) = - cfScore u r := by
  unfold cfScore
  rw [mul_neg, Real.sin_neg]

/-- Cauchy-like window `w_σ(r) = 1 / (1 + (r/σ)²)`. -/
def cauchyWeight (σ r : ℝ) : ℝ := 1 / (1 + (r / σ) ^ 2)

/-- Known-scale Cauchy-MLE per-observation score `r / (γ² + r²)`
(proportional to `∂/∂μ` of the negative Cauchy log-likelihood). -/
def cauchyMleScore (γ r : ℝ) : ℝ := r / (γ ^ 2 + r ^ 2)

/-- **Degree-1 ↔ Cauchy-MLE identity.** With bandwidth `σ = γ`, the degree-one
weak-PMM score `w_σ(r)·r` equals `γ²` times the known-scale Cauchy-MLE score.
Hence the degree-one weak estimator with the Cauchy-like window solves exactly
the Cauchy-MLE estimating equation. -/
theorem cauchyWeight_mul_eq_mle (γ r : ℝ) (hγ : γ ≠ 0) :
    cauchyWeight γ r * r = γ ^ 2 * cauchyMleScore γ r := by
  unfold cauchyWeight cauchyMleScore
  have hγ2 : (0 : ℝ) < γ ^ 2 := by positivity
  have hden : (0 : ℝ) < γ ^ 2 + r ^ 2 := by positivity
  have h1 : (1 : ℝ) + (r / γ) ^ 2 = (γ ^ 2 + r ^ 2) / γ ^ 2 := by
    field_simp
  rw [h1]
  field_simp

end

end WeakMoment
