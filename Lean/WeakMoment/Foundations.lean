/-
# κ^w optimality (first-order condition) and local identifiability

* `foc_kappaW`: the implemented coupling coefficient
  `κ^w = (m₆ − 3 m₂ m₄)/(m₄ − 3 m₂²)` solves the first-order stationarity
  condition of the degree-3 asymptotic-variance ratio `V(κ)` (paper
  Proposition on κ^w optimality). The reduction `V'(κ)=0 ⇔ foc(κ)=0` is the
  paper's calculus; here we machine-check that `κ^w` is the FOC root.
* `ident_of_unit_det`: the linear-algebra core of local identifiability —
  a full-rank (invertible) moment-map Jacobian forces the linearized
  estimating equation to have only the trivial solution.
-/
import Mathlib

namespace WeakMoment

noncomputable section

/-- First-order stationarity residual of the degree-3 variance ratio
`V(κ) = (κ² m₂ − 2κ m₄ + m₆)/(κ − 3 m₂)²`. One has `V'(κ)=0 ⇔ foc κ = 0`
(analytic reduction in the paper). -/
def foc (m2 m4 m6 κ : ℝ) : ℝ := κ * (m4 - 3 * m2 ^ 2) - (m6 - 3 * m2 * m4)

/-- The implemented kurtosis-coupling coefficient. -/
def kappaW (m2 m4 m6 : ℝ) : ℝ := (m6 - 3 * m2 * m4) / (m4 - 3 * m2 ^ 2)

/-- **κ^w solves the first-order stationarity condition.** With nondegenerate
windowed excess kurtosis (`m₄ ≠ 3 m₂²`), `κ^w` is the unique root of `foc`,
hence the stationary point (minimizer) of the asymptotic-variance ratio. -/
theorem foc_kappaW (m2 m4 m6 : ℝ) (h : m4 - 3 * m2 ^ 2 ≠ 0) :
    foc m2 m4 m6 (kappaW m2 m4 m6) = 0 := by
  unfold foc kappaW
  field_simp
  ring

/-- **Local identifiability (linear core).** If the moment-map Jacobian `G` is
invertible (full rank), the linearized estimating equation `G δ = 0` has only
`δ = 0`; this is the kernel-triviality behind local uniqueness of the root. -/
theorem ident_of_unit_det {n : ℕ} (G : Matrix (Fin n) (Fin n) ℝ)
    (hG : IsUnit G.det) (δ : Fin n → ℝ) (h : G.mulVec δ = 0) : δ = 0 := by
  have he : G⁻¹.mulVec (G.mulVec δ) = G⁻¹.mulVec 0 := by rw [h]
  rwa [Matrix.mulVec_mulVec, Matrix.nonsing_inv_mul G hG, Matrix.one_mulVec,
       Matrix.mulVec_zero] at he

end

end WeakMoment
