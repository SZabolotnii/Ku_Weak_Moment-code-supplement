/-
# Schwartz kernels ⇒ weak moments of all orders exist

For a Schwartz kernel `φ ∈ 𝒮(ℝ)`, the integrand `x ↦ xʲ φ(x)` is integrable for
every `j`, so the weak moments `m_jᵂ = ∫ xʲ φ` are finite for all orders — the
analytic fact behind the degree-3 restriction to fast-decaying (Schwartz /
compact) kernels (paper §4.3 / §6).
-/
import Mathlib

namespace WeakMoment

open MeasureTheory

/-- Weak moments of all orders are finite for a Schwartz kernel. -/
theorem weak_moment_integrable (f : SchwartzMap ℝ ℝ) (j : ℕ) :
    Integrable (fun x : ℝ => x ^ j * f x) := by
  refine (SchwartzMap.integrable_pow_mul (μ := volume) f j).mono'
    (((continuous_pow j).mul f.continuous).aestronglyMeasurable) ?_
  filter_upwards with x
  simp [Real.norm_eq_abs, abs_mul, abs_pow]

end WeakMoment
