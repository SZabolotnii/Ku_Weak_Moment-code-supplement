/-
Weak-Moment Stochastic-Polynomial Estimators — verified deterministic backbone.

This library formalizes the deterministic (analytic / algebraic) core of the
weak-moment regression estimators, complementing the classical asymptotic
theory written in the paper. See `docs/` and `paper/main.tex`.

Modules:
  * `WeakMoment.Score`     — bounded / redescending scores; bounded CF sine-score.
  * (further modules added incrementally: Schwartz moment existence,
     degree-1 ↔ M-estimator identities, κ^w orthogonality optimality,
     local identifiability rank.)
-/
import WeakMoment.Score
import WeakMoment.Foundations
import WeakMoment.Moments
