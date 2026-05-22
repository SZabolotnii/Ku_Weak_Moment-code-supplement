import Lake
open Lake DSL

package WeakMoment where
  srcDir := "Lean"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.26.0"

lean_lib WeakMoment where
  roots := #[`WeakMoment]
