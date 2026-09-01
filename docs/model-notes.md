# Model notes

For decimal odds `o_i`, raw implied probability is `q_i = 1/o_i`. Because `sum(q_i)` includes bookmaker margin, it is not a fair probability.

Implemented methods:

- Proportional: `p_i = q_i / sum(q)`.
- Additive: `p_i = q_i - (sum(q)-1)/n`.
- Power: solve `sum(q_i^k)=1` and return `q_i^k`.
- Shin: solve the insider-share parameter `z` using the Shin transformation by bisection.

These are market-implied estimates, not predictions. A future model layer should keep chronological training/testing splits and report log loss, Brier score, calibration and uncertainty before changing production weights.

