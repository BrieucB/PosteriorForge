# Case 2: signed conditional-mean attribution

Built with PriorLabs-TabPFN. This bounded CPU-only diagnosis uses all 50,000 saved accepted rows from the **original neural parent**, before its joint-global refresh. It makes no new posterior draws, neural predictions, reference integrations, candidate changes or gate waivers. Case 2 remains a physical model failure in the [fixed grid](../corrected-sigma-grid-01/README.md).

Source checks establish the mechanical-2 Gaussian conditional prior with mean `population_map @ G + prior_offset`, covariance `diag(.45²,.70²,.25²)`, design columns `[x,x²,1]` for ten fixed x values, zero forward offset, and independent observation noise with absolute SD `sigma_m2`. For each original row, the diagnostic solves

`Vpost = (V0^-1 + X.T @ X / sigma²)^-1`

`m = Vpost @ (V0^-1 @ m0(G) + X.T @ (y-forward_offset) / sigma²)`.

The actual prior/design/offsets come from the frozen source. Conditional prior covariance is used, rather than covariance after marginalizing globals. Covariance-space and original prior-plus-likelihood log-ratio checks independently verify the calculation. Original parent globals and noise scales are retained; refreshed globals do not construct the primary decomposition.

| Parameter | Total signed error / ref SD | Accepted-sampler conditional residual | Conditional-mean mixing discrepancy |
| --- | ---: | ---: | ---: |
| a_m2 | +0.2461431642 | +0.1399176650 | +0.1062254991 |
| b_m2 | -0.2726798519 | -0.1818453383 | -0.0908345136 |

This is the identity `E_q[beta]-reference_mean = E_q[beta-m] + E_q[m]-reference_mean`. Both components reinforce the signed error. The second term measures a conditional-mean expectation under the accepted original joint distribution of globals and noise; it does not separate global error from noise error.

Approximate paired-row Monte Carlo standard errors for (residual, mixing, total) are `(0.00394004, 0.00218132, 0.00460003)` for a_m2 and `(0.00419155, 0.00208719, 0.00479549)` for b_m2. Their covariance is retained, rather than treating the two terms as independent. The JSON includes per-component approximate CLT 95% intervals under iid accepted-row and finite-variance assumptions, and intervals enlarged by the separately propagated declared numerical reference errors. These are diagnostic approximations, not rigorous coverage statements or new accuracy criteria. Declared reference mean-error envelopes are about 0.000006 and 0.000009 reference SD; reference-SD relative errors are about 0.000012 and 0.000018.

**The first term is an accepted-sampler conditional residual, not an identified neural-only bias.** It can include FP16, cached-execution and finite-precision head/sampling effects as well as whole-row selection. The separate finite case-0 numerical proof supplies no case-2 mean-error bound. A matching original-FP32 failing-case control should be considered before making a causal backbone claim; no such extra GPU run is performed here. Whole-row finite/support rejection removed 299 of 50,442 checked rows (about 0.593%). Another 143 accepted rows were trimmed and 758 generated FIFO rows were unused. Rejected vectors were not saved. Their contribution cannot be recovered from accepted rows alone, and a small rejection rate provides no finite mean-error bound for unbounded Gaussian coordinates/residuals. No rejection-causes-bias claim is made.

The exact [diagnostic JSON](diagnostic.json) records source/artifact hashes, reference custody and uncertainty, paired covariance, numerical checks, counts and limits. Its raw SHA256 is `bd5929843f369d2d5a72d2e2b29e809504b41c92a285391cbdf448c61c85ec5b`. The [script](../../../scripts/diagnose_case2_conditional_mean.py) and [13 deterministic CPU tests](../../../tests/test_case2_conditional_mean.py) are public; tests pass in both environments. The scientific grid continues unchanged.


## Independent raw-row replay

A separate root audit reauthenticated the saved original neural parent, source and case-2 reference and replayed every scientific diagnostic field exactly. It also computed the conditional mean in covariance space over **all 50,000 original rows**, independently of the primary precision-space mean helper: `m=m0+D X.T (X D X.T+sigma² I)^-1(y-offset-Xm0)`. The signed residual/mixing/total means agree within 0.000000000000006 reference SD and paired Monte Carlo standard errors within 0.000000000000000004 reference SD. This goes beyond checking identities in the delivered JSON. [The verification artifact](independent-replay.json) records the separate implementation, input hashes and zero draws/neural GPU calls/reference integrations. It took 0.544 CPU seconds and imported no Torch. These checks verify the calculation; the rejection/execution causal limitations above remain unchanged.
