# One exact mu_a conditional refresh: frozen diagnostic

This mathematical and decision contract is fixed before generating or assessing the hybrid cohort. It follows the first pure SS15 case0 development result. The pure SS15 cohort,including its failed physical gate,remains the baseline. This is a separately labeled **hybrid NPE-PFN/Gibbs** candidate,not pure NPE-PFN.

## Single intervention

Use exactly the saved50,000 complete supported21-dimensional physical vectors from the fixed SS15 case0,context4096,seed20261111 pilot. Independently refresh mu_a once per row from its exact full conditional given the other20 coordinates. Do not update any other coordinate,iterate the kernel,choose a seed from results,change observations or target order,expand/filter the context bank,train,simulate,or repair tails. No reference,held-out truth,observed value or fitted correction is an input to the kernel. Reference artifacts are consumed only after the hybrid cohort is fixed,for unchanged assessment.

The actual model has g=(mu_a,mu_b,mu_d0)~N(m,Sigma),independent conditional local blocks beta_i|g~N(A_i g+c_i,D_i),and independent truncated-Gaussian noise scales. Its likelihood depends on beta_i,sigma_i,not g. Therefore p(g|all locals,all noises,y)=p(g|all locals) is Gaussian,with

P=Sigma^-1+sum_i A_i^T D_i^-1 A_i,
eta=Sigma^-1 m+sum_i A_i^T D_i^-1(beta_i-c_i).

The scalar full conditional is mu_a|other20,y ~ N((eta_0-P_01*mu_b-P_02*mu_d0)/P_00,1/P_00). Coefficients are derived from the unchanged source model in Float64; rounded values are explanatory only. The conditional SD is approximately.1806481. It includes all six local a coefficients and the correlated-prior contribution from the other globals. Noise scales and observations cancel once the locals are fixed. All globals have unbounded support; noises and all other coordinates are retained byte for byte. No support rejection or trimming is needed after a finite valid refresh.

Use one private NumPy Generator(PCG64) stream from the existing derive_seed function with role `ss15_mu_a_gibbs_once_v1`,context seed20261111,case0,repetition0. Save its actual seed,source hashes,initial/final private-state hashes,and the50000 standard-normal draw descriptor. Default RNG state must remain unchanged. Return exactly50,000 complete finite supported physical vectors. Any invalid row fails the whole candidate; do not redraw or fill it.

## Decisions fixed before the result

The primary diagnostic asks whether replacing the learned mu_a conditional by the exact prior-derived one reduces its conservative marginal-CDF upper bound and absolute-mean-bias upper bound on this development case. Report either outcome,with no rerun or seed selection. This is a diagnostic hypothesis,not a new success threshold.

Scientific success still requires the entire original qualified-reference gate: every21 marginal and32 fixed joint CDF upper bound at most.10,every conservative mean-bias upper bound at most.20 reference SD,every SD-ratio interval contained in[.8,1.2],reference CDF precision at most.005,and50,000 finite supported complete vectors. Keep the original reference uncertainty and all rows. Because all other20 columns are identical,the existing b_m1,a_m3,b_m3 mean failures must remain identical; do not call improvement in mu_a full scientific success. One exact conditional kernel preserves the true posterior but does not turn an arbitrary approximate joint into the true posterior.

Inspect ordinary physical density and CDF overlays of raw33,pure SS15,hybrid and retained reference as appropriate,with full-count normalization and saved reference bins. Retain all failures. The fixed3-seed by8-case diagnostic grid remains required for the fixed pure SS15 baseline and this fixed paired refresh,without changing the kernel after inspecting further cases. Development cases are not untouched confirmation or SBC.

## Ranked alternatives

The scalar refresh is first because it directly tests the failed global conditional and changes only one coordinate without additional model calls. A joint three-global exact refresh is a larger separately declared alternative; it is not applied in this candidate. Global target permutation or an invertible prior/likelihood-derived coordinate change ranks afterward and would require fresh matching neural-conditional execution checks. Graph-local neural conditionals may address remaining local errors separately,but the actual factorization is independence between six whole local blocks conditional on all globals; direct graph neighbors do not justify arbitrary scalar feature deletion.

GPU optimization,retention timing,compilation and multi-GPU scaling remain deferred. This CPU diagnostic uses no new GPU allocation and does not alter the200 allocated GPU-hour cap.
