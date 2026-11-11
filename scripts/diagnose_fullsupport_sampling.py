#!/usr/bin/env python3
"""CPU-only diagnostic of the installed TabPFN full-support distribution API.

Built with PriorLabs-TabPFN. This is a constructed API proof, not a new baseline,
a model evaluation, or a proposed implementation change. No checkpoint is loaded.
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
import math
import platform
import time
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tabpfn.model.bar_distribution import BarDistribution, FullSupportBarDistribution

ATTRIBUTION = "Built with PriorLabs-TabPFN"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=20000)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(20261111)
    borders = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.float64, device="cpu")
    probabilities = torch.tensor([0.9, 0.05, 0.05], dtype=torch.float64, device="cpu")
    logits = probabilities.log()
    distribution = FullSupportBarDistribution(borders)
    assert distribution.icdf.__func__ is BarDistribution.icdf
    assert "cdf" not in FullSupportBarDistribution.__dict__
    assert "icdf" not in FullSupportBarDistribution.__dict__
    left = distribution.halfnormal_with_p_weight_before(distribution.bucket_widths[0])
    right = distribution.halfnormal_with_p_weight_before(distribution.bucket_widths[-1])

    def density_cdf(x):
        x = torch.as_tensor(x, dtype=torch.float64)
        result = torch.empty_like(x)
        below = x < borders[1]
        middle = (x >= borders[1]) & (x < borders[-2])
        above = x >= borders[-2]
        result[below] = probabilities[0] * (1 - left.cdf(borders[1] - x[below]))
        result[middle] = probabilities[0] + probabilities[1] * (x[middle] - borders[1])
        result[above] = 1 - probabilities[-1] + probabilities[-1] * right.cdf(x[above] - borders[-2])
        return result

    def finite_bar_cdf(x):
        x = torch.as_tensor(x, dtype=torch.float64)
        widths = borders[1:] - borders[:-1]
        fractions = ((x[:, None] - borders[:-1]) / widths).clamp(0, 1)
        return (fractions * probabilities).sum(-1)

    analytic_full_mean = float(-probabilities[0] * left.mean + probabilities[1] * 0.5
                               + probabilities[2] * (1 + right.mean))
    analytic_full_second = float(probabilities[0] * left.variance + probabilities[0] * left.mean.square()
                                 + probabilities[1] / 3
                                 + probabilities[2] * (1 + 2 * right.mean + right.variance + right.mean.square()))
    uniform_means = (borders[:-1] + borders[1:]) / 2
    analytic_sampler_mean = float((probabilities * uniform_means).sum())
    uniform_seconds = (borders[:-1].square() + borders[:-1] * borders[1:] + borders[1:].square()) / 3
    sampler_sd = float(((probabilities * uniform_seconds).sum() - analytic_sampler_mean**2).sqrt())
    api_mean = float(distribution.mean(logits))
    start = time.perf_counter()
    samples = distribution.sample(logits.expand(args.samples, -1))
    sample_seconds = time.perf_counter() - start
    assert samples.device.type == "cpu"
    sample_mean = float(samples.double().mean())
    sample_mean_se = sampler_sd / math.sqrt(args.samples)
    assert abs(api_mean - analytic_full_mean) < 1e-12
    assert bool(((samples >= borders[0]) & (samples <= borders[-1])).all())
    assert abs(sample_mean - analytic_sampler_mean) < 6 * sample_mean_se
    assert abs(sample_mean - analytic_full_mean) > 20 * sample_mean_se

    points = torch.tensor([-2.0, -1.0, -8/9, 0.5, 2.0, 3.0], dtype=torch.float64)
    full_cdf = density_cdf(points)
    finite_cdf = finite_bar_cdf(points)
    empirical_cdf = torch.tensor([float((samples <= x).double().mean()) for x in points], dtype=torch.float64)
    quantile_p = 0.1
    api_quantile = float(distribution.icdf(logits, quantile_p))
    full_cdf_at_api_quantile = float(density_cdf(torch.tensor([api_quantile]))[0])
    api_cdf = distribution.cdf(logits[None, :], points)[0]
    assert api_cdf.shape == finite_cdf.shape
    assert torch.allclose(api_cdf.double(), finite_cdf)
    assert abs(full_cdf_at_api_quantile - quantile_p) > 0.3
    nll = distribution(logits.expand(len(points), -1), points)
    api_pdf = distribution.pdf(logits.expand(len(points), -1), points)
    density_from_nll = (-nll).exp()
    assert torch.allclose(api_pdf, nll.exp())
    assert not torch.allclose(api_pdf, density_from_nll)

    sdk_source = Path(inspect.getsourcefile(FullSupportBarDistribution)).resolve()
    result = {
        "schema": "posteriorforge.sdk_sampling_semantics.v1",
        "label": "Constructed CPU-only API diagnostic; not the original scientific acceptance gate",
        "attribution": ATTRIBUTION,
        "scope": {"checkpoint_loaded": False, "model_calls": 0, "gpu_hours": 0,
                  "scientific_pipeline_changes": False, "baseline_cause_demonstrated": False},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "tabpfn": version("tabpfn"), "device": "cpu"},
        "custody": {"script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__),
                    "sdk_source": str(sdk_source), "sdk_source_sha256": sha256(sdk_source),
                    "class_first_line": inspect.getsourcelines(FullSupportBarDistribution)[1],
                    "inherited_icdf_first_line": inspect.getsourcelines(BarDistribution.icdf)[1],
                    "forward_first_line": inspect.getsourcelines(FullSupportBarDistribution.forward)[1],
                    "sample_first_line": inspect.getsourcelines(FullSupportBarDistribution.sample)[1],
                    "mean_first_line": inspect.getsourcelines(FullSupportBarDistribution.mean)[1],
                    "pdf_first_line": inspect.getsourcelines(FullSupportBarDistribution.pdf)[1]},
        "constructed_distribution": {"borders": borders.tolist(), "probabilities": probabilities.tolist(),
                                     "endpoint_mass": float(probabilities[0] + probabilities[-1]),
                                     "half_normal_scale": float(left.scale),
                                     "seed": 20261111, "sample_count": args.samples},
        "confirmed_semantics": {
            "icdf_inherited_finite_uniform_interpolation": True,
            "cdf_inherited_finite_uniform_interpolation": True,
            "forward_and_mean_use_unbounded_half_normal_end_bins": True,
            "sample_uses_inherited_icdf": True,
            "sampling_and_density_mean_define_different_distributions": True,
            "pdf_exp_positive_nll_instead_of_negative_nll": True,
            "pdf_is_unused_by_original_baseline": True},
        "mean_comparison": {"api_full_support_mean": api_mean,
                            "analytic_full_support_mean": analytic_full_mean,
                            "analytic_full_support_sd": math.sqrt(analytic_full_second - analytic_full_mean**2),
                            "analytic_finite_sampler_mean": analytic_sampler_mean,
                            "empirical_sampler_mean": sample_mean,
                            "sampler_mean_monte_carlo_se": sample_mean_se,
                            "mean_error_vs_density_in_sampler_se": (sample_mean - analytic_full_mean) / sample_mean_se,
                            "sample_seconds_cpu": sample_seconds,
                            "empirical_min": float(samples.min()), "empirical_max": float(samples.max()),
                            "full_support_mass_below_left_finite_border": float(density_cdf(borders[:1])[0]),
                            "full_support_mass_above_right_finite_border": float(1 - density_cdf(borders[-1:])[0])},
        "fixed_logits_distance_formula": {
            "approximate_kolmogorov_distance": float(probabilities[[0, -1]].max() / 2),
            "approximate_total_variation_distance": float((probabilities[0] + probabilities[-1]) / 2),
            "assumptions": "The installed half-normal scale puts approximately half each endpoint mass outside its finite endpoint bar; finite floating-point median calibration causes only numerical differences. These are fixed-logit row distances before autoregressive propagation or prior support rejection."},
        "quantile_witness": {"requested_probability": quantile_p, "api_icdf": api_quantile,
                             "density_implied_cdf_at_api_icdf": full_cdf_at_api_quantile},
        "point_witnesses": [{"x": float(x), "density_implied_cdf": float(f), "api_finite_cdf": float(a),
                            "sampler_empirical_cdf": float(e), "density_from_negative_nll": float(d),
                            "api_pdf": float(p)}
                           for x, f, a, e, d, p in zip(points, full_cdf, api_cdf, empirical_cdf, density_from_nll, api_pdf)],
        "interpretation": {
            "confirmed_defect": "In TabPFN 2.0.9 FullSupportBarDistribution, forward/mean describe half-normal endpoint tails, but inherited icdf/cdf and sample describe finite uniform endpoint bars.",
            "baseline_causality": "This constructed proof does not measure actual endpoint probabilities for the released PosteriorForge checkpoint and queries. Baseline impact remains unknown until those probabilities and target-coordinate transformations are measured.",
            "direction": "It is not yet established whether restoring half-normal sampling would improve or worsen any original posterior gate metric.",
            "unused_pdf": "The pdf sign error is confirmed by direct API evaluation. The released PosteriorForge sample path does not call pdf; exclude it as a demonstrated baseline failure cause.",
            "timing": "The CPU time is for this constructed sample only and is not a full-pipeline performance attribution."}
    }
    (output / "science-fullsupport-semantics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    grid = np.linspace(-4, 4, 1201)
    full_curve = density_cdf(torch.tensor(grid)).numpy()
    finite_curve = finite_bar_cdf(torch.tensor(grid)).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), layout="constrained")
    axes[0].plot(grid, full_curve, label="forward/mean: half-normal tails", lw=2)
    axes[0].plot(grid, finite_curve, label="icdf/sample: finite uniform bars", lw=2, ls="--")
    axes[0].scatter(points.numpy(), empirical_cdf.numpy(), s=22, color="black", label=f"sample ECDF (n={args.samples:,})", zorder=5)
    axes[0].axvline(-1, color="gray", lw=0.7)
    axes[0].axvline(2, color="gray", lw=0.7)
    axes[0].set(xlabel="Constructed target coordinate", ylabel="CDF", ylim=(-0.02, 1.02))
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].bar(["density / API mean", "finite-bar expectation", "sample mean"], [api_mean, analytic_sampler_mean, sample_mean], color=["#377eb8", "#ff7f00", "#4daf4a"])
    axes[1].errorbar([2], [sample_mean], yerr=[3*sample_mean_se], fmt="none", ecolor="black", capsize=4, label="3 Monte Carlo SE")
    axes[1].set(ylabel="Mean of constructed distribution")
    axes[1].tick_params(axis="x", labelsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("SDK semantics diagnostic: known endpoint mass 0.95; no checkpoint or GPU", fontsize=11)
    fig.supxlabel(ATTRIBUTION + "  |  Constructed diagnostic, not the original gate", fontsize=9)
    fig.savefig(output / "science-fullsupport-semantics.png", dpi=180)
    plt.close(fig)
    files = {p.name: sha256(p) for p in sorted(output.glob("science-fullsupport-semantics.*"))}
    (output / "science-fullsupport-checksums.json").write_text(json.dumps(files, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(output), "sdk_sha256": result["custody"]["sdk_source_sha256"],
                      "mean_comparison": result["mean_comparison"], "quantile_witness": result["quantile_witness"]}, indent=2))


if __name__ == "__main__":
    main()

