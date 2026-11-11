"""Standalone Gaussian-physics references with one unknown absolute noise SD.

Each reference consumes a fixed marginal physics prior and its own readings.
The global prior is integrated into fixed local moments once; no global draws,
true globals, other readings, learned hierarchy or old posterior are used.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.integrate import quad
from scipy.special import expit, logsumexp, ndtr
from scipy.stats import truncnorm


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_deadline(deadline):
    require(deadline is None or time.monotonic() < deadline, "Standalone reference deadline exhausted")


class StandaloneGaussianNoiseReference:
    """p(theta,sigma|own y), theta Gaussian and sigma independently truncated."""

    def __init__(self, mean, covariance, design, offset, *, bounds=(.01, .5),
                 noise_mean=.255, noise_sd=.1225):
        self.mean, self.covariance, self.design, self.offset = (
            np.asarray(a, dtype=np.float64) for a in (mean, covariance, design, offset))
        p, m = len(self.mean), len(self.offset)
        require(p in (1, 3) and self.mean.shape == (p,) and self.covariance.shape == (p, p)
                and self.design.shape == (m, p) and self.offset.shape == (m,) and m >= p
                and all(np.isfinite(a).all() for a in (self.mean, self.covariance, self.design, self.offset)),
                "Finite independent reference dimensions required")
        require(np.allclose(self.covariance, self.covariance.T, atol=1e-14, rtol=0), "Physics covariance must be symmetric")
        self.prior_chol = np.linalg.cholesky(self.covariance)
        self.prior_precision = np.linalg.solve(self.covariance, np.eye(p))
        self.physics_dimension, self.observation_dimension = p, m
        self.lower, self.upper = map(float, bounds)
        self.noise_mean, self.noise_sd = float(noise_mean), float(noise_sd)
        require(0 < self.lower < self.upper and noise_sd > 0
                and np.isfinite([self.lower, self.upper, noise_mean, noise_sd]).all(), "Invalid independent SD prior")
        mass = ndtr((self.upper-noise_mean)/noise_sd)-ndtr((self.lower-noise_mean)/noise_sd)
        require(mass > 0, "Truncated SD prior has zero normalization")
        self.log_noise_normalizer = math.log(noise_sd*math.sqrt(2*math.pi)*mass)
        self.observation_mean = self.design @ self.mean + self.offset
        self.signal_covariance = self.design @ self.covariance @ self.design.T
        eigenvalues, self.eigenvectors = np.linalg.eigh(self.signal_covariance)
        tolerance = 64*np.finfo(np.float64).eps*max(1., np.linalg.norm(self.signal_covariance, ord=2))
        require(eigenvalues.min() >= -tolerance, "Signal covariance is not positive semidefinite")
        self.eigenvalues = np.maximum(eigenvalues, 0.)

    def noise_prior_logpdf(self, sigma):
        sigma = np.asarray(sigma, dtype=np.float64)
        result = -.5*((sigma-self.noise_mean)/self.noise_sd)**2-self.log_noise_normalizer
        return np.where((sigma >= self.lower) & (sigma <= self.upper), result, -np.inf)

    def _observations(self, observations):
        values = np.atleast_2d(np.asarray(observations, dtype=np.float64))
        require(values.ndim == 2 and values.shape[1] == self.observation_dimension
                and np.isfinite(values).all(), "Finite own readings required")
        return values

    def log_kernel(self, observations, sigma):
        """Gaussian-integrated physics likelihood times physical SD prior."""
        values = self._observations(observations)
        sigma = np.asarray(sigma, dtype=np.float64)
        require(sigma.ndim == 1 and np.isfinite(sigma).all() and np.all(sigma > 0), "Finite positive SD nodes required")
        residual = (values-self.observation_mean) @ self.eigenvectors
        variance = self.eigenvalues[None, :] + sigma[:, None]**2
        quadratic = np.einsum("bi,ni->bn", residual**2, 1/variance)
        return (-.5*(self.observation_dimension*math.log(2*math.pi)
                     + np.log(variance).sum(1)[None, :] + quadratic)
                + self.noise_prior_logpdf(sigma)[None, :])

    def quadrature(self, order):
        require(type(order) is int and order >= 16, "Quadrature order must be at least16")
        nodes, weights = np.polynomial.legendre.leggauss(order)
        width = self.upper-self.lower
        return self.lower+width*(nodes+1)/2, weights*width/2

    def log_evidence(self, observations, *, order=1024, chunk=256, deadline=None):
        values = self._observations(observations)
        sigma, weights = self.quadrature(order)
        parts = []
        for start in range(0, len(values), chunk):
            check_deadline(deadline)
            parts.append(logsumexp(self.log_kernel(values[start:start+chunk], sigma)+np.log(weights)[None, :], axis=1))
        return np.concatenate(parts)

    def scalar_integrals(self, observation, *, thresholds=(), deadline=None):
        """Independent adaptive control uses direct covariance solves, not eigenvalues."""
        values = self._observations(observation)
        require(len(values) == 1, "Adaptive control needs one own observed world")
        residual = values[0]-self.observation_mean
        m = self.observation_dimension
        def kernel(sigma):
            check_deadline(deadline)
            covariance = self.signal_covariance+sigma*sigma*np.eye(m)
            sign, logdet = np.linalg.slogdet(covariance)
            require(sign > 0, "Adaptive observation covariance must be positive definite")
            return float(self.noise_prior_logpdf(sigma))-.5*(m*math.log(2*math.pi)+logdet+residual @ np.linalg.solve(covariance, residual))
        grid = np.linspace(self.lower, self.upper, 257)
        logs = np.asarray([kernel(sigma) for sigma in grid])
        anchor, mode = float(logs.max()), float(grid[logs.argmax()])
        def integrate(power=0, upper=None):
            stop = self.upper if upper is None else float(upper)
            require(self.lower <= stop <= self.upper, "Adaptive CDF threshold outside SD support")
            if stop == self.lower:
                return 0., 0.
            return quad(lambda sigma: sigma**power*math.exp(kernel(sigma)-anchor), self.lower, stop,
                        epsabs=1e-11, epsrel=1e-11, limit=300,
                        points=[mode] if self.lower < mode < stop else None)
        integral, error = integrate()
        require(integral > 0 and error/integral < 1e-8, "Adaptive scalar evidence failed to converge")
        moments = [integrate(power)[0]/integral for power in (1, 2)]
        return {"log_evidence": math.log(integral)+anchor, "relative_error_estimate": error/integral,
                "noise_mean": moments[0], "noise_second_moment": moments[1],
                "noise_cdf": [integrate(upper=t)[0]/integral for t in thresholds]}

    def conditional_physics(self, sigma, observation):
        sigma = np.asarray(sigma, dtype=np.float64)
        values = self._observations(observation)
        require(sigma.ndim == 1 and np.all(sigma > 0) and len(values) == 1, "Single-world conditional reference required")
        precision = self.prior_precision[None, :, :]+(self.design.T @ self.design)[None, :, :]/sigma[:, None, None]**2
        covariance = np.linalg.inv(precision)
        rhs = (self.prior_precision @ self.mean)[None, :]+(self.design.T @ (values[0]-self.offset))[None, :]/sigma[:, None]**2
        mean = np.einsum("nij,nj->ni", covariance, rhs)
        return mean, (covariance+covariance.transpose(0, 2, 1))/2

    def log_joint(self, observations, targets, *, log_evidence=None, order=1024, deadline=None):
        values = self._observations(observations)
        targets = np.asarray(targets, dtype=np.float64)
        require(targets.shape == (len(values), self.physics_dimension+1) and np.isfinite(targets).all(), "Finite independent joint targets required")
        sigma = targets[:, -1]
        require(np.all((sigma >= self.lower) & (sigma <= self.upper)), "Joint target SD outside physical support")
        evidence = self.log_evidence(values, order=order, deadline=deadline) if log_evidence is None else np.asarray(log_evidence)
        if evidence.ndim == 0:
            evidence = np.full(len(values), float(evidence))
        require(evidence.shape == (len(values),) and np.isfinite(evidence).all(), "Independent log evidence shape differs")
        white = np.linalg.solve(self.prior_chol, (targets[:, :-1]-self.mean).T).T
        prior = -.5*(self.physics_dimension*math.log(2*math.pi)+(white**2).sum(1))-np.log(np.diag(self.prior_chol)).sum()
        residual = values-(targets[:, :-1] @ self.design.T+self.offset)
        likelihood = -.5*(self.observation_dimension*math.log(2*math.pi)
                          +2*self.observation_dimension*np.log(sigma)+(residual**2).sum(1)/sigma**2)
        return prior+self.noise_prior_logpdf(sigma)+likelihood-evidence

    def moments(self, observation, *, order=1024):
        sigma, weights = self.quadrature(order)
        logweights = self.log_kernel(observation, sigma)[0]+np.log(weights)
        mass = np.exp(logweights-logsumexp(logweights))
        means, covariances = self.conditional_physics(sigma, observation)
        mean = np.r_[mass @ means, mass @ sigma]
        second = np.r_[mass @ (means**2+covariances.diagonal(axis1=1, axis2=2)), mass @ sigma**2]
        variance = second-mean**2
        require(np.all(variance >= -1e-12), "Reference mixture moments have negative variance")
        return mean, np.sqrt(np.maximum(variance, 0))

    def noise_cdf_grid(self, observation, *, cells=4096, cell_order=16, log_evidence=None, deadline=None):
        values = self._observations(observation)
        require(len(values) == 1 and type(cells) is int and cells >= 16, "Single-world finite CDF grid required")
        check_deadline(deadline)
        grid = np.linspace(self.lower, self.upper, cells+1)
        nodes, weights = np.polynomial.legendre.leggauss(cell_order)
        width = (self.upper-self.lower)/cells
        sigma = (grid[:-1, None]+width*(nodes+1)/2).reshape(-1)
        evidence = float(self.log_evidence(values, deadline=deadline)[0]) if log_evidence is None else float(log_evidence)
        log_mass = self.log_kernel(values, sigma)[0]+np.tile(np.log(weights*width/2), cells)-evidence
        mass = np.exp(log_mass).reshape(cells, cell_order).sum(1)
        cdf = np.r_[0., np.cumsum(mass)]
        require(np.isfinite(cdf).all() and abs(cdf[-1]-1) < 1e-9 and np.all(np.diff(cdf) >= 0), "Full-support numerical noise CDF failed normalization")
        # Endpoint correction is limited to integration roundoff, recorded in
        # convergence evidence; no posterior mass is cropped or discarded.
        return grid, cdf/cdf[-1], float(cdf[-1]-1)

    def prepare_observed(self, observation, *, order=1024, initial_cells=4096,
                         max_cells=65536, cdf_tolerance=1e-7, deadline=None):
        evidence = float(self.log_evidence(observation, order=order, deadline=deadline)[0])
        cells = initial_cells
        low_grid, low_cdf, low_error = self.noise_cdf_grid(observation, cells=cells, log_evidence=evidence, deadline=deadline)
        history = []
        while cells*2 <= max_cells:
            high_grid, high_cdf, high_error = self.noise_cdf_grid(observation, cells=2*cells, log_evidence=evidence, deadline=deadline)
            midpoint_error = float(np.max(abs(high_cdf[1::2]-(low_cdf[:-1]+low_cdf[1:])/2)))
            node_error = float(np.max(abs(high_cdf[::2]-low_cdf)))
            history.append({"coarse_cells": cells, "fine_cells": 2*cells,
                            "coarse_linear_CDF_midpoint_error": midpoint_error,
                            "shared_node_CDF_delta": node_error,
                            "full_support_integral_error": high_error})
            if midpoint_error <= cdf_tolerance and node_error <= 1e-9:
                mean, sd = self.moments(observation, order=order)
                low_mean, low_sd = self.moments(observation, order=order//2)
                require(max(np.max(abs(mean-low_mean)), np.max(abs(sd-low_sd))) < 1e-8, "Observed reference moments failed doubled-order convergence")
                thresholds = np.linspace(self.lower, self.upper, 9)[1:-1]
                adaptive = self.scalar_integrals(observation, thresholds=thresholds, deadline=deadline)
                adaptive_cdf_delta = float(np.max(abs(np.interp(thresholds, high_grid, high_cdf)-adaptive["noise_cdf"])))
                require(abs(adaptive["log_evidence"]-evidence) < 1e-8
                        and adaptive_cdf_delta <= cdf_tolerance
                        and abs(adaptive["noise_mean"]-mean[-1]) < 1e-8
                        and abs(adaptive["noise_second_moment"]-(sd[-1]**2+mean[-1]**2)) < 1e-8,
                        "Observed adaptive evidence/noise CDF/moment controls failed")
                return {"log_evidence": evidence, "noise_grid": high_grid, "noise_cdf": high_cdf,
                        "mean": mean, "sd": sd, "convergence": {"status": "passed", "CDF_history": history,
                        "coarse_full_support_integral_error": low_error, "adaptive_controls": adaptive,
                        "adaptive_CDF_max_delta": adaptive_cdf_delta, "CDF_tolerance": cdf_tolerance,
                        "sampling": "IID uniforms inverted through continuous piecewise-linear fine-grid CDF, followed by exact Gaussian physics conditional on each SD. Numerical CDF interpolation is approximate; doubled-grid/adaptive checks are empirical controls, not a rigorous quadrature certificate."}}
            cells *= 2
            low_grid, low_cdf = high_grid, high_cdf
        raise ValueError("Observed noise CDF interpolation did not converge within the explicit cell limit")

    def sample_posterior(self, observation, prepared, *, draws, rng):
        require(type(draws) is int and draws > 1 and isinstance(rng, np.random.Generator), "Private IID reference RNG required")
        sigma = np.interp(rng.random(draws), prepared["noise_cdf"], prepared["noise_grid"])
        mean, covariance = self.conditional_physics(sigma, observation)
        chol = np.linalg.cholesky(covariance)
        physics = mean+np.einsum("nij,nj->ni", chol, rng.standard_normal((draws, self.physics_dimension)))
        result = np.column_stack((physics, sigma))
        require(np.isfinite(result).all(), "Nonfinite standalone reference samples")
        return result

    def sample_prior(self, *, draws, rng):
        physics = self.mean+rng.standard_normal((draws, self.physics_dimension)) @ self.prior_chol.T
        sigma = truncnorm.ppf(rng.random(draws), (self.lower-self.noise_mean)/self.noise_sd,
                              (self.upper-self.noise_mean)/self.noise_sd, loc=self.noise_mean, scale=self.noise_sd)
        return np.column_stack((physics, sigma))


def decode_local(standardized, mean, scale, *, bounds=(.01, .5)):
    values = np.asarray(standardized, dtype=np.float64)*np.asarray(scale)+np.asarray(mean)
    values[:, -1] = bounds[0]+(bounds[1]-bounds[0])*expit(values[:, -1])
    return values


def encode_local(physical, mean, scale, *, bounds=(.01, .5)):
    values = np.asarray(physical, dtype=np.float64).copy()
    require(np.all((values[:, -1] > bounds[0]) & (values[:, -1] < bounds[1])), "Density encoding needs SD strictly inside support")
    values[:, -1] = np.log(values[:, -1]-bounds[0])-np.log(bounds[1]-values[:, -1])
    return (values-np.asarray(mean))/np.asarray(scale)


def local_log_jacobian(physical, scale, *, bounds=(.01, .5)):
    """log|d physical/d standardized|, preserving full local affine scales."""
    sigma = np.asarray(physical, dtype=np.float64)[:, -1]
    scale = np.asarray(scale, dtype=np.float64)
    require(np.all(scale > 0) and np.all((sigma > bounds[0]) & (sigma < bounds[1])), "Interior SD and positive local scales required")
    return np.log(scale).sum()+np.log(sigma-bounds[0])+np.log(bounds[1]-sigma)-math.log(bounds[1]-bounds[0])


def references_from_problem(problem):
    """Use exact declared marginal moments, including all physics cross-covariances."""
    base = problem.prior.base
    result = {}
    for index, (local, forward) in enumerate(zip(base.conditional_blocks, problem.observation_model.blocks)):
        require(local.dataset_id == forward.dataset_id == problem.datasets[index].dataset_id,
                "Standalone prior/forward dataset order differs")
        mean = local.population_map @ base.global_mean+local.offset
        covariance = local.covariance+local.population_map @ base.global_covariance @ local.population_map.T
        require(problem.prior.noise_gaussian_priors[index] == (.255, .1225)
                and problem.prior.bounds == (.01, .5), "Declared independent physical SD prior differs")
        result[local.dataset_id] = StandaloneGaussianNoiseReference(mean, covariance, forward.design, forward.offset)
    require(len(result) == 6, "Exactly six standalone references required")
    return result
