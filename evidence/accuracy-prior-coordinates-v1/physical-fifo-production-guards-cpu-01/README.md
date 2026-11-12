# Physical Float64 FIFO production guard CPU evidence

Built with PriorLabs-TabPFN.

This package preserves the exact closed CPU receipt and four raw logs from
`work/accuracy-prior-coordinates-v1/physical-fifo-production-guards-cpu-01/`.
The receipt SHA-256 is
`a3105eabb3af93acf481dfa0bbc17455209bf77883a72bc3469201fd0edcc786`.
Packaging copied these bytes unchanged and did not import runtime modules or
rerun tests, models, simulators, reference integrations, cohort processing or GPU
work. The only Python operations used for packaging were standard-library file
copying, hashes, JSON serialization and AST literal extraction.

## Actual tested scope

These are the 13 focused source, reverse-AST, private-namespace, committed-ancestry
inventory, and exception-observer guard tests in
`tests/test_performance_prior_physical_fifo_v1.py`. They cover observation failure
without masking the primary exception, including source-read and secondary-status
write failures. They are separate from the shared helper's 11 inference tests and
from the archived full-success production fixture test. This package does not
claim either of those separate results, a completed posterior request, numerical
qualification, or posterior accuracy.

| Environment | Python | NumPy | SciPy | Focused test result | Recorded test time |
| --- | --- | --- | --- | --- | --- |
| inference | 3.11.13 | 1.26.4 | 1.12.0 | 13 passed | 1.07 s |
| reference | 3.11.13 | 2.2.6 | 1.15.3 | 13 passed | 1.14 s |

Both raw preflights report `Torch_imported=false`, exact current checkout bytes,
and exact committed HEAD bytes for the 146-source ancestry inventory. The pinned
inventory SHA-256 is
`68812fb23efc4f9ccbe1a2bd2041c791a9649e32ea71af457139cb52e904dacb`.
`runtime-source-inventory.json` packages the full literal inventory extracted from
the reviewed producer; extraction is not another guard execution or validation.
The inventory is a conservative superset of first-party runtime sources, not a
minimal dependency graph. Existing pinned external SDK/PyTorch guards remain
part of the producer ancestry and are not claimed as Git-managed binaries here.

## Pre-commit label and subsequent Root report

The raw receipt and both raw preflight logs deliberately retain
`own_producer_commit_guard_pending=true`. This static label records that the
receipt command checked the 146-source ancestry inventory and did not itself
execute the new producer's separate own-source commit guard. It must not be
silently rewritten as a post-commit observation.

Separately, Root reported committing the exact producer and focused tests as
`ae6566f` and then executing the actual full `require_sources()` guard with the
producer's own committed bytes plus all 146 ancestry sources: PASS, without
importing Torch or using a GPU. This paragraph attributes that result to Root's
reported tool execution. No independent post-commit artifact is fabricated or
included in this package.

## Bound source bytes

The closed receipt records identical before/after source hashes:

| Source | SHA-256 |
| --- | --- |
| `src/posteriorforge/execution_physical_fifo_v1.py` | `f8823ae75e7ae2eea6f1aa667af48ecf5c892450361b5abd490ccda36a28f7e0` |
| `src/posteriorforge/performance_prior_coordinates_v1.py` | `71e5e2906fd663534e56878de1fc4aff55093e3d950a13f348f57417ca7c01d3` |
| `src/posteriorforge/performance_prior_physical_fifo_v1.py` | `a30e0c15513eb0c1a6ed3c68f0781327edef4714a244a119a3a4d4c68f8780bf` |
| `tests/test_execution_physical_fifo_v1.py` | `81245d09a46a6d93615f8741fdda1894715a744070ab5cf07dbb6dc9c21f7d69` |
| `tests/test_performance_prior_physical_fifo_v1.py` | `59ffd4821206c332ed003e7f99992b80f8a7a6dc531f86d990bd3a17389a430d` |

`SHA256SUMS` covers the copied receipt, exact raw logs, extracted inventory and
this README. It excludes itself. The original failed pilot and all frozen neural,
codec, helper and sampler sources remain unchanged.
