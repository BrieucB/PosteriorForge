Corrected all-FP32 backend/qualifier CPU test receipt
Built with PriorLabs-TabPFN.

The exact two recorded test files were executed in each locked environment.
All 39 tests passed in each; commands, raw logs, versions, timings and tested
source hashes before/after are retained. CUDA_VISIBLE_DEVICES was empty and
pytest caches were separated in scratch. No PFN/checkpoint inference, GPU
allocation, posterior draw, simulator world or reference integration occurred.
Fake SDK/model methods in unit tests are engineering fixtures, not model calls.
No tested source or runtime was edited by this receipt operation.

This is a CPU test receipt, not the GPU numerical qualification or a posterior
accuracy result. The separate GPU proof and original failed550 evidence retain
their own source, numerical status and accounting.
