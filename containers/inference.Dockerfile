# Built with PriorLabs-TabPFN. No checkpoint, credentials or private sources.
# Build from repository root: docker build -f containers/inference.Dockerfile -t posteriorforge:0.1.0 .
FROM python:3.11.13-slim-bookworm@sha256:86adf8dbadc3d6e82ee5dd2c74bec2e1c2467cdad47886280501df722372d2e1
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WORKDIR /opt/posteriorforge
COPY environments/inference.lock /opt/locks/inference.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /opt/locks/inference.lock
COPY . /opt/posteriorforge
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation .
# Run with a suitable NVIDIA host driver and externally mounted, verified model.
ENTRYPOINT ["posteriorforge"]
CMD ["doctor"]
