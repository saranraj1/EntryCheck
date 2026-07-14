# ExplainCheck — reproducible research container
# Python 3.12, CPU-only, deterministic environment
#
# Build:  docker build -t explaincheck:dev .
# Run:    docker run --rm -v $(pwd)/artifacts:/app/artifacts explaincheck:dev \
#             uv run explaincheck pilot synthetic --config configs/pilot.yaml

FROM python:3.12-slim-bookworm

# System packages needed for scientific Python and XGBoost CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Copy dependency spec and lock file first for layer caching
COPY pyproject.toml uv.lock .python-version ./

# Install all dependencies (no dev extras in production image)
RUN uv sync --no-dev

# Copy source
COPY src/ src/
COPY configs/ configs/
COPY amendments/ amendments/

# Artifacts dir is expected to be mounted as a volume
RUN mkdir -p artifacts/pilot artifacts/exploratory artifacts/confirmatory

# Record environment at build time
RUN uv run python -c \
    "import sys, json, numpy, pandas, xgboost, shap, sklearn; \
     print(json.dumps({'python': sys.version, 'numpy': numpy.__version__, \
     'pandas': pandas.__version__, 'xgboost': xgboost.__version__, \
     'shap': shap.__version__, 'sklearn': sklearn.__version__}))" \
    > /app/build-environment.json

ENTRYPOINT ["uv", "run", "explaincheck"]
CMD ["--help"]
