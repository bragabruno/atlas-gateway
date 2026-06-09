# syntax=docker/dockerfile:1
# atlas-gateway runtime image — multi-stage, non-root, pinned base.
# Base pinned exactly (atlas-docs/02 §2), matching the CI image. Runtime deps are
# the exact-pinned [project.dependencies] from pyproject.toml (no dev deps).

FROM python:3.12.13-slim-bookworm AS build
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
COPY . .
# Install the package + its pinned runtime deps into an isolated venv.
RUN python -m venv /venv \
 && /venv/bin/pip install --no-cache-dir .

FROM python:3.12.13-slim-bookworm AS runtime
# Non-root runtime user.
RUN groupadd --system app \
 && useradd --system --gid app --home-dir /app --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=build /venv /venv
COPY --from=build /app /app
ENV PATH="/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
USER app
EXPOSE 8000
# FastAPI app served by uvicorn on :8000. Config is env-driven (ATLAS_ prefix);
# with no ATLAS_REDIS_URL / provider keys the gateway runs Mock-only (model=mock,
# MockProvider — ADR-012). Real deployments inject ATLAS_* per-env via the Key
# Vault CSI mount (atlas-docs/04); the image ships no secrets.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
