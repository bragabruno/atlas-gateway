# atlas-gateway — build scripts

Single source of truth for build & validation. Developers and CI run the **same**
scripts; `make ci` (or `./scripts/ci.sh`) runs the full gate locally, and the
Bitbucket pipeline calls the same per-stage scripts. Cross-repo guide:
[atlas-docs/07-build-system.md](../../atlas-docs/07-build-system.md).

| Script | Make target | What it does |
|---|---|---|
| `lint.sh` | `make lint` | dep age audit (XCUT-4) + Trunk(ruff lint+format) + pyright (strict) |
| `test.sh` | `make test` | offline unit tests (pytest; MockProvider + fakeredis) |
| `coverage.sh` | `make coverage` | pytest-cov gate (recommended; `ATLAS_COV_MIN`) |
| `build.sh` | `make build` | import smoke + OpenAPI contract export + drift check (ADR-014) |
| `docker.sh` | `make docker` | container build (no Dockerfile yet — discovered gap) |
| `infra.sh` | `make infra` | `helm lint` / `template` the `deploy/` chart |
| `security.sh` | `make security` | secret / CVE / fs scans (advisory; `ATLAS_SECURITY_STRICT=1`) |
| `ci.sh` | `make ci` | runs all of the above, in order |
| `local.sh` | `make local` | run the service locally (MockProvider) |

`lib/common.sh` + `lib/colors.sh` hold the shared helpers (logging, timing,
command checks, error trap). All scripts are bash with `set -Eeuo pipefail`,
shellcheck-clean, idempotent, and run on Linux + macOS. Stages that are N/A for
this repo, or whose tools are absent, print `↷ skip` and exit 0 — so the same
command works on a laptop and in CI.
