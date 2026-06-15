#!/usr/bin/env bash
# infra.sh — deploy-manifest validation: lint + render the in-repo Helm chart.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
# shellcheck source=scripts/lib/colors.sh
source scripts/lib/colors.sh
# shellcheck source=scripts/lib/common.sh
source scripts/lib/common.sh
trap 'on_err "$LINENO" "$?"' ERR

if [[ ! -d deploy ]]; then
  skip "infra" "no deploy/ chart in this repo"
  exit 0
fi
if ! has_cmd helm; then
  skip "helm lint" "helm not installed"
  exit 0
fi
run "helm lint" helm lint deploy

# secrets.tenantId is injected from a Terraform output at deploy time
# (atlas-docs/04), so a structural render check supplies a dummy UUID and renders
# each per-env overlay (base values.yaml leaves env-injected fields empty).
render_dummy=(--set secrets.tenantId=00000000-0000-0000-0000-000000000000)
rendered_any=0
for vals in deploy/values-dev.yaml deploy/values-prod.yaml; do
  [[ -f "$vals" ]] || continue
  rendered_any=1
  run "helm template ($(basename "$vals"))" \
    helm template atlas-gateway deploy -f "$vals" "${render_dummy[@]}"
done
[[ "$rendered_any" -eq 1 ]] || run "helm template (render)" helm template atlas-gateway deploy

if has_cmd kubeconform; then
  run "kubeconform (dev manifest schema)" \
    bash -c 'helm template atlas-gateway deploy -f deploy/values-dev.yaml --set secrets.tenantId=00000000-0000-0000-0000-000000000000 | kubeconform -strict -summary -ignore-missing-schemas'
else
  skip "kubeconform" "not installed (recommended: K8s manifest schema validation)"
fi
log_ok "deploy manifests valid"
