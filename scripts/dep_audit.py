"""dep_audit.py — Dependency pin + minimum-age audit (XCUT-4).

Enforces the Atlas supply-chain policy (atlas-docs/02 §2):

1. Every declared dependency MUST be pinned with an exact ``==`` version.
   Ranges (``>=``, ``~=``, ``<``, ``*``), ``@`` URLs, or ``latest`` fail.
2. Every pinned version MUST be at least ``--min-age-days`` (default 14) days
   old at audit time, measured from its PyPI upload date.

The script reads dependencies from ``pyproject.toml`` (PEP 621 ``project``
table: ``dependencies`` + every ``optional-dependencies`` group). It exits
non-zero if any dependency violates either rule, printing one line per
violation so CI logs are actionable.

It ALSO audits ``.trunk/trunk.yaml`` (when present): every version-bearing
entry (``cli.version``, ``plugins.sources[].ref``, ``runtimes.enabled[]``,
``lint.enabled[]``) must be pinned, and the Trunk-managed ruff pin is
additionally age-checked against the same ``--min-age-days`` floor.

Network use
-----------
The age check queries ``https://pypi.org/pypi/<name>/json`` (read-only, no
auth). Pass ``--offline`` to skip the age check and validate pins only — use
this only in environments with no PyPI egress; CI runs the full audit.

Exit codes
----------
0   All dependencies pinned and old enough.
1   At least one violation (unpinned, too new, or unresolvable).
2   Usage / configuration error (e.g. pyproject.toml not found).
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

PYPI_JSON_URL = "https://pypi.org/pypi/{name}/json"
HTTP_TIMEOUT_SECONDS = 15


class DependencyError(Exception):
    """Raised when a dependency cannot be parsed or resolved."""


def _split_requirement(requirement: str) -> tuple[str, str]:
    """Split a requirement string into (name, exact_version).

    Accepts only exact ``name==version`` pins. Anything else raises
    ``DependencyError``. Extras (``name[extra]==x``) and environment markers
    (``name==x ; python_version >= '3.12'``) are tolerated; the marker is
    stripped before validation.
    """
    spec = requirement.split(";", 1)[0].strip()
    if not spec:
        raise DependencyError(f"empty requirement: {requirement!r}")

    if "==" not in spec:
        raise DependencyError(
            f"{spec!r} is not pinned with '==' (ranges/'latest'/URLs are forbidden)"
        )

    name_part, _, version_part = spec.partition("==")
    name = name_part.split("[", 1)[0].strip()
    version = version_part.strip()

    if not name:
        raise DependencyError(f"could not parse package name from {requirement!r}")
    if not version:
        raise DependencyError(f"empty version pin in {requirement!r}")
    # A trailing comparator after the version (e.g. '==1.0,<2') means a range.
    if any(token in version for token in (",", "<", ">", "*", "~", "!")):
        raise DependencyError(f"{spec!r} is a range, not an exact '==' pin")

    return name, version


def collect_dependencies(pyproject_path: Path) -> list[str]:
    """Return every declared dependency string from a pyproject.toml."""
    with pyproject_path.open("rb") as handle:
        data: dict[str, Any] = tomllib.load(handle)

    project = data.get("project")
    if not isinstance(project, dict):
        raise DependencyError(f"no [project] table in {pyproject_path}")
    project_table: dict[str, Any] = project

    requirements: list[str] = []

    deps = project_table.get("dependencies", [])
    if isinstance(deps, list):
        deps_list: list[Any] = deps
        requirements.extend(str(item) for item in deps_list)

    optional = project_table.get("optional-dependencies", {})
    if isinstance(optional, dict):
        optional_table: dict[str, Any] = optional
        for group in optional_table.values():
            if isinstance(group, list):
                group_list: list[Any] = group
                requirements.extend(str(item) for item in group_list)

    return requirements


def collect_trunk_pins(trunk_path: Path) -> list[tuple[str, str]]:
    """Return (label, version) for every version-bearing entry in .trunk/trunk.yaml.

    Covers cli.version, plugins.sources[].ref, runtimes.enabled[], lint.enabled[].
    Entries use 'name@version'; a bare entry (no '@') yields an empty version so
    the pin check flags it. (We enable no versionless Trunk built-ins.)
    """
    import yaml  # pyyaml is a declared project dependency

    with trunk_path.open("rb") as handle:
        loaded: Any = yaml.safe_load(handle)
    data: dict[str, Any] = loaded if isinstance(loaded, dict) else {}

    pins: list[tuple[str, str]] = []

    cli = data.get("cli")
    if isinstance(cli, dict):
        cli_table: dict[str, Any] = cli
        pins.append(("cli", str(cli_table.get("version", ""))))

    plugins = data.get("plugins")
    if isinstance(plugins, dict):
        plugins_table: dict[str, Any] = plugins
        sources: list[Any] = plugins_table.get("sources", []) or []
        for src in sources:
            if isinstance(src, dict):
                src_table: dict[str, Any] = src
                pins.append((f"plugin:{src_table.get('id', '?')}", str(src_table.get("ref", ""))))

    for section in ("runtimes", "lint"):
        node = data.get(section)
        if isinstance(node, dict):
            node_table: dict[str, Any] = node
            enabled: list[Any] = node_table.get("enabled", []) or []
            for item in enabled:
                name, _, ver = str(item).partition("@")
                pins.append((f"{section}:{name}", ver))

    return pins


def audit_trunk(
    trunk_path: Path,
    *,
    min_age_days: int,
    offline: bool,
    now: datetime | None = None,
) -> list[str]:
    """Audit .trunk/trunk.yaml: every tool pinned; ruff additionally age-checked."""
    current_time = now or datetime.now(UTC)
    violations: list[str] = []

    for label, version in collect_trunk_pins(trunk_path):
        if not version:
            violations.append(f"PIN: trunk {label} is not pinned to an exact version")
            continue
        # Only ruff is resolvable on PyPI; Trunk's other tools come from Trunk's
        # CDN / GitHub and cannot be PyPI-age-checked here.
        if label == "lint:ruff" and not offline:
            try:
                uploaded = _fetch_upload_date("ruff", version)
            except DependencyError as exc:
                violations.append(f"AGE: {exc}")
                continue
            age_days = (current_time - uploaded).days
            if age_days < min_age_days:
                violations.append(
                    f"AGE: ruff=={version} (Trunk) is {age_days}d old "
                    f"(< {min_age_days}d floor; uploaded {uploaded.date().isoformat()})"
                )

    return violations


def _fetch_upload_date(name: str, version: str) -> datetime:
    """Return the earliest upload datetime (UTC) for ``name==version`` on PyPI."""
    url = PYPI_JSON_URL.format(name=quote(name, safe=""))
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            payload: dict[str, Any] = json.load(response)
    except urllib.error.HTTPError as exc:
        raise DependencyError(f"PyPI returned HTTP {exc.code} for {name!r}") from exc
    except urllib.error.URLError as exc:
        raise DependencyError(f"could not reach PyPI for {name!r}: {exc.reason}") from exc

    releases = payload.get("releases", {})
    if not isinstance(releases, dict):
        raise DependencyError(f"malformed PyPI 'releases' for {name!r}")
    releases_table: dict[str, Any] = releases
    files = releases_table.get(version)
    if not isinstance(files, list) or not files:
        raise DependencyError(f"version {version!r} of {name!r} not found on PyPI")
    files_list: list[Any] = files

    upload_times: list[datetime] = []
    for file_info in files_list:
        if not isinstance(file_info, dict):
            continue
        file_table: dict[str, Any] = file_info
        raw = file_table.get("upload_time_iso_8601") or file_table.get("upload_time")
        if not isinstance(raw, str):
            continue
        normalized = raw.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        upload_times.append(parsed.astimezone(UTC))

    if not upload_times:
        raise DependencyError(f"no upload date recorded for {name}=={version}")

    return min(upload_times)


def audit(
    pyproject_path: Path,
    *,
    min_age_days: int,
    offline: bool,
    now: datetime | None = None,
) -> list[str]:
    """Audit one pyproject.toml. Return a list of violation messages (empty = ok)."""
    current_time = now or datetime.now(UTC)
    violations: list[str] = []

    for requirement in collect_dependencies(pyproject_path):
        try:
            name, version = _split_requirement(requirement)
        except DependencyError as exc:
            violations.append(f"PIN: {exc}")
            continue

        if offline:
            continue

        try:
            uploaded = _fetch_upload_date(name, version)
        except DependencyError as exc:
            violations.append(f"AGE: {exc}")
            continue

        age_days = (current_time - uploaded).days
        if age_days < min_age_days:
            violations.append(
                f"AGE: {name}=={version} is {age_days}d old "
                f"(< {min_age_days}d floor; uploaded {uploaded.date().isoformat()})"
            )

    return violations


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dep_audit.py",
        description="Fail if any dependency is unpinned or younger than the age floor.",
    )
    parser.add_argument(
        "pyproject",
        nargs="?",
        default="pyproject.toml",
        help="Path to pyproject.toml (default: ./pyproject.toml).",
    )
    parser.add_argument(
        "--min-age-days",
        type=int,
        default=14,
        help="Minimum dependency age in days (default: 14).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip the PyPI age check; validate exact pins only.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns exit code (0 ok, 1 violations, 2 usage error)."""
    args = _parse_args(argv)
    pyproject_path = Path(args.pyproject)

    if not pyproject_path.is_file():
        print(f"dep_audit.py: error: {pyproject_path} not found", file=sys.stderr)
        return 2

    try:
        violations = audit(
            pyproject_path,
            min_age_days=args.min_age_days,
            offline=args.offline,
        )
    except DependencyError as exc:
        print(f"dep_audit.py: error: {exc}", file=sys.stderr)
        return 2

    trunk_path = pyproject_path.parent / ".trunk" / "trunk.yaml"
    if trunk_path.is_file():
        try:
            violations.extend(
                audit_trunk(trunk_path, min_age_days=args.min_age_days, offline=args.offline)
            )
        except DependencyError as exc:
            print(f"dep_audit.py: error: {exc}", file=sys.stderr)
            return 2

    if violations:
        print(f"dep_audit.py: {len(violations)} violation(s) in {pyproject_path}:")
        for line in violations:
            print(f"  - {line}")
        return 1

    mode = "pins only (offline)" if args.offline else f"pins + {args.min_age_days}d age"
    print(f"dep_audit.py: OK — all dependencies pass [{mode}] in {pyproject_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
