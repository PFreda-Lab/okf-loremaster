"""Refuse to ship a distribution that contains anything git does not track.

Hatchling already omits VCS-ignored files, so this asserts a property that currently
holds rather than fixing one that does not. It is here because the cost of the property
quietly breaking is a published `.env`: a build back-end upgrade, a stray `include` in
`pyproject.toml`, or a file added to the tree but never committed is enough, and PyPI
does not let you take a release back — a leaked key would have to be rotated, not
deleted. Every path in the sdist must be a tracked path, which is a stronger and simpler
claim than "no path is ignored".

The wheel is checked for the two things a build can get wrong in the other direction:
the annotated template has to be in it or `init` writes nothing for anyone who installed
from PyPI, and a real `.env` must never be.

    python scripts/check_dist.py dist/
"""

from __future__ import annotations

import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

# Written by the build back-end from `pyproject.toml`, so it is not in the tree and never
# will be. The only member of the sdist exempt from the tracked-path rule.
GENERATED = {"PKG-INFO"}

# Where `[tool.hatch.build.targets.wheel.force-include]` puts the root `.env.example`.
PACKAGED_TEMPLATE = "okf_loremaster/env.example"


def tracked_paths() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout
    return {line for line in out.splitlines() if line}


def sdist_members(path: Path) -> list[str]:
    """Sdist paths with the `name-version/` prefix every member carries stripped off."""
    with tarfile.open(path, "r:gz") as tar:
        names = [m.name for m in tar.getmembers() if m.isfile()]
    return [name.split("/", 1)[1] for name in names if "/" in name]


def check_sdist(path: Path, tracked: set[str]) -> list[str]:
    errors = []
    members = sdist_members(path)
    if not members:
        return [f"{path.name}: no files in the archive at all"]
    for member in sorted(members):
        if member in GENERATED or member in tracked:
            continue
        errors.append(f"{path.name}: ships '{member}', which git does not track")
    return errors


def check_wheel(path: Path) -> list[str]:
    errors = []
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    for name in names:
        # `.env.example` is the template and is fine; `.env` in any directory is not.
        if Path(name).name == ".env":
            errors.append(f"{path.name}: ships '{name}' — that file holds real secrets")
    if PACKAGED_TEMPLATE not in names:
        errors.append(
            f"{path.name}: missing '{PACKAGED_TEMPLATE}'. Without it `okf-loremaster init` "
            "writes no .env for anyone who installed from PyPI."
        )
    return errors


def main(argv: list[str]) -> int:
    dist = Path(argv[1] if len(argv) > 1 else "dist")
    sdists = sorted(dist.glob("*.tar.gz"))
    wheels = sorted(dist.glob("*.whl"))
    if not sdists or not wheels:
        print(f"expected an sdist and a wheel in {dist}/, found {len(sdists)} and {len(wheels)}")
        return 1

    tracked = tracked_paths()
    errors: list[str] = []
    for sdist in sdists:
        errors.extend(check_sdist(sdist, tracked))
    for wheel in wheels:
        errors.extend(check_wheel(wheel))

    if errors:
        print(f"{len(errors)} problem(s) with the distributions:")
        for error in errors:
            print(f"  {error}")
        return 1

    checked = ", ".join(p.name for p in [*sdists, *wheels])
    print(f"ok — every sdist path is tracked, no .env, template present ({checked})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
