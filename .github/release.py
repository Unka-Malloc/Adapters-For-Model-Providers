#!/usr/bin/env python3
"""Generated release train helper; regenerate it instead of editing this copy.
.github/release.lock records the source inputs that produced it.

Every subcommand runs inside the repository checkout, writes step outputs to
$GITHUB_OUTPUT and a readable report to $GITHUB_STEP_SUMMARY. RELEASE_KIND
says what started the run: `train` (the weekly schedule) or `manual` (Run
workflow).
"""
from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
LEVELS = ("patch", "minor", "major")
SAFE_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*")
VERSION = re.compile(r"(\d+)\.(\d+)\.(\d+)")
DEP_KEYS = ("dependencies", "dev-dependencies", "build-dependencies")
POLL_SECONDS = int(os.environ.get("TRAIN_POLL_SECONDS", "300"))
# The sparse-index base. CARGO_REGISTRY_INDEX overrides it (a mirror or a
# fixture); publish.yml's shell probe shares the same default — keep them
# in sync.
INDEX = (os.environ.get("CARGO_REGISTRY_INDEX") or "https://index.crates.io").rstrip("/")


class TrainError(Exception):
    pass


def git(*args: str, cwd: pathlib.Path | None = None) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if done.returncode != 0:
        raise TrainError(f"git {' '.join(args)}: {done.stderr.strip()}")
    return done.stdout.strip()


def git_ok(*args: str) -> bool:
    return subprocess.run(["git", *args], capture_output=True).returncode == 0


def emit(**values: str) -> None:
    lines = [f"{key}={value}" for key, value in values.items()]
    target = os.environ.get("GITHUB_OUTPUT")
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    else:
        print("\n".join(lines))


def report(markdown: str) -> None:
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(markdown.rstrip() + "\n\n")
    else:
        print(markdown, file=sys.stderr)


def parse_version(text: str) -> tuple[int, int, int] | None:
    match = VERSION.fullmatch(text)
    return tuple(int(part) for part in match.groups()) if match else None


def show(version: tuple[int, int, int] | None) -> str:
    return ".".join(map(str, version)) if version else "none"


def bump(version: tuple[int, int, int], level: str) -> tuple[int, int, int]:
    major, minor, patch = version
    if level == "major":
        return (major + 1, 0, 0)
    if level == "minor":
        return (major, minor + 1, 0)
    return (major, minor, patch + 1)


def parse_offset(text: str) -> dt.timezone:
    match = re.fullmatch(r"([+-])(\d{2}):(\d{2})", text)
    if not match:
        raise TrainError(f"invalid UTC offset {text!r}")
    delta = dt.timedelta(hours=int(match[2]), minutes=int(match[3]))
    return dt.timezone(-delta if match[1] == "-" else delta)


def last_cutoff(now: dt.datetime, weekday: str, hhmm: str, offset: str) -> dt.datetime:
    """Latest cutoff instant at or before `now`; `hhmm` is the last included minute."""
    zone = parse_offset(offset)
    hour, minute = (int(part) for part in hhmm.split(":"))
    local = now.astimezone(zone)
    back = (local.weekday() - WEEKDAYS.index(weekday)) % 7
    day = local.date() - dt.timedelta(days=back)
    cutoff = dt.datetime.combine(day, dt.time(hour, minute), zone) + dt.timedelta(minutes=1)
    if cutoff > now:
        cutoff -= dt.timedelta(days=7)
    return cutoff


def index_path(name: str) -> str:
    name = name.lower()
    if len(name) <= 2:
        return f"{len(name)}/{name}"
    if len(name) == 3:
        return f"3/{name[0]}/{name}"
    return f"{name[:2]}/{name[2:4]}/{name}"


def published_versions(name: str) -> set[tuple[int, int, int]]:
    """Every version the registry holds, yanked included: a yanked number is still taken."""
    request = urllib.request.Request(
        f"{INDEX}/{index_path(name)}", headers={"User-Agent": "release-train"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode()
    except urllib.error.HTTPError as error:
        if error.code in (403, 404):
            return set()
        raise
    found = set()
    for line in body.splitlines():
        if line.strip():
            version = parse_version(json.loads(line)["vers"])
            if version:
                found.add(version)
    return found


def dependency_specs(manifest: dict):
    tables = [manifest, manifest.get("workspace", {})]
    tables += list(manifest.get("target", {}).values())
    for table in tables:
        for key in DEP_KEYS:
            yield from table.get(key, {}).items()


def sibling_requirements(repo: pathlib.Path, siblings: set[str]) -> dict[str, set[str]]:
    """Version requirements this repository places on crates of sibling checkouts."""
    parent = repo.resolve().parent
    wanted: dict[str, set[str]] = {}
    for relative in git("ls-files", "*Cargo.toml", cwd=repo).splitlines():
        manifest_path = repo / relative
        if manifest_path.name != "Cargo.toml":
            continue
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        for name, spec in dependency_specs(manifest):
            if not isinstance(spec, dict) or "path" not in spec:
                continue
            target = (manifest_path.parent / spec["path"]).resolve()
            try:
                top = target.relative_to(parent).parts[0]
            except ValueError:
                top = None
            if top == repo.resolve().name:
                continue
            if top not in siblings:
                raise TrainError(
                    f"{relative}: `{name}` points at {spec['path']}, which is not a declared "
                    "sibling; regenerate this workflow"
                )
            wanted.setdefault(spec.get("package", name), set()).add(spec.get("version", "*"))
    return wanted


def resolve_published(wanted: dict[str, set[str]]):
    """Resolve the requirements against crates.io; (versions, None) or (None, reason)."""
    with tempfile.TemporaryDirectory() as scratch:
        root = pathlib.Path(scratch)
        (root / "src").mkdir()
        (root / "src" / "lib.rs").write_text("", encoding="utf-8")
        lines = [
            "[package]",
            'name = "release-train-gate"',
            'version = "0.0.0"',
            'edition = "2021"',
            "publish = false",
            "",
            "[dependencies]",
        ]
        for position, (package, requirements) in enumerate(sorted(wanted.items())):
            requirement = ", ".join(sorted(requirements))
            lines.append(
                f"gate{position} = {{ package = {json.dumps(package)}, "
                f"version = {json.dumps(requirement)} }}"
            )
        (root / "Cargo.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
        # Cargo discovers .cargo/config.toml by walking up from its cwd —
        # inside the candidate checkout a committed config (a credential
        # provider is executable) would load under the release identity.
        # Run from the scratch root instead, as cargo_metadata does.
        done = subprocess.run(
            ["cargo", "generate-lockfile", "--manifest-path", str(root / "Cargo.toml")],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if done.returncode != 0:
            return None, done.stderr.strip().splitlines()[-1:] or ["cargo failed"]
        lock = tomllib.loads((root / "Cargo.lock").read_text(encoding="utf-8"))
    resolved: dict[str, tuple[int, int, int]] = {}
    for package in lock.get("package", []):
        version = parse_version(package["version"])
        if version and version > resolved.get(package["name"], (-1, -1, -1)):
            resolved[package["name"]] = version
    return resolved, None


def remote_tags(slug: str) -> dict[tuple[int, int, int], str]:
    listing = git("ls-remote", "--tags", "--refs", f"https://github.com/{slug}.git")
    tags = {}
    for line in listing.splitlines():
        name = line.split("refs/tags/", 1)[-1]
        version = parse_version(name[1:]) if name.startswith("v") else None
        if version:
            tags[version] = name
    return tags


def cmd_candidate(args: argparse.Namespace) -> int:
    mode = os.environ.get("RELEASE_MODE", "dry-run")
    kind = os.environ.get("RELEASE_KIND", "train")
    # The publish gate binds the candidate to the default-branch run head,
    # so selection must live on that same branch: a side line would pick
    # candidates the gate is bound to reject, and the release's back-merge
    # lands on this branch too.
    branch = args.default_branch
    cutoff = ""
    if args.override:
        if not SAFE_REF.fullmatch(args.override):
            raise TrainError(f"refusing candidate {args.override!r}")
        remote = f"refs/remotes/origin/{args.override}"
        ref = remote if git_ok("rev-parse", "--verify", "-q", remote) else args.override
        sha = git("rev-parse", "--verify", f"{ref}^{{commit}}")
        if not git_ok("merge-base", "--is-ancestor", sha, f"refs/remotes/origin/{branch}"):
            raise TrainError(f"{args.override} is not on {branch}")
        source = f"requested candidate {args.override}"
    elif kind == "manual":
        sha = git("rev-parse", "--verify", f"refs/remotes/origin/{branch}^{{commit}}")
        source = f"head of {branch}"
    else:
        moment = last_cutoff(dt.datetime.now(dt.timezone.utc), args.weekday, args.time,
                             args.utc_offset)
        cutoff = moment.isoformat()
        sha = git("rev-list", "-1", "--first-parent", f"--before={cutoff}",
                  f"refs/remotes/origin/{branch}")
        if not sha:
            raise TrainError(f"{branch} has no commit before {cutoff}")
        source = f"last commit on {branch} before {cutoff}"
    git("checkout", "-q", "--detach", sha)
    report(f"### Candidate\n\n`{sha[:12]}`: {source} ({kind}, {mode}).")
    emit(sha=sha, branch=branch, cutoff=cutoff)
    return 0

def cmd_deps(args: argparse.Namespace) -> int:
    mode = os.environ.get("RELEASE_MODE", "dry-run")
    siblings = json.loads(os.environ.get("SIBLINGS") or "[]")
    if not siblings:
        emit(ready="true")
        return 0
    wanted = sibling_requirements(pathlib.Path.cwd(), {s["dir"] for s in siblings})
    deadline = time.monotonic() + args.wait_minutes * 60
    while True:
        resolved, problem = resolve_published(wanted)
        if resolved is not None:
            break
        if time.monotonic() >= deadline:
            lines = "\n".join(f"- `{p}` {', '.join(sorted(r))}" for p, r in sorted(wanted.items()))
            verdict = "would wait for" if mode == "dry-run" else "is still waiting for"
            report(f"### Dependencies\n\nThe train {verdict} unpublished dependencies:\n\n"
                   f"{lines}\n\ncargo: {' '.join(problem)}")
            emit(ready="false")
            return 0 if mode == "dry-run" else 1
        print(f"dependencies not on crates.io yet: {' '.join(problem)}", flush=True)
        time.sleep(POLL_SECONDS)
    rows, refs = [], {}
    # Refs are keyed by sibling directory, not position: a reordered
    # declaration can never pair a checkout with the wrong resolved tag.
    for sibling in siblings:
        tags = remote_tags(sibling["slug"])
        versions = sorted(resolved[c] for c in sibling["crates"] if c in resolved)
        if versions:
            if versions[-1] not in tags:
                raise TrainError(f"{sibling['slug']} published {show(versions[-1])} without a "
                                 f"v{show(versions[-1])} tag")
            ref = tags[versions[-1]]
        elif tags:
            ref = tags[max(tags)]
        else:
            raise TrainError(f"{sibling['slug']} has no release tag to build against")
        refs[f"ref_{sibling['dir']}"] = ref
        rows.append(f"| {sibling['dir']} | `{ref}` |")
    report("### Dependencies\n\n| Sibling | Checked out |\n| --- | --- |\n" + "\n".join(rows))
    emit(ready="true", **refs)
    return 0


def labelled_level(slug: str, branch: str, candidate: str, tag: str) -> tuple[str, list[str]]:
    since = git("log", "-1", "--format=%cs", tag)
    # GitHub search returns at most 1000 rows for any query; there is no
    # pagination past that ceiling. A full page therefore means the merged-PR
    # set may be truncated — and a truncated set can silently drop the
    # release:minor/major labels that decide the bump. Fail closed rather
    # than default to a lower bump: releasing manually with an explicit
    # level is the safe path when the interval overflows.
    listing = subprocess.run(
        ["gh", "pr", "list", "--repo", slug, "--base", branch, "--state", "merged",
         "--search", f"merged:>={since}", "--limit", "1000",
         "--json", "number,labels,mergeCommit"],
        capture_output=True, text=True,
    )
    if listing.returncode != 0:
        raise TrainError(f"listing pull requests failed: {listing.stderr.strip()}")
    pulls = json.loads(listing.stdout)
    if len(pulls) >= 1000:
        raise TrainError(
            f"merged pull requests into {branch} since {since} reached the "
            "1000-row search ceiling; the label set may be truncated. Release "
            "manually with an explicit level — do not guess a lower bump.")
    level, numbers = "patch", []
    for pull in pulls:
        merged = (pull.get("mergeCommit") or {}).get("oid")
        if not merged or not git_ok("merge-base", "--is-ancestor", merged, candidate):
            continue
        if git_ok("merge-base", "--is-ancestor", merged, tag):
            continue
        numbers.append(f"#{pull['number']}")
        for label in pull.get("labels", []):
            name = label.get("name", "")
            if name.startswith("release:") and name[8:] in LEVELS:
                level = max(level, name[8:], key=LEVELS.index)
    return level, numbers


def choose_version(
    manifest: tuple[int, int, int],
    published: dict[str, set[tuple[int, int, int]]],
    tags: set[tuple[int, int, int]],
    level: str,
    requested: tuple[int, int, int] | None = None,
) -> tuple[tuple[int, int, int], bool]:
    """The version to release and whether it resumes an interrupted upload.

    `requested` is a version a person asked for; it must still be a number
    nobody has taken, unless it is the one an interrupted upload left behind.
    """
    on_registry = set().union(*published.values())
    top_published = max(on_registry, default=None)
    top_tag = max(tags, default=None)
    untagged = top_published is not None and (top_tag is None or top_published > top_tag)
    if untagged:
        holders = sum(top_published in versions for versions in published.values())
        if holders < len(published):
            if requested not in (None, top_published):
                raise TrainError(f"{show(top_published)} is only partly on crates.io; finish it "
                                 f"before releasing {show(requested)}")
            return top_published, True
        if top_tag is not None:
            raise TrainError(
                f"{show(top_published)} is on crates.io without a v{show(top_published)} tag; "
                "tag the commit it was published from before the next train"
            )
    base = max([*tags, *on_registry], default=None)
    if requested is not None:
        if base is not None and requested <= base:
            raise TrainError(f"{show(requested)} is not above {show(base)}, the highest number "
                             "already taken")
        return requested, False
    return (manifest if base is None else max(bump(base, level), manifest)), False

def cargo_metadata(repo: pathlib.Path) -> dict:
    """Workspace manifest data only: cargo runs from an empty directory with
    --manifest-path so a checkout's .cargo/config.toml — a credential
    provider is executable — can never load beside the release identity.
    --no-deps suffices: member manifests carry their own dependencies."""
    with tempfile.TemporaryDirectory() as scratch:
        try:
            done = subprocess.run(
                ["cargo", "metadata", "--format-version", "1", "--no-deps",
                 "--manifest-path", str(repo / "Cargo.toml")],
                cwd=scratch, capture_output=True, text=True, check=True,
            )
        except subprocess.CalledProcessError as error:
            tail = (error.stderr or "").strip().splitlines()
            raise TrainError(
                f"cargo metadata failed: {tail[-1] if tail else 'unknown error'}")
    return json.loads(done.stdout)


def cmd_version(args: argparse.Namespace) -> int:
    slug = os.environ.get("GITHUB_REPOSITORY", "")
    kind = os.environ.get("RELEASE_KIND", "train")
    metadata = cargo_metadata(pathlib.Path.cwd())
    members = set(metadata["workspace_members"])
    packages = [p for p in metadata["packages"] if p["id"] in members]
    publishable = [p for p in packages if p["publish"] is None or "crates-io" in p["publish"]]
    if not publishable:
        raise TrainError("the workspace has no publishable package")
    manifest = max(parse_version(p["version"]) or (0, 0, 0) for p in publishable)
    published = {p["name"]: published_versions(p["name"]) for p in publishable}
    on_registry = set().union(*published.values())
    tags = {}
    for name in git("tag", "--list", "v*").splitlines():
        version = parse_version(name[1:])
        if version:
            tags[version] = name
    released = [v for v in tags if v in on_registry]
    last_tag = tags[max(released)] if released else None
    notes, pulls = [], []
    if last_tag:
        changed = not git_ok("diff", "--quiet", last_tag, args.candidate)
    else:
        changed = True
        notes.append("No tag matches a published version, so the whole history counts as new.")
    if tags and max(tags) not in on_registry:
        notes.append(f"`{tags[max(tags)]}` is tagged but not on crates.io; its number is "
                     "skipped.")
    requested = None
    if kind == "manual":
        level = os.environ.get("INPUT_LEVEL") or "patch"
        if level not in LEVELS:
            raise TrainError(f"unknown level {level!r}")
        text = (os.environ.get("INPUT_VERSION") or "").strip().removeprefix("v")
        if text:
            requested = parse_version(text)
            if requested is None:
                raise TrainError(f"version {text!r} is not <major>.<minor>.<patch>")
            level = "exact"
    else:
        level = "patch"
        if changed and last_tag and args.branch:
            level, pulls = labelled_level(slug, args.branch, args.candidate, last_tag)
    upcoming, resume = choose_version(manifest, published, set(tags), level, requested)
    if resume:
        changed = True
        notes.append(f"{show(upcoming)} is only partly on crates.io; this run finishes it.")
    done = sorted(name for name, versions in published.items() if upcoming in versions)
    fresh = sorted(name for name, versions in published.items() if not versions)
    if fresh:
        notes.append("Never published, so Trusted Publishing cannot upload them until a first "
                     "manual upload: " + ", ".join(f"`{n}`" for n in fresh) + ".")
    release = changed and len(done) < len(publishable)
    if changed and not release:
        notes.append(f"Every crate already holds {show(upcoming)} on crates.io.")
    excludes = sorted({p["name"] for p in packages if p not in publishable} | set(done))
    needs_set = any(parse_version(p["version"]) != upcoming for p in publishable)
    source = {"train": "pull request labels", "manual": "Run workflow input"}
    summary = [
        "### Version",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Last released tag | {last_tag or 'none'} |",
        f"| Manifest version | {show(manifest)} |",
        f"| Changed since release | {'yes' if changed else 'no'} |",
        f"| Bump level | {level} from {source.get(kind, kind)} "
        f"({', '.join(pulls) or 'no labelled pull requests' if kind == 'train' else '-'}) |",
        f"| Next version | {show(upcoming) if release else 'no release'} |",
    ]
    report("\n".join(summary + ([""] + [f"- {n}" for n in notes] if notes else [])))
    emit(
        release="true" if release else "false",
        next=show(upcoming),
        needs_set="true" if needs_set else "false",
        excludes=" ".join(excludes),
    )
    return 0

def publishable_members() -> dict[str, dict]:
    metadata = cargo_metadata(pathlib.Path.cwd())
    members = set(metadata["workspace_members"])
    packages = {p["name"]: p for p in metadata["packages"] if p["id"] in members}
    return {n: p for n, p in packages.items()
            if p["publish"] is None or "crates-io" in p["publish"]}


def cmd_order(args: argparse.Namespace) -> int:
    packages = publishable_members()
    excludes = set((os.environ.get("EXCLUDES") or "").split())
    targets = sorted(set(packages) - excludes)
    ordered, pending = [], dict.fromkeys(targets)
    while pending:
        ready = [n for n in pending
                 if not any(d["name"] in pending
                            for d in packages[n]["dependencies"] if d["kind"] != "dev")]
        if not ready:
            raise TrainError(f"dependency cycle among {', '.join(sorted(pending))}")
        ordered.extend(sorted(ready))
        for name in ready:
            del pending[name]
    for name in ordered:
        p = packages[name]
        print(f"{name}\t{p['version']}")
    return 0


def cmd_freshness(args: argparse.Namespace) -> int:
    """Every publishable crate's src/**/*.rs `last_verified: "YYYY-MM-DD"`
    must sit inside the window — a stale native-capability table means the
    card's claims were not re-checked against the shipped binary."""
    window = dt.timedelta(days=args.window_days)
    today = dt.datetime.now(dt.timezone.utc).date()
    checked, stale = 0, []
    for name, package in sorted(publishable_members().items()):
        src = pathlib.Path(package["manifest_path"]).parent / "src"
        dates = [
            dt.date.fromisoformat(match.group(1))
            for path in src.rglob("*.rs") if src.exists()
            for match in [re.search(r'last_verified:\s*"(\d{4}-\d{2}-\d{2})"',
                                    path.read_text())]
            if match
        ]
        if not dates:
            continue
        checked += 1
        oldest = min(dates)
        if today - oldest > window:
            stale.append(f"{name}: last_verified {oldest} "
                         f"({(today - oldest).days}d > {args.window_days}d)")
    if stale:
        report("\n".join(f"- {entry}" for entry in stale))
        raise TrainError(f"{len(stale)} native capability table(s) stale: "
                         + "; ".join(stale))
    emit(checked=str(checked))
    report(f"{checked} native capability table(s) within "
           f"{args.window_days}d.")
    return 0


SECTION = re.compile(r"^\[\[?\s*(?P<name>.+?)\s*\]?\]\s*(?:#.*)?$")
ENTRY = re.compile(
    r"^(?P<key>'[^']+'|\"[^\"]+\"|[A-Za-z0-9_-]+)"
    r"(?:\.(?P<sub>'[^']+'|\"[^\"]+\"|[A-Za-z0-9_-]+))?\s*=\s*(?P<value>.*)$"
)
QUOTED = r'"[^"]*"|\'[^\']*\''
VERSION_FIELD = re.compile(rf"version\s*=\s*(?:{QUOTED})")
ASSIGN_VALUE = re.compile(rf"(=\s*)(?:{QUOTED})")
WORKSPACE_FIELD = re.compile(r"\bworkspace\s*=\s*true\b")
PACKAGE_FIELD = re.compile(rf"package\s*=\s*({QUOTED})")
STRING_VALUE = re.compile(rf"^\s*({QUOTED})")


def section_of(line: str) -> str | None:
    """The table name a `[a.b]`/`[[a.b]]` header opens, or None."""
    text = line.strip()
    if not text.startswith("["):
        return None
    found = SECTION.match(text)
    if not found:
        return None
    return found.group("name").rstrip("]").strip()


def key_segments(section: str) -> list[str]:
    """`target.'cfg(unix)'.dependencies` -> `[target, cfg(unix), dependencies]`:
    dots inside quoted segments stay inside the segment."""
    parts, current, quote = [], [], ""
    for ch in section:
        if quote:
            if ch == quote:
                quote = ""
            else:
                current.append(ch)
        elif ch in "\"'":
            quote = ch
        elif ch == ".":
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    parts.append("".join(current).strip())
    return parts


def dep_section(segments: list[str]) -> list[str] | None:
    """The dependency key path when the section opens a dependency table —
    `[workspace.dependencies]` and `[target.<cfg>.dependencies]` qualify —
    or a sub-table inside one (`[dependencies.<name>]`); None otherwise."""
    if segments[:1] == ["workspace"]:
        segments = segments[1:]
    elif segments[:1] == ["target"] and len(segments) >= 3:
        segments = segments[2:]
    if segments and segments[0] in DEP_KEYS:
        return segments[1:]
    return None


def open_depth(text: str) -> int:
    """`{`/`[` minus `}`/`]` outside quoted strings — a multi-line inline
    table or array keeps the entry it started on open until it closes."""
    depth, quote, escaped = 0, "", False
    for ch in text:
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
    return depth


def gather(rest: list[str], index: int) -> tuple[str, int]:
    """The entry text starting at rest[index], continued while its inline
    table or array stays open; returns (text, index of its last line)."""
    block = [rest[index]]
    while open_depth("".join(block)) > 0 and index + 1 < len(rest):
        index += 1
        block.append(rest[index])
    return "".join(block), index


def string_value(text: str) -> str | None:
    """The bare-string scalar a value starts with, unquoted, or None."""
    found = STRING_VALUE.match(text or "")
    return found.group(1)[1:-1] if found else None


def unrecognized(path: pathlib.Path, section: str, line: str) -> TrainError:
    return TrainError(
        f"{path}: [{section}] entry {line.strip()!r} is not a dependency shape "
        "set-version recognizes")


def entries(path: pathlib.Path, section: str, rest: list[str]):
    """The dependency-table entries inside `rest` as (text, key, sub,
    value): `text` is the entry's whole block — a multi-line inline table
    or array is one row — `key`/`sub` the dotted-key segments unquoted and
    `value` the raw value text of the entry's first line. Blank lines and
    comments yield (text, None, None, None) so callers can pass them
    through; a line the entry grammar cannot parse fails closed — it
    might be hiding a member pin."""
    index = 0
    while index < len(rest):
        stripped = rest[index].strip()
        text, index = gather(rest, index)
        index += 1
        if not stripped or stripped.startswith("#"):
            yield text, None, None, None
            continue
        found = ENTRY.match(stripped)
        if found is None:
            raise unrecognized(path, section, text)
        yield (text, found.group("key").strip("'\""),
               (found.group("sub") or "").strip("'\""), found.group("value"))


def rewrite_package(path: pathlib.Path, section: str, body: list[str],
                    next_v: str) -> tuple[list[str], int]:
    """`[package]`/`[workspace.package]`: the member's own version field.
    A field already at `next_v` is a no-op, not a failure — rerunning the
    bump on a partly or fully bumped workspace must pass clean."""
    out, edited = [], 0
    for line in body:
        if re.match(r"version\s*=", line.strip()):
            found = VERSION_FIELD.search(line)
            if found is None:
                raise TrainError(
                    f"{path}: [{section}] version is not a plain string")
            new = (line[:found.start()] + f'version = "{next_v}"'
                   + line[found.end():])
            if new != line:
                line, edited = new, edited + 1
        out.append(line)
    return out, edited


def rewrite_dep_table(path: pathlib.Path, section: str, body: list[str],
                      members: set[str], next_v: str) -> tuple[list[str], int]:
    """A flat dependency table: bare strings, inline tables (multi-line
    included) and dotted keys. A member requirement in a form this cannot
    rewrite fails instead of silently keeping the old pin."""
    header, rest = body[0], body[1:]
    # Dotted-key renames (`foo.package = "real"`) may sit below the entries
    # they qualify — collect them before rewriting.
    renames: dict[str, str] = {}
    for _, key, sub, value in entries(path, section, rest):
        if sub == "package":
            renamed = string_value(value)
            if renamed:
                renames[key] = renamed
    out, edited = [header], 0
    for text, key, sub, value in entries(path, section, rest):
        if key is None:
            out.append(text)
            continue
        if sub:
            if sub == "version" and renames.get(key, key) in members:
                before = text
                text, count = ASSIGN_VALUE.subn(f'\\g<1>"{next_v}"', text, count=1)
                if not count:
                    raise TrainError(
                        f"{path}: [{section}] `{key}.version` is not a plain string")
                edited += text != before
            out.append(text)
            continue
        if WORKSPACE_FIELD.search(text):
            out.append(text)
            continue
        package = PACKAGE_FIELD.search(text)
        if (package.group(1)[1:-1] if package else key) not in members:
            out.append(text)
            continue
        before = text
        if VERSION_FIELD.search(text):
            text = VERSION_FIELD.sub(f'version = "{next_v}"', text, count=1)
        elif value.lstrip().startswith(("\"", "'")):
            text = ASSIGN_VALUE.sub(f'\\g<1>"{next_v}"', text, count=1)
        elif "{" in text:
            opening = re.search(r"\{(\s*)", text)
            assert opening is not None
            gap = opening.group(1)
            closing = text[opening.end(1):].lstrip().startswith("}")
            comma = "" if closing else ","
            insert = (f'version = "{next_v}"{comma}' + gap
                      if "\n" in gap else f'version = "{next_v}"{comma} ')
            text = text[:opening.end(1)] + insert + text[opening.end(1):]
        else:
            raise TrainError(
                f"{path}: [{section}] member dependency `{key}` is in a form "
                "set-version cannot rewrite")
        edited += text != before
        out.append(text)
    return out, edited


def rewrite_dep_subtable(path: pathlib.Path, section: str, dep_key: str,
                         body: list[str], members: set[str],
                         next_v: str) -> tuple[list[str], int]:
    """`[dependencies.<dep_key>]` (and the namespaced variants): the member
    pin lives on `version =` lines inside the table, and `package =` may
    rename the dependency. A path-only member table gains an explicit pin,
    matching what the flat `{ path = ... }` form gets."""
    header, rest = body[0], body[1:]
    package = None
    for _, key, sub, value in entries(path, section, rest):
        if key == "package" and not sub:
            package = string_value(value)
    if (package or dep_key) not in members:
        return body, 0
    out, edited, pinned, inherited = [header], 0, False, False
    for text, key, sub, value in entries(path, section, rest):
        if key is None:
            out.append(text)
            continue
        if key == "version" and not sub:
            before = text
            text, count = ASSIGN_VALUE.subn(f'\\g<1>"{next_v}"', text, count=1)
            if not count:
                raise TrainError(
                    f"{path}: [{section}] member `{dep_key}` pins a version "
                    "set-version cannot rewrite")
            pinned = True
            edited += text != before
        elif (key == "version" and sub == "workspace") \
                or (key == "workspace" and WORKSPACE_FIELD.search(text)):
            inherited = True
        out.append(text)
    if not pinned and not inherited:
        out.insert(1, f'version = "{next_v}"\n')
        edited += 1
    return out, edited


def rewrite_section(path: pathlib.Path, section: str | None, body: list[str],
                    members: set[str], next_v: str) -> tuple[list[str], int]:
    if section is None:
        return body, 0
    segments = key_segments(section)
    if segments in (["package"], ["workspace", "package"]):
        return rewrite_package(path, section, body, next_v)
    subs = dep_section(segments)
    if subs is None:
        return body, 0
    if not subs:
        return rewrite_dep_table(path, section, body, members, next_v)
    if len(subs) > 1:
        if subs[0] in members:
            raise TrainError(
                f"{path}: [{section}] nests under member dependency `{subs[0]}` "
                "in a form set-version cannot rewrite")
        return body, 0
    return rewrite_dep_subtable(path, section, subs[0], body, members, next_v)


def rewrite_manifest(path: pathlib.Path, members: set[str], next_v: str) -> int:
    """Move a manifest to `next_v`: `[package]`/`[workspace.package]` version
    and every dependency-table entry whose package names a workspace member
    — bare strings, inline tables, dotted keys and `[<table>.<member>]`
    sub-tables, under `[dev-]`/`[build-]`/`[workspace.]`/`[target.<cfg>.]`
    variants alike — so the coordinated bump keeps manifests internally
    consistent and the lockfile update afterwards resolves against the new
    member versions. A member requirement in a shape the rewriter cannot
    update stops the train instead of leaving a stale pin that would first
    surface inside the publishing job."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    # Segment the document into (section, body) pairs: a dependency
    # sub-table's `package =` rename can sit lines away from its pin, so
    # rewriting needs the whole section in view.
    segments: list[tuple[str | None, list[str]]] = []
    name: str | None = None
    body: list[str] = []
    for line in lines:
        found = section_of(line)
        if found is not None:
            segments.append((name, body))
            name, body = found, [line]
        else:
            body.append(line)
    segments.append((name, body))
    out: list[str] = []
    edited = 0
    for name, body in segments:
        rewritten, count = rewrite_section(path, name, body, members, next_v)
        out.extend(rewritten)
        edited += count
    if edited:
        path.write_text("".join(out), encoding="utf-8")
    return edited


def set_version(repo: pathlib.Path, next_v: str) -> int:
    """cargo_metadata-driven rewrite, data only: every member manifest and
    the workspace root — where [workspace.dependencies] pins member
    requirements — so no path dep is left requiring the old numbers.
    Returns the number of manifests rewritten. The bump is then proven
    against a fresh metadata read — the rewriter is hand-rolled text
    surgery, so what it believes it changed is a claim, not evidence."""
    metadata = cargo_metadata(repo)
    member_ids = set(metadata["workspace_members"])
    packages = [p for p in metadata["packages"] if p["id"] in member_ids]
    members = {p["name"] for p in packages}
    files = {pathlib.Path(p["manifest_path"]) for p in packages}
    files.add(pathlib.Path(metadata["workspace_root"]) / "Cargo.toml")
    rewritten = sum(1 for path in sorted(files)
                    if rewrite_manifest(path, members, next_v))
    verify_bump(repo, members, next_v)
    return rewritten


def verify_bump(repo: pathlib.Path, members: set[str], next_v: str) -> None:
    """The bump is proven by cargo metadata, not by what the rewriter
    reported: every publishable member must sit at next_v and every
    dependency requirement naming a member must pin it. A member pin in a
    form the rewriter cannot see surfaces here as a stopped train, not as
    a shipped inconsistency inside the publishing job."""
    metadata = cargo_metadata(repo)
    member_ids = set(metadata["workspace_members"])
    wanted = parse_version(next_v)
    problems: list[str] = []
    for package in metadata["packages"]:
        if package["id"] not in member_ids:
            continue
        if package["publish"] is None or "crates-io" in package["publish"]:
            if parse_version(package["version"]) != wanted:
                problems.append(f"{package['name']} is at {package['version']}")
        for dep in package["dependencies"]:
            if dep["name"] in members and dep["req"].removeprefix("^") != next_v:
                problems.append(
                    f"{package['name']} requires {dep['name']} {dep['req']}"
                    f" instead of {next_v}")
    if problems:
        raise TrainError(
            f"set-version could not confirm the bump to {next_v}: "
            + "; ".join(problems))


def cmd_set_version(args: argparse.Namespace) -> int:
    if parse_version(args.version) is None:
        raise TrainError(f"set-version wants <major>.<minor>.<patch>, got {args.version!r}")
    rewritten = set_version(pathlib.Path.cwd(), args.version)
    report(f"### Version bump\n\nSet every workspace member and member requirement "
           f"to `{args.version}` ({rewritten} manifest(s) rewritten; the bump "
           "is verified against cargo metadata).")
    return 0


# --- Release-environment preflight ----------------------------------------
#
# The publish gate asks GitHub whether the `release` environment carries
# the shape the OIDC job relies on — required reviewers plus a deployment
# branch rule limited to the default branch — and refuses only what an
# answer can prove: a `release` environment absent from a complete,
# readable listing (GitHub would auto-create one unprotected), no
# required reviewers, or an all-branches deployment rule. Repository
# read access normally suffices for both endpoints, so permission walls
# are the exceptional case; anything the gate cannot positively verify
# is 'unverified' — a rehearsal annotates it and proceeds (the job-entry
# check is server-side anyway), while a live bind fails closed: a
# possibly absent environment would run unprotected. Remedies for a
# live unverified stop are a token with Environments read stored in the
# RELEASE_ENV_READ_TOKEN secret (the step binds GH_TOKEN to it or falls
# back to github.token), or the recorded operator exception — the
# repository variable ALLOW_UNVERIFIED_RELEASE_ENV=1 — which exempts
# only unverified answers where the environment's existence is itself
# proven; it can never cover a proven refusal, and it can never wave a
# possibly absent environment into existence. Either remedy is named in
# the log and step summary when it fires.

# Remedy vocabulary — one source so the workflow comments, the log and
# the annotations cannot drift apart.
TOKEN_REMEDY = ("store a token with Environments read in the "
                "RELEASE_ENV_READ_TOKEN secret")
OPT_IN_VAR = "ALLOW_UNVERIFIED_RELEASE_ENV"
JOB_ENTRY = ("; the environment's own protection still applies at job "
             "entry")


def _unverified_remedy(status: int, exists: bool) -> str:
    """The one remedy that can actually unblock this unverified answer:
    a transient failure wants a retry, proven existence can also take
    the recorded opt-in, and unproven existence accepts only a token —
    the opt-in can never wave a possibly absent environment."""
    if status == 0 or status >= 500:
        return ("the environment endpoints could not be read — retry "
                "the run; a stored token or the recorded opt-in cannot "
                "substitute for an answer")
    if exists:
        return (f"{TOKEN_REMEDY}, or set the repository variable "
                f"{OPT_IN_VAR}=1 to proceed unverified")
    return (f"{TOKEN_REMEDY} so the gate can verify — the recorded "
            "opt-in cannot cover a possibly absent environment")


ENV_GUIDE = ("required reviewers plus a deployment-branch rule limited to "
             "the default branch — create it under Settings → Environments → "
             "release, then bind crates.io Trusted Publishing to it")

LINK_NEXT = re.compile(r'<([^>]+)>\s*;\s*rel="next"')


def gh_get(path: str) -> tuple[int, object, object]:
    """One GET against the repository's REST API: a `path` starting with
    http is used as-is (a pagination Link target), anything else resolves
    under /repos/<slug>. Returns (status, payload, headers) — status 0 on
    a transport failure, payload None whenever the body cannot be read."""
    url = (path if path.startswith("http")
           else f"{os.environ.get('GITHUB_API_URL', 'https://api.github.com')}"
                f"/repos/{os.environ['GITHUB_REPOSITORY']}{path}")
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            try:
                payload = json.loads(response.read().decode())
            except (ValueError, UnicodeDecodeError):
                payload = None
            return response.status, payload, response.headers
    except urllib.error.HTTPError as error:
        return error.code, None, error.headers
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, None, None


def link_next(header: str | None) -> str | None:
    """The rel="next" URL out of a Link header, or None."""
    if not header:
        return None
    found = LINK_NEXT.search(header)
    return found.group(1) if found else None


def gh_paginated(path: str, key: str) -> tuple[int, list, bool]:
    """A paginated list endpoint walked to completion — `per_page=100`,
    then each Link rel="next" (a `page=` walk when no Link header
    arrives) until the collected rows reach the payload's own
    total_count. Returns (status, rows, complete): complete stays False
    the moment any page is unreadable or the walk cannot finish — a
    partial enumeration proves nothing and never stands in for absence."""
    sep = "&" if "?" in path else "?"
    url, rows, total, page = f"{path}{sep}per_page=100", [], None, 1
    for _ in range(50):
        status, payload, headers = gh_get(url)
        if status != 200 or not isinstance(payload, dict):
            return status, rows, False
        batch = payload.get(key)
        if not isinstance(batch, list):
            return 200, rows, False
        rows.extend(batch)
        if total is None and isinstance(payload.get("total_count"), int):
            total = payload["total_count"]
        if total is not None and len(rows) >= total:
            return 200, rows, True
        if total is None and len(batch) < 100:
            return 200, rows, True  # a short page without a count ends the list
        nxt = link_next(headers.get("Link") if headers else None)
        if nxt:
            url = nxt
        else:
            page += 1
            url = f"{path}{sep}per_page=100&page={page}"
    return 200, rows, False


def needed_evidence(status: int, env: object) -> set[str]:
    """Which sub-fetches a (status, payload) pair calls for — the driver
    gathers only the facets the payload itself references: `listing` to
    settle the existence question whenever the environment endpoint
    itself cannot be read, `branch`/`policies` when the deployment
    branch policy enables those mechanisms."""
    if status != 200:
        return {"listing"}
    if not isinstance(env, dict):
        return set()
    policy = env.get("deployment_branch_policy")
    if not isinstance(policy, dict):
        return set()
    needed = set()
    if policy.get("protected_branches"):
        needed.add("branch")
    if policy.get("custom_branch_policies"):
        needed.add("policies")
    return needed


def status_detail(status: int) -> str:
    """How an evidence fetch failed, phrased once for every verdict."""
    return "transport error" if status == 0 else f"HTTP {status}"


def preflight_verdict(
    status: int,
    env: object,
    evidence: dict,
    default: str,
) -> tuple[str, str, bool]:
    """The `release` environment decision from fetched evidence alone —
    no I/O, so every row of the contract is exercised from a table.

    `evidence` carries the sub-fetch results `needed_evidence` asked the
    driver for: "listing" is (status, complete, names) of a paginated
    /environments walk, "policies" is (status, complete, names) of a
    paginated deployment-branch-policies walk, and "branch" is
    (status, protected) of the default branch. Returns (verdict,
    message, existence_proven); verdict is "pass", "unverified" or
    "refuse" — the caller maps refuse and live unverified to a hard
    stop, with one recorded exception: a live bind may proceed on an
    unverified answer by opt-in only when existence_proven is true (the
    environment exists; only its settings are unreadable). The
    job-entry reassurance and the remedy text are the caller's to
    render so they can never disagree with the stop decision."""
    if status != 200:
        # Existence is conclusive only from a complete, readable
        # enumeration: the endpoint's own error cannot tell a missing
        # environment from a permission wall. A listing that names
        # `release` proves existence even when the endpoint refuses;
        # one that omits it proves absence — and a missing environment
        # auto-creates unprotected — regardless of the status seen.
        list_status, complete, names = evidence.get("listing", (0, False, []))
        if list_status == 200 and complete:
            if "release" not in names:
                return "refuse", (
                    "this repository has no `release` environment — GitHub "
                    "would create an unprotected one when the release job "
                    "starts, silently removing the approval gate"), False
            detail = (f"HTTP {status}" if status != 404 else
                      "its own endpoint answered HTTP 404")
            return "unverified", (
                f"the environments listing names `release` yet "
                f"{detail} — settings treated as unreadable"), True
        detail = (
            "the environments listing could not be read to completion"
            if list_status == 200
            else f"listing: {status_detail(list_status)}")
        return "unverified", (
            f"the `release` environment answered "
            f"{status_detail(status)} and existence could not be proven "
            f"({detail})"), False
    if not isinstance(env, dict):
        return "unverified", (
            "the `release` environment returned an unreadable payload"), True

    skipped: list[str] = []
    # Key absent → a facet this payload does not expose (skipped, then
    # warned); key present → the facet is asserted. A present
    # protection_rules without a required_reviewers rule carrying
    # reviewers is a misconfiguration, refused outright.
    if "protection_rules" not in env:
        skipped.append("required reviewers (payload omits protection_rules)")
    elif not any(
        isinstance(rule, dict)
        and rule.get("type") == "required_reviewers"
        and rule.get("reviewers")
        for rule in env.get("protection_rules") or []
    ):
        return "refuse", ("the `release` environment has no required "
                          "reviewers"), True
    if "deployment_branch_policy" not in env:
        skipped.append("deployment-branch rule (payload omits "
                       "deployment_branch_policy)")
    else:
        policy = env.get("deployment_branch_policy")
        if policy is None:
            # null is GitHub's documented spelling for "every branch may
            # deploy" — a set policy with no restriction — not an omitted
            # setting, so it is refused, not skipped.
            return "refuse", (
                "the `release` environment lets every branch deploy — "
                "deployment_branch_policy reads null (no restriction)"), True
        if not isinstance(policy, dict):
            skipped.append("deployment-branch rule (unreadable "
                           "deployment_branch_policy)")
        elif not policy.get("protected_branches") \
                and not policy.get("custom_branch_policies"):
            return "refuse", ("the `release` environment lets every "
                              "branch deploy"), True
        else:
            if policy.get("protected_branches"):
                branch_status, protected = evidence.get("branch", (0, None))
                if branch_status != 200:
                    skipped.append(f"branch protection on `{default}` "
                                   f"({status_detail(branch_status)})")
                elif not protected:
                    return "refuse", (
                        f"only protected branches may deploy to `release`, "
                        f"but `{default}` is not a protected branch"), True
            if policy.get("custom_branch_policies"):
                policies_status, policies_done, names = evidence.get(
                    "policies", (0, False, []))
                if policies_status != 200 or not policies_done:
                    skipped.append(
                        f"deployment-branch rules "
                        f"({status_detail(policies_status)}, "
                        "enumeration incomplete)")
                elif not any(
                    fnmatch.fnmatchcase(default, str(name)) for name in names
                ):
                    return "refuse", (
                        f"no `release` deployment-branch rule lets the "
                        f"default branch `{default}` deploy "
                        f"(rules: {', '.join(map(str, names)) or 'none'})"), True
    if skipped:
        return "unverified", (
            f"release environment partially verified — unverified: "
            f"{'; '.join(skipped)} — confirm them under Settings → "
            "Environments → release"), True
    return "pass", ("required reviewers and a bounded deployment-branch "
                    "rule are in place"), True


def cmd_preflight_env(args: argparse.Namespace) -> int:
    """The publish gate's thin client: fetch the evidence the payload
    calls for, hand it to the pure verdict, render the result. A refusal
    exits 1 on a live bind and lands as an error annotation on a
    rehearsal; an unverified verdict is likewise a hard stop on a live
    bind — except where the environment's existence is proven and the
    operator recorded ALLOW_UNVERIFIED_RELEASE_ENV=1, a logged exception
    that can never cover a proven refusal or a possibly absent
    environment. Rehearsals always warn and continue."""
    if not os.environ.get("GH_TOKEN") or not os.environ.get("GITHUB_REPOSITORY"):
        raise TrainError("preflight-env needs GH_TOKEN and GITHUB_REPOSITORY")
    default = os.environ.get("DEFAULT_BRANCH")
    if not default:
        raise TrainError("preflight-env needs DEFAULT_BRANCH")
    # The fail-closed branch must be the default: only the mode the
    # producer actually emits for a rehearsal takes the warning path,
    # so an unset or misspelled BIND_MODE keeps the strict semantics
    # instead of silently restoring warn-only behavior.
    live = os.environ.get("BIND_MODE") != "dry-run"
    status, env, _ = gh_get("/environments/release")
    evidence: dict[str, tuple] = {}
    for need in needed_evidence(status, env):
        if need == "listing":
            list_status, rows, complete = gh_paginated(
                "/environments", "environments")
            evidence["listing"] = (
                list_status, complete,
                [row.get("name") for row in rows if isinstance(row, dict)])
        elif need == "branch":
            branch_status, branch, _ = gh_get(
                f"/branches/{urllib.parse.quote(default, safe='')}")
            evidence["branch"] = (
                branch_status,
                branch.get("protected") if isinstance(branch, dict) else None)
        elif need == "policies":
            policies_status, rows, complete = gh_paginated(
                "/environments/release/deployment-branch-policies",
                "branch_policies")
            evidence["policies"] = (
                policies_status, complete,
                [row.get("name") for row in rows if isinstance(row, dict)])
    opt_in = os.environ.get(OPT_IN_VAR) == "1"
    verdict, message, exists = preflight_verdict(
        status, env, evidence, default)
    if verdict == "refuse":
        report(f"release env preflight: REFUSED — {message}")
        if live:
            print(f"::error::{message} — {ENV_GUIDE}")
            return 1
        # A rehearsal surfaces the refusal loudly without blocking: the
        # same verdict a live run stops on lands as an error annotation.
        print(f"::error::rehearsal — {message} "
              f"(a live run stops here) — {ENV_GUIDE}")
        return 0
    if verdict == "unverified":
        # A live bind fails closed on anything it cannot positively
        # verify. The recorded opt-in exempts only the narrow case where
        # the environment's existence is itself proven: a possibly absent
        # environment would be auto-created unprotected at job entry,
        # which is the catastrophic case the gate exists for, so no
        # operator knob waves it. Remedies are selected by the answer's
        # own shape — retry a transient failure, a readable token
        # everywhere, and the opt-in only where it can apply.
        if live:
            if not opt_in:
                report(f"release env preflight: REFUSED — unverified: "
                       f"{message}")
                print(f"::error::{message} — live publication could not "
                      f"positively verify the `release` environment, so "
                      f"the run stops here; "
                      f"{_unverified_remedy(status, exists)} — {ENV_GUIDE}")
                return 1
            if not exists:
                report("release env preflight: REFUSED — existence "
                       f"unproven: {message}")
                print(f"::error::{message} — even the existence of the "
                      f"`release` environment is unproven, and a missing "
                      f"one would be auto-created unprotected; "
                      f"{_unverified_remedy(status, False)} — {ENV_GUIDE}")
                return 1
            suffix = (f" — proceeding unverified by operator opt-in "
                      f"({OPT_IN_VAR}=1){JOB_ENTRY}")
            print(f"::warning::{message}{suffix}")
            report(f"release env preflight: warned — {message}{suffix}")
            return 0
        # A rehearsal warns as the early signal it exists to provide;
        # the suffix previews what the identical live run would do.
        if exists and opt_in:
            suffix = (" — a live run would proceed unverified by the "
                      f"recorded opt-in ({OPT_IN_VAR}=1)")
        elif exists:
            suffix = (f" — a live run would stop here unless "
                      f"{OPT_IN_VAR}=1 is recorded")
        elif opt_in:
            suffix = (" — a live run would still stop: the recorded "
                      "opt-in cannot cover unproven existence")
        else:
            suffix = (f" — a live run would stop here; "
                      f"{_unverified_remedy(status, False)}")
        print(f"::warning::{message}{suffix}")
        report(f"release env preflight: warned — {message}{suffix}")
        return 0
    print(message)
    report(f"release env preflight: verified — {message}")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="release.py")
    commands = root.add_subparsers(dest="command", required=True)
    candidate = commands.add_parser("candidate")
    candidate.add_argument("--weekday", choices=WEEKDAYS, required=True)
    candidate.add_argument("--time", required=True)
    candidate.add_argument("--utc-offset", required=True)
    candidate.add_argument("--default-branch", required=True)
    candidate.add_argument("--override", default="")
    candidate.set_defaults(run=cmd_candidate)
    deps = commands.add_parser("deps")
    deps.add_argument("--wait-minutes", type=int, default=0)
    deps.set_defaults(run=cmd_deps)
    version = commands.add_parser("version")
    version.add_argument("--candidate", required=True)
    version.add_argument("--branch", default="")
    version.set_defaults(run=cmd_version)
    order = commands.add_parser("order")
    order.set_defaults(run=cmd_order)
    set_version_cmd = commands.add_parser("set-version")
    set_version_cmd.add_argument("version")
    set_version_cmd.set_defaults(run=cmd_set_version)
    freshness = commands.add_parser("freshness")
    freshness.add_argument("--window-days", type=int, default=14)
    freshness.set_defaults(run=cmd_freshness)
    preflight = commands.add_parser("preflight-env")
    preflight.set_defaults(run=cmd_preflight_env)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        return args.run(args)
    except TrainError as error:
        report(f"### Stopped\n\n{error}")
        print(f"::error::{error}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
