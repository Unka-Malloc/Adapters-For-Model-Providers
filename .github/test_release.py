#!/usr/bin/env python3
"""Generated release train helper tests; regenerate it instead of editing
this copy. .github/release.lock records the source inputs that produced it.

Table-driven coverage of the manifest rewriter and its post-check over
synthetic manifests — plain python3, no cargo, no network: the
cargo_metadata read is stubbed with fabricated workspace data.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import pathlib
import tempfile
import unittest
import unittest.mock

HELPER = pathlib.Path(__file__).resolve().parent / "release.py"
spec = importlib.util.spec_from_file_location("train", HELPER)
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

NEXT = "0.2.0"


def metadata(packages: dict, workspace_root: str = "") -> dict:
    """A cargo_metadata-shaped workspace: name -> {version, deps, ...}."""
    return {
        "workspace_members": list(packages),
        "workspace_root": workspace_root,
        "packages": [
            {
                "id": name,
                "name": name,
                "version": spec["version"],
                "publish": spec.get("publish"),
                "manifest_path": spec.get("manifest_path", ""),
                "dependencies": [
                    {"name": dep, "req": req}
                    for dep, req in spec.get("deps", {}).items()
                ],
            }
            for name, spec in packages.items()
        ],
    }


# (name, manifest lines, members, expected edit count, expected substrings,
#  absent substrings)
REWRITE_CASES = [
    (
        "bare-string member dep",
        [
            "[package]", 'name = "root"', 'version = "0.1.0"', "",
            "[dependencies]", 'x = "0.1.0"', 'serde = "1"',
        ],
        {"root", "x"},
        2,
        ['version = "0.2.0"', 'x = "0.2.0"', 'serde = "1"'],
        ['"0.1.0"'],
    ),
    (
        "inline table without version gains a pin",
        ["[dependencies]", 'x = { path = "crates/x" }'],
        {"x"},
        1,
        ['x = { version = "0.2.0", path = "crates/x" }'],
        [],
    ),
    (
        "multi-line inline table gains a pin",
        [
            "[dependencies]",
            "x = {",
            '  path = "crates/x",',
            '  features = ["f"],',
            "}",
        ],
        {"x"},
        1,
        ['version = "0.2.0",', 'path = "crates/x"'],
        [],
    ),
    (
        "foo.package + foo.version rename pair",
        [
            "[dependencies]",
            'renamed.package = "x"',
            'renamed.version = "0.1.0"',
        ],
        {"x"},
        1,
        ['renamed.package = "x"', 'renamed.version = "0.2.0"'],
        ['renamed.version = "0.1.0"'],
    ),
    (
        "[dependencies.foo] path-only gains a pin",
        ["[dependencies.x]", 'path = "crates/x"'],
        {"x"},
        1,
        ['[dependencies.x]\nversion = "0.2.0"\npath = "crates/x"'],
        [],
    ),
    (
        "target cfg dev-dependencies sub-table gains a pin",
        ["[target.'cfg(unix)'.dev-dependencies.x]", 'path = "crates/x"'],
        {"x"},
        1,
        ["[target.'cfg(unix)'.dev-dependencies.x]\nversion = \"0.2.0\""],
        [],
    ),
    (
        "a member already at target does not stop the bump",
        [
            "[package]", 'name = "root"', 'version = "0.2.0"', "",
            "[dependencies]", 'x = "0.1.0"',
        ],
        {"root", "x"},
        1,
        ['version = "0.2.0"', 'x = "0.2.0"'],
        ['"0.1.0"'],
    ),
    (
        "a manifest already at target is untouched",
        [
            "[package]", 'name = "root"', 'version = "0.2.0"', "",
            "[dependencies]", 'x = "0.2.0"',
        ],
        {"root", "x"},
        0,
        ['version = "0.2.0"', 'x = "0.2.0"'],
        [],
    ),
]


class HelperTest(unittest.TestCase):
    def stub_metadata(self, data: dict) -> None:
        previous = train.cargo_metadata
        train.cargo_metadata = lambda _repo: data
        self.addCleanup(setattr, train, "cargo_metadata", previous)


class RewriteManifestTest(HelperTest):
    def test_dependency_shapes(self):
        for name, lines, members, edits, present, absent in REWRITE_CASES:
            with self.subTest(name):
                with tempfile.TemporaryDirectory() as tmp:
                    path = pathlib.Path(tmp) / "Cargo.toml"
                    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    self.assertEqual(
                        train.rewrite_manifest(path, set(members), NEXT), edits)
                    text = path.read_text(encoding="utf-8")
                for needle in present:
                    self.assertIn(needle, text)
                for needle in absent:
                    self.assertNotIn(needle, text)


class VerifyBumpTest(HelperTest):
    """The post-check is proven against a fresh metadata read: every
    publishable member must sit at the target and every requirement naming
    a member must pin it."""

    def test_flags_a_member_left_at_the_old_version(self):
        self.stub_metadata(metadata({
            "root": {"version": "0.1.0", "deps": {"x": "^0.2.0"}},
            "x": {"version": "0.2.0"},
        }))
        with self.assertRaises(train.TrainError) as error:
            train.verify_bump(pathlib.Path("/workspace"), {"root", "x"}, NEXT)
        self.assertIn("could not confirm the bump", str(error.exception))
        self.assertIn("root is at 0.1.0", str(error.exception))

    def test_flags_a_member_requirement_left_behind(self):
        self.stub_metadata(metadata({
            "root": {"version": "0.2.0", "deps": {"x": "^0.1.0"}},
            "x": {"version": "0.2.0"},
        }))
        with self.assertRaises(train.TrainError) as error:
            train.verify_bump(pathlib.Path("/workspace"), {"root", "x"}, NEXT)
        self.assertIn("root requires x ^0.1.0 instead of 0.2.0",
                      str(error.exception))

    def test_a_workspace_at_target_passes(self):
        self.stub_metadata(metadata({
            "root": {"version": "0.2.0", "deps": {"x": "^0.2.0"}},
            "x": {"version": "0.2.0"},
        }))
        train.verify_bump(pathlib.Path("/workspace"), {"root", "x"}, NEXT)


class SetVersionRerunTest(HelperTest):
    """A rerun on an already-target workspace must succeed as a no-op —
    mixed-version workspaces, publish=false members and hand bumps are
    states the version command itself plans for, not abort conditions."""

    def test_an_already_target_workspace_is_a_clean_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "crates/x").mkdir(parents=True)
            root_manifest = root / "Cargo.toml"
            member_manifest = root / "crates/x/Cargo.toml"
            root_manifest.write_text("\n".join([
                "[package]", 'name = "root"', 'version = "0.2.0"', "",
                "[workspace]", 'members = ["crates/*"]', "",
                "[dependencies]", 'x = "0.2.0"',
            ]) + "\n", encoding="utf-8")
            member_manifest.write_text(
                '[package]\nname = "x"\nversion = "0.2.0"\n',
                encoding="utf-8")
            before_root = root_manifest.read_text(encoding="utf-8")
            before_member = member_manifest.read_text(encoding="utf-8")
            self.stub_metadata(metadata({
                "root": {"version": "0.2.0", "deps": {"x": "^0.2.0"},
                         "manifest_path": str(root_manifest)},
                "x": {"version": "0.2.0",
                      "manifest_path": str(member_manifest)},
            }, workspace_root=str(root)))
            self.assertEqual(train.set_version(root, NEXT), 0)
            self.assertEqual(root_manifest.read_text(encoding="utf-8"),
                             before_root)
            self.assertEqual(member_manifest.read_text(encoding="utf-8"),
                             before_member)


REVIEWERS = [{"type": "required_reviewers", "reviewers": [{"id": 1}]}]
CUSTOM_POLICY = {"protected_branches": False, "custom_branch_policies": True}
BRANCH_POLICY = {"protected_branches": True, "custom_branch_policies": False}

# (name, GET status, environment payload, evidence, live, verdict,
#  message fragment). The verdict function is pure — status plus fetched
# evidence in, a verdict out — so every row of the publish gate's
# environment contract sits in this table.
PREFLIGHT_CASES = [
    (
        "fully verified payload passes",
        200,
        {"protection_rules": REVIEWERS,
         "deployment_branch_policy": CUSTOM_POLICY},
        {"policies": (200, True, ["main", "release-*"])},
        True, "pass", "bounded deployment-branch rule",
    ),
    (
        "verified through branch protection passes",
        200,
        {"protection_rules": REVIEWERS,
         "deployment_branch_policy": BRANCH_POLICY},
        {"branch": (200, True)},
        True, "pass", "in place",
    ),
    (
        "both policy mechanisms verified passes",
        200,
        {"protection_rules": REVIEWERS,
         "deployment_branch_policy": {"protected_branches": True,
                                      "custom_branch_policies": True}},
        {"branch": (200, True), "policies": (200, True, ["main"])},
        True, "pass", "in place",
    ),
    (
        "null deployment policy refuses — null means every branch",
        200,
        {"protection_rules": REVIEWERS,
         "deployment_branch_policy": None},
        {},
        True, "refuse", "every branch deploy",
    ),
    (
        "policy with no mechanism refuses",
        200,
        {"protection_rules": REVIEWERS,
         "deployment_branch_policy": {"protected_branches": False,
                                      "custom_branch_policies": False}},
        {},
        True, "refuse", "every branch deploy",
    ),
    (
        "empty protection_rules refuses",
        200,
        {"protection_rules": [],
         "deployment_branch_policy": CUSTOM_POLICY},
        {"policies": (200, True, ["main"])},
        True, "refuse", "no required reviewers",
    ),
    (
        "protection_rules without a reviewer rule refuses",
        200,
        {"protection_rules": [{"type": "wait_timer"}],
         "deployment_branch_policy": CUSTOM_POLICY},
        {"policies": (200, True, ["main"])},
        True, "refuse", "no required reviewers",
    ),
    (
        "both facet keys absent warns — skipped, not refused",
        200,
        {},
        {},
        True, "unverified", "partially verified",
    ),
    (
        "unprotected default branch refuses",
        200,
        {"protection_rules": REVIEWERS,
         "deployment_branch_policy": BRANCH_POLICY},
        {"branch": (200, False)},
        True, "refuse", "not a protected branch",
    ),
    (
        "no rule covering the default branch refuses",
        200,
        {"protection_rules": REVIEWERS,
         "deployment_branch_policy": CUSTOM_POLICY},
        {"policies": (200, True, ["release-*"])},
        True, "refuse", "deployment-branch rule",
    ),
    (
        "unreadable branch evidence warns",
        200,
        {"protection_rules": REVIEWERS,
         "deployment_branch_policy": BRANCH_POLICY},
        {"branch": (403, None)},
        True, "unverified", "partially verified",
    ),
    (
        "incomplete policy enumeration warns",
        200,
        {"protection_rules": REVIEWERS,
         "deployment_branch_policy": CUSTOM_POLICY},
        {"policies": (200, False, ["main"])},
        True, "unverified", "partially verified",
    ),
    (
        "404 proven absent refuses on live",
        404, None,
        {"listing": (200, True, ["staging"])},
        True, "refuse", "no `release` environment",
    ),
    (
        "404 proven absent refuses on rehearsal too",
        404, None,
        {"listing": (200, True, ["staging"])},
        False, "refuse", "no `release` environment",
    ),
    (
        "404 listed-but-unreadable warns",
        404, None,
        {"listing": (200, True, ["release", "staging"])},
        True, "unverified", "unreadable",
    ),
    (
        "404 with incomplete listing warns — never absence",
        404, None,
        {"listing": (200, False, ["staging"])},
        True, "unverified", "could not be proven",
    ),
    (
        "404 with unreadable listing warns",
        404, None,
        {"listing": (403, False, [])},
        True, "unverified", "could not be proven",
    ),
    ("401 is unverified on live",
     401, None, {}, True, "unverified", "HTTP 401"),
    ("403 is unverified on rehearsal",
     403, None, {}, False, "unverified", "still applies"),
    ("transport failure is unverified",
     0, None, {}, True, "unverified", "transport"),
    ("an unexpected status is unverified",
     500, None, {}, True, "unverified", "HTTP 500"),
    (
        "an unreadable 200 payload is unverified",
        200, None, {},
        True, "unverified", "unreadable payload",
    ),
]


class PreflightVerdictTest(unittest.TestCase):
    """The publish gate's environment decision is pure: an HTTP status
    and its fetched evidence go in, a verdict comes out — 'refuse' only
    on what the answer can prove (a missing environment auto-creates
    unprotected; a misconfigured one removes the approval gate) and it
    binds at every mode, 'unverified' on everything the evidence cannot
    positively establish, 'pass' on a fully verified shape. On live,
    unverified messages omit the job-entry reassurance because the job
    never enters the environment."""

    def test_every_contract_row(self):
        for name, status, env, evidence, live, verdict, fragment in \
                PREFLIGHT_CASES:
            with self.subTest(name):
                got, message = train.preflight_verdict(
                    status, env, evidence, "main", live)
                self.assertEqual(got, verdict, message)
                self.assertIn(fragment, message)
                if got == "unverified" and live:
                    self.assertNotIn("still applies at job entry", message)


class PreflightFetchTest(unittest.TestCase):
    """The driver gathers only the evidence a payload calls for, and a
    listing walks pages until its own total_count is met — an incomplete
    enumeration proves nothing and never stands in for absence."""

    def test_only_the_facets_the_payload_carries_are_fetched(self):
        cases = [
            (404, None, {"listing"}),
            (403, None, set()),
            (0, None, set()),
            (200, {}, set()),
            (200, {"deployment_branch_policy": None}, set()),
            (200, {"deployment_branch_policy": BRANCH_POLICY}, {"branch"}),
            (200, {"deployment_branch_policy": CUSTOM_POLICY}, {"policies"}),
            (200,
             {"deployment_branch_policy": {"protected_branches": True,
                                           "custom_branch_policies": True}},
             {"branch", "policies"}),
        ]
        for status, env, wanted in cases:
            with self.subTest((status, env)):
                self.assertEqual(train.needed_evidence(status, env), wanted)

    def test_link_next_parsing(self):
        header = ('<https://api.github.com/x?page=2>; rel="next", '
                  '<https://api.github.com/x?page=5>; rel="last"')
        self.assertEqual(train.link_next(header),
                         "https://api.github.com/x?page=2")
        self.assertIsNone(
            train.link_next('<https://api.github.com/x?page=5>; rel="last"'))
        self.assertIsNone(train.link_next(None))


class PreflightCommandTest(unittest.TestCase):
    """The thin client turns a verdict into an exit code. A live bind
    fails closed on anything it cannot positively verify — a possibly
    absent environment would proceed unprotected — unless the operator
    recorded the ALLOW_UNVERIFIED_RELEASE_ENV=1 opt-in; the opt-in never
    exempts a proven refusal. Rehearsals always warn and continue."""

    def run_preflight(self, mode, opt_in, status, env, listing=None):
        environ = {
            "GH_TOKEN": "t",
            "GITHUB_REPOSITORY": "o/r",
            "DEFAULT_BRANCH": "main",
        }
        if mode is not None:
            environ["BIND_MODE"] = mode
        if opt_in:
            environ["ALLOW_UNVERIFIED_RELEASE_ENV"] = opt_in
        previous_get, previous_page = train.gh_get, train.gh_paginated
        train.gh_get = lambda _path: (status, env, None)
        train.gh_paginated = lambda _path, _key: (
            200, [{"name": name} for name in (listing or [])], True)
        self.addCleanup(setattr, train, "gh_get", previous_get)
        self.addCleanup(setattr, train, "gh_paginated", previous_page)
        out = io.StringIO()
        with unittest.mock.patch.dict(os.environ, environ, clear=True), \
                contextlib.redirect_stdout(out):
            rc = train.cmd_preflight_env(
                train.parser().parse_args(["preflight-env"]))
        return rc, out.getvalue()

    def test_live_fails_closed_on_unverified(self):
        rc, out = self.run_preflight("live", None, 403, None)
        self.assertEqual(rc, 1)
        self.assertIn("::error::", out)
        self.assertIn("ALLOW_UNVERIFIED_RELEASE_ENV", out)

    def test_rehearsal_warns_on_unverified(self):
        rc, out = self.run_preflight("dry-run", None, 403, None)
        self.assertEqual(rc, 0)
        self.assertIn("::warning::", out)
        self.assertIn("a live run stops here unless "
                      "ALLOW_UNVERIFIED_RELEASE_ENV=1", out)

    def test_mode_defaults_to_strict(self):
        # An unset or misspelled BIND_MODE keeps live semantics — the
        # warning path only exists for the producer's own "dry-run".
        for mode in (None, "rehearsal", "DRY-RUN"):
            with self.subTest(mode):
                rc, _ = self.run_preflight(mode, None, 403, None)
                self.assertEqual(rc, 1)

    def test_opt_in_exempts_only_unverified(self):
        rc, out = self.run_preflight("live", "1", 403, None)
        self.assertEqual(rc, 0)
        self.assertIn("proceeding unverified by operator opt-in", out)
        rc, out = self.run_preflight("dry-run", "1", 403, None)
        self.assertEqual(rc, 0)
        self.assertIn("recorded opt-in", out)
        self.assertNotIn("stops here unless", out)
        self.assertEqual(self.run_preflight("live", "yes", 403, None)[0], 1)
        # A proven refusal — a complete listing without `release` — is
        # never exempted by the opt-in.
        self.assertEqual(
            self.run_preflight("live", "1", 404, None, ["ci"])[0], 1)

    def test_rehearsal_annotates_refusal_without_stopping(self):
        rc, out = self.run_preflight("dry-run", None, 404, None, ["ci"])
        self.assertEqual(rc, 0)
        self.assertIn("::error::rehearsal", out)
        self.assertIn("a live run stops here", out)

    def test_verified_environment_passes(self):
        rc, _ = self.run_preflight(
            "live", None, 200,
            {"protection_rules": [{"type": "required_reviewers",
                                   "reviewers": [{"id": 1}]}],
             "deployment_branch_policy": {"protected_branches": False,
                                          "custom_branch_policies": True}},
            ["main"])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
