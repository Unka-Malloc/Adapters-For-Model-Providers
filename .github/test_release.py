#!/usr/bin/env python3
"""Generated release train helper tests; regenerate it instead of editing
this copy. .github/release.lock records the source inputs that produced it.

Table-driven coverage of the manifest rewriter and its post-check over
synthetic manifests — plain python3, no cargo, no network: the
cargo_metadata read is stubbed with fabricated workspace data.
"""
from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import unittest

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
        True, "warn", "partially verified",
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
        True, "warn", "partially verified",
    ),
    (
        "incomplete policy enumeration warns",
        200,
        {"protection_rules": REVIEWERS,
         "deployment_branch_policy": CUSTOM_POLICY},
        {"policies": (200, False, ["main"])},
        True, "warn", "partially verified",
    ),
    (
        "404 proven absent refuses on live",
        404, None,
        {"listing": (200, True, ["staging"])},
        True, "refuse", "no `release` environment",
    ),
    (
        "404 proven absent warns on rehearsal",
        404, None,
        {"listing": (200, True, ["staging"])},
        False, "warn", "no `release` environment",
    ),
    (
        "404 listed-but-unreadable warns",
        404, None,
        {"listing": (200, True, ["release", "staging"])},
        True, "warn", "unreadable",
    ),
    (
        "404 with incomplete listing warns — never absence",
        404, None,
        {"listing": (200, False, ["staging"])},
        True, "warn", "could not be proven",
    ),
    (
        "404 with unreadable listing warns",
        404, None,
        {"listing": (403, False, [])},
        True, "warn", "could not be proven",
    ),
    ("401 warns", 401, None, {}, True, "warn", "still applies"),
    ("403 warns", 403, None, {}, False, "warn", "still applies"),
    ("transport failure warns", 0, None, {}, True, "warn", "transport"),
    ("an unexpected status warns", 500, None, {}, True, "warn", "HTTP 500"),
    (
        "an unreadable 200 payload warns",
        200, None, {},
        True, "warn", "unreadable payload",
    ),
]


class PreflightVerdictTest(unittest.TestCase):
    """The publish gate's environment decision is pure: an HTTP status
    and its fetched evidence go in, a verdict comes out — refuse only on
    what the answer can prove (a missing environment auto-creates
    unprotected; a misconfigured one removes the approval gate), warn on
    everything a scope-less token cannot read."""

    def test_every_contract_row(self):
        for name, status, env, evidence, live, verdict, fragment in \
                PREFLIGHT_CASES:
            with self.subTest(name):
                got, message = train.preflight_verdict(
                    status, env, evidence, "main", live)
                self.assertEqual(got, verdict, message)
                self.assertIn(fragment, message)


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


if __name__ == "__main__":
    unittest.main()
