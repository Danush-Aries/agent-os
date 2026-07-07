"""Offline tests for the toolpacks integration pack.

No real network or API keys: every network call is driven through an injected
fake transport, and filesystem/shell tests use a tmp root and harmless
commands.
"""

from __future__ import annotations

import pytest

from agentos.tools import ToolRegistry
from agentos.toolpacks import register_all
from agentos.toolpacks import fs as fsmod
from agentos.toolpacks import github as ghmod
from agentos.toolpacks import shell as shmod
from agentos.toolpacks import web as webmod


# --------------------------------------------------------------------------- #
# web
# --------------------------------------------------------------------------- #
DDG_CANNED = {
    "Heading": "Python (programming language)",
    "AbstractText": "Python is a high-level programming language.",
    "AbstractURL": "https://en.wikipedia.org/wiki/Python",
    "RelatedTopics": [
        {"FirstURL": "https://example.com/a", "Text": "Alpha - first topic"},
        {"Topics": [
            {"FirstURL": "https://example.com/b", "Text": "Beta - nested topic"},
        ]},
        {"FirstURL": "https://en.wikipedia.org/wiki/Python", "Text": "dup"},
    ],
}


def test_parse_ddg_shapes_json():
    results = webmod._parse_ddg(DDG_CANNED, k=5)
    assert results[0] == {
        "title": "Python (programming language)",
        "url": "https://en.wikipedia.org/wiki/Python",
        "snippet": "Python is a high-level programming language.",
    }
    urls = [r["url"] for r in results]
    assert "https://example.com/a" in urls
    assert "https://example.com/b" in urls  # nested Topics flattened
    # dedup: wikipedia url appears once despite the duplicate RelatedTopic
    assert urls.count("https://en.wikipedia.org/wiki/Python") == 1
    for r in results:
        assert set(r) == {"title", "url", "snippet"}


def test_parse_ddg_respects_k():
    assert len(webmod._parse_ddg(DDG_CANNED, k=1)) == 1


def test_web_search_with_injected_fetch():
    calls = {}

    def fake_fetch(url, params):
        calls["url"] = url
        calls["params"] = params
        return DDG_CANNED

    results = webmod.web_search("python", k=3, fetch=fake_fetch)
    assert calls["url"] == webmod.DDG_ENDPOINT
    assert calls["params"]["q"] == "python"
    assert results[0]["url"] == "https://en.wikipedia.org/wiki/Python"


def test_web_search_with_injected_client():
    class FakeResp:
        def json(self):
            return DDG_CANNED

    class FakeClient:
        def get(self, url, params=None):
            return FakeResp()

    results = webmod.web_search("python", client=FakeClient())
    assert len(results) >= 2


def test_http_fetch_with_transport():
    class FakeResp:
        status_code = 200
        text = "hello body"
        headers = {"content-type": "text/plain"}

    class FakeTransport:
        def request(self, method, url, content=None, headers=None):
            assert method == "GET"
            return FakeResp()

    out = webmod.http_fetch("https://x.test", transport=FakeTransport())
    assert out == {"status": 200, "text": "hello body",
                   "headers": {"content-type": "text/plain"}}


# --------------------------------------------------------------------------- #
# github
# --------------------------------------------------------------------------- #
REPO_PAYLOAD = {
    "full_name": "octo/hello",
    "html_url": "https://github.com/octo/hello",
    "description": "a repo",
    "stargazers_count": 42,
    "language": "Python",
    "forks_count": 3,
    "open_issues_count": 1,
}
SEARCH_PAYLOAD = {"items": [REPO_PAYLOAD, REPO_PAYLOAD, REPO_PAYLOAD]}
ISSUES_PAYLOAD = [
    {"number": 1, "title": "bug", "state": "open",
     "html_url": "https://github.com/octo/hello/issues/1",
     "user": {"login": "alice"},
     "labels": [{"name": "bug"}, {"name": "p1"}]},
]


def test_shape_repo():
    out = ghmod._shape_repo(REPO_PAYLOAD)
    assert out == {
        "full_name": "octo/hello",
        "html_url": "https://github.com/octo/hello",
        "description": "a repo",
        "stars": 42,
        "language": "Python",
        "forks": 3,
        "open_issues": 1,
    }


def test_shape_search_repos_caps_k():
    out = ghmod._shape_search_repos(SEARCH_PAYLOAD, k=2)
    assert len(out) == 2
    assert out[0]["full_name"] == "octo/hello"


def test_shape_issue():
    out = ghmod._shape_issue(ISSUES_PAYLOAD[0])
    assert out == {
        "number": 1,
        "title": "bug",
        "state": "open",
        "html_url": "https://github.com/octo/hello/issues/1",
        "user": "alice",
        "labels": ["bug", "p1"],
    }


def test_gh_search_repos_injected_request():
    def fake_request(method, url, *, params=None, json=None):
        assert method == "GET"
        assert "search/repositories" in url
        return SEARCH_PAYLOAD

    out = ghmod.gh_search_repos("q", k=2, request=fake_request)
    assert len(out) == 2


def test_gh_get_repo_injected_request():
    def fake_request(method, url, *, params=None, json=None):
        assert url.endswith("/repos/octo/hello")
        return REPO_PAYLOAD

    assert ghmod.gh_get_repo("octo", "hello", request=fake_request)["stars"] == 42


def test_gh_list_issues_injected_request():
    def fake_request(method, url, *, params=None, json=None):
        assert params["state"] == "open"
        return ISSUES_PAYLOAD

    out = ghmod.gh_list_issues("octo", "hello", request=fake_request)
    assert out[0]["title"] == "bug"


def test_gh_create_issue_injected_request():
    def fake_request(method, url, *, params=None, json=None):
        assert method == "POST"
        assert json["title"] == "new"
        return {"number": 9, "title": "new", "state": "open",
                "html_url": "u", "user": {"login": "me"}, "labels": []}

    out = ghmod.gh_create_issue("octo", "hello", "new", "body", request=fake_request)
    assert out["number"] == 9


def test_gh_create_issue_requires_approval():
    tool = ghmod.gh_create_issue_tool()
    assert tool.requires_approval is True


# --------------------------------------------------------------------------- #
# fs
# --------------------------------------------------------------------------- #
def test_fs_write_read_roundtrip(tmp_path):
    write = fsmod.make_fs_write(tmp_path)
    read = fsmod.make_fs_read(tmp_path)
    res = write("sub/note.txt", "hello world")
    assert res["bytes_written"] > 0
    assert read("sub/note.txt") == "hello world"


def test_fs_traversal_blocked_read(tmp_path):
    read = fsmod.make_fs_read(tmp_path)
    with pytest.raises(fsmod.SandboxError):
        read("../escape.txt")


def test_fs_traversal_blocked_write(tmp_path):
    write = fsmod.make_fs_write(tmp_path)
    with pytest.raises(fsmod.SandboxError):
        write("../../etc/pwned", "x")


def test_fs_absolute_outside_root_blocked(tmp_path):
    read = fsmod.make_fs_read(tmp_path)
    with pytest.raises(fsmod.SandboxError):
        read("/etc/hosts")


def test_fs_list(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "d").mkdir()
    listing = fsmod.make_fs_list(tmp_path)(".")
    names = {e["name"] for e in listing}
    assert names == {"a.txt", "d"}
    by_name = {e["name"]: e for e in listing}
    assert by_name["d"]["is_dir"] is True
    assert by_name["a.txt"]["is_dir"] is False


def test_fs_tool_registry_roundtrip(tmp_path):
    reg = ToolRegistry()
    register_all(reg, enable=["fs_write", "fs_read"], fs_root=tmp_path)
    w = reg.call("fs_write", {"path": "x.txt", "content": "hi"})
    assert not w.is_error
    r = reg.call("fs_read", {"path": "x.txt"})
    assert r.content == "hi"
    # traversal through the registry surfaces as is_error, not a crash
    bad = reg.call("fs_read", {"path": "../../secret"})
    assert bad.is_error


# --------------------------------------------------------------------------- #
# shell
# --------------------------------------------------------------------------- #
def test_shell_run_echo():
    out = shmod.shell_run("echo hi")
    assert out["returncode"] == 0
    assert "hi" in out["stdout"]


def test_shell_run_denylist_refused():
    with pytest.raises(PermissionError):
        shmod.shell_run("rm -rf /")
    with pytest.raises(PermissionError):
        shmod.shell_run("dd if=/dev/zero of=/dev/sda")


def test_shell_run_requires_approval():
    assert shmod.shell_run_tool().requires_approval is True


def test_shell_run_via_registry():
    reg = ToolRegistry()
    register_all(reg, enable=["shell_run"])
    res = reg.call("shell_run", {"command": "echo hi"})
    assert not res.is_error
    assert res.content["returncode"] == 0
    refused = reg.call("shell_run", {"command": "mkfs.ext4 /dev/sda"})
    assert refused.is_error


# --------------------------------------------------------------------------- #
# register_all
# --------------------------------------------------------------------------- #
EXPECTED_TOOLS = {
    "web_search", "http_fetch",
    "gh_search_repos", "gh_get_repo", "gh_list_issues", "gh_create_issue",
    "fs_read", "fs_write", "fs_list",
    "shell_run",
}
APPROVAL_GATED = {"fs_write", "gh_create_issue", "shell_run"}


def test_register_all_registers_everything():
    reg = ToolRegistry()
    names = register_all(reg)
    assert set(names) == EXPECTED_TOOLS
    spec_names = {s["name"] for s in reg.specs()}
    assert EXPECTED_TOOLS <= spec_names


def test_register_all_enable_filter():
    reg = ToolRegistry()
    names = register_all(reg, enable=["web_search", "fs_read"])
    assert set(names) == {"web_search", "fs_read"}


def test_approval_gated_tools_flagged():
    reg = ToolRegistry()
    register_all(reg)
    for name in APPROVAL_GATED:
        assert reg.get(name).requires_approval is True
    # a read-only tool is not gated
    assert reg.get("web_search").requires_approval is False
