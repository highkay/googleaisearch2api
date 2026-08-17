"""Tests for the Duck.ai x-vqd-hash-1 challenge solver (duck_solver).

Given: the two captured live challenge fixtures (warp + res2260) committed under
tests/fixtures/duckchallenge/, and the wasmrt QuickJS runtime.
When: solve_vqd executes a decoded challenge in the wasmrt context with the
Duck2api-ported DOM prelude installed, then mutates the resolved object.
Then: the returned header value is standard-padded base64 of a JSON object with
sha256-hashed client_hashes (32 bytes each) and a stamped meta block, while all
other challenge keys are preserved.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from googleaisearch2api.duck_solver import DEFAULT_ORIGIN, decode_challenge, solve_vqd

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "duckchallenge"


@pytest.fixture()
def challenge_warp() -> str:
    return (FIXTURES_DIR / "hash1_warp.js.txt").read_text().strip()


@pytest.fixture()
def challenge_res2260() -> str:
    return (FIXTURES_DIR / "hash1_res2260.js.txt").read_text().strip()


def _decode_payload(payload: str) -> dict:
    obj = json.loads(base64.b64decode(payload.encode("ascii")))
    assert isinstance(obj, dict)
    return obj


def _assert_solved_contract(obj: dict) -> None:
    """Asserts the invariants a valid x-vqd-hash-1 response must satisfy."""
    client_hashes = obj["client_hashes"]
    assert isinstance(client_hashes, list)
    assert client_hashes, "challenge did not produce any client_hashes"
    for entry in client_hashes:
        assert isinstance(entry, str)
        raw = base64.b64decode(entry)
        assert len(raw) == 32  # sha256 digest length
        # canonical form: standard alphabet, padded, minimum encoding
        assert base64.b64encode(raw).decode("ascii") == entry

    meta = obj["meta"]
    assert isinstance(meta, dict)
    assert meta["origin"] == DEFAULT_ORIGIN
    assert isinstance(meta["stack"], str) and meta["stack"]
    assert isinstance(meta["duration"], str) and meta["duration"].isdigit()
    # challenge-provided meta fields survive the mutation
    assert meta["v"] == "4"
    assert "timestamp" in meta

    server_hashes = obj["server_hashes"]
    assert isinstance(server_hashes, list)
    assert server_hashes
    assert all(isinstance(s, str) for s in server_hashes)

    assert obj["signals"] == {}


# ---- decode_challenge --------------------------------------------------------


def test_decode_challenge_handles_missing_padding():
    given = "aGVsbG8"  # base64("hello") without padding
    assert decode_challenge(given) == "hello"
    assert decode_challenge("aGVsbG8=") == "hello"


def test_decode_challenge_roundtrips_captured_fixture(challenge_warp):
    encoded = base64.b64encode(challenge_warp.encode("utf-8")).decode("ascii")
    assert decode_challenge(encoded.rstrip("=")) == challenge_warp


# ---- solve_vqd on the captured fixtures --------------------------------------


def test_solve_vqd_warp_challenge(challenge_warp):
    payload = solve_vqd(challenge_warp)
    _assert_solved_contract(_decode_payload(payload))


def test_solve_vqd_res2260_challenge(challenge_res2260):
    payload = solve_vqd(challenge_res2260)
    _assert_solved_contract(_decode_payload(payload))


def test_solve_vqd_executes_dynamically_for_each_challenge(challenge_warp, challenge_res2260):
    payload_warp = solve_vqd(challenge_warp)
    payload_res2260 = solve_vqd(challenge_res2260)
    assert payload_warp != payload_res2260


# ---- error paths --------------------------------------------------------------


def test_solve_vqd_rejects_plain_text_challenge():
    with pytest.raises(ValueError):
        solve_vqd("not javascript")


def test_solve_vqd_rejects_result_without_client_hashes():
    challenge = "(async function(){ return { server_hashes: [], meta: {} }; })()"
    with pytest.raises(ValueError, match="client_hashes"):
        solve_vqd(challenge)
