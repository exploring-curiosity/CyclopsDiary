"""CyclopsDiary: the shared memory in MongoDB and the private object tracker.

Footage and its steps belong to a workspace (everyone searches everyone's
cameras); a tracker and its sightings belong to one person. Steps here are
written straight into the store with synthetic word lists (an event's words
lifted), so the search and the tracker are checked without loading the
model; the model's own read is the ingest's, checked on real footage.

Needs a MongoDB; `scripts/mongo_local.sh` starts one on :27100 (or set
CYCLOPSDIARY_TEST_URI). Each test gets a fresh database and drops it.

Run: python3 -m pytest tests/test_cyclopsdiary.py -v -p no:django
"""
import json
import re
import os
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np                                    # noqa: E402
import pytest                                         # noqa: E402

pymongo = pytest.importorskip("pymongo")
URI = os.environ.get("CYCLOPSDIARY_TEST_URI", "mongodb://localhost:27100")
PIN = dict(model="test", rate=2.0, n_frames=2, top_k=24, frame_hw=[384, 512], window=8192)


@pytest.fixture()
def db():
    c = pymongo.MongoClient(URI, serverSelectionTimeoutMS=1500, tz_aware=True)
    try:
        c.admin.command("ping")
    except Exception:                                 # noqa: BLE001
        pytest.skip(f"no MongoDB at {URI} (scripts/mongo_local.sh starts one)")
    name = f"cyclopsdiary_test_{uuid.uuid4().hex[:8]}"
    from cyclopsdiary import atlas
    atlas.ensure(c[name])
    yield c[name]
    c.drop_database(name)


def _t(s):
    return datetime(2026, 9, 26, 9, 0, tzinfo=timezone.utc) + timedelta(seconds=s)


def _people(db):
    from cyclopsdiary import workspace as WS
    WS.add_person(db, "alex", "home")
    WS.add_person(db, "bea", "home")
    WS.add_source(db, "alex-glasses", "alex", "glasses")
    WS.add_source(db, "bea-phone", "bea", "phone")


def _lists(n, events, rng, vocab=60, k=24, lift=6.0):
    ids, lps = [], []
    for j in range(n):
        logit = rng.normal(size=vocab)
        logit[:10] -= 3.0
        if any(a <= j < b for a, b in events):
            logit[:4] += lift
        lp = logit - np.log(np.exp(logit).sum())
        top = np.argsort(-lp)[:k]
        ids.append(top)
        lps.append(lp[top])
    return ids, lps


def _footage(db, fid, source, started, events, n=30, seed=0):
    from cyclopsdiary import ingest as IN
    ids, lps = _lists(n, events, np.random.default_rng(seed))
    src = db.sources.find_one({"_id": source})
    f = dict(_id=fid, workspace=src["workspace"], source=source, person=src["person"], started_at=started,
             status="ready", pin=PIN, steps=n, duration_s=float(n))
    db.footage.insert_one(f)
    IN.write_steps(db, f, [(float(j), float(j + 1), np.zeros(4, np.float32), ids[j], lps[j]) for j in range(n)])
    return f


def _tiou(a, b):
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return i / (max(a[1], b[1]) - min(a[0], b[0]))


def test_the_schema_is_made_once_and_making_it_again_changes_nothing(db):
    from cyclopsdiary import atlas
    assert atlas.ensure(db)["created"] == []
    idx = db.trackers.index_information().values()
    assert any(v.get("unique") and [k for k, _ in v["key"]] == ["owner", "name"] for v in idx)


def test_writing_the_same_step_twice_keeps_one_copy(db):
    from cyclopsdiary import ingest as IN
    _people(db)
    f = _footage(db, "fa", "alex-glasses", _t(0), [(5, 9)])
    n = db.steps.count_documents({"footage": "fa"})
    IN.write_steps(db, f, [(0.0, 1.0, np.zeros(4, np.float32), np.arange(24), -np.arange(24.0))])
    assert db.steps.count_documents({"footage": "fa"}) == n == 30


def test_a_tracker_is_its_owners_alone(db):
    from cyclopsdiary import tracker as TR
    _people(db)
    _footage(db, "fa", "alex-glasses", _t(0), [(5, 9)])
    TR.register(db, "alex", "keys", "fa", 5, 9)
    assert TR.get(db, "bea", "keys") is None and TR.trackers(db, "bea") == []
    TR.register(db, "bea", "keys", "fa", 5, 9)                  # her own keys, the same name
    his = db.sightings.find_one({"owner": "alex"})
    assert not TR.confirm(db, "bea", his["_id"]) and not TR.reject(db, "bea", his["_id"])
    assert db.sightings.find_one({"_id": his["_id"]})["rejected_at"] is None


def test_the_example_finds_the_object_in_someone_elses_later_footage(db):
    """The tracker is Alex's; the footage it finds it in is Bea's phone an hour later."""
    from cyclopsdiary import tracker as TR
    _people(db)
    _footage(db, "fa", "alex-glasses", _t(0), [(5, 9)], seed=1)
    _footage(db, "fb", "bea-phone", _t(3600), [(20, 24)], seed=2)
    TR.register(db, "alex", "keys", "fa", 5, 9)
    last = TR.locate(db, "alex", "keys")["last_known"]
    assert last["footage"] == "fb" and last["source"] == "bea-phone"
    assert _tiou((last["t0"], last["t1"]), (20.0, 24.0)) >= 0.5
    assert last["observed_at"] == _t(3600) + timedelta(seconds=last["t0"])


def test_a_rejected_sighting_is_no_longer_believed(db):
    from cyclopsdiary import tracker as TR
    _people(db)
    _footage(db, "fa", "alex-glasses", _t(0), [(5, 9)], seed=1)
    _footage(db, "fb", "bea-phone", _t(3600), [(20, 24)], seed=2)
    TR.register(db, "alex", "keys", "fa", 5, 9)
    found = TR.locate(db, "alex", "keys")["last_known"]
    assert TR.reject(db, "alex", found["_id"])
    back = TR.last_known(db, "alex", "keys")
    assert back["footage"] == "fa" and (back["t0"], back["t1"]) == (5.0, 9.0)     # where he showed it


def test_confirming_keeps_when_it_was_seen_apart_from_when_it_was_confirmed(db):
    from cyclopsdiary import tracker as TR
    _people(db)
    _footage(db, "fa", "alex-glasses", _t(0), [(5, 9)], seed=1)
    _footage(db, "fb", "bea-phone", _t(3600), [(20, 24)], seed=2)
    TR.register(db, "alex", "keys", "fa", 5, 9)
    found = TR.locate(db, "alex", "keys")["last_known"]
    assert found["confirmed_at"] is None
    assert TR.confirm(db, "alex", found["_id"])
    s = db.sightings.find_one({"_id": found["_id"]})
    assert s["confirmed_at"] is not None and s["observed_at"] == found["observed_at"]
    assert s["recorded_at"] == found["recorded_at"]


def test_a_frame_keeps_its_aspect_inside_the_pixel_budget():
    from cyclopsdiary import tower as TW
    assert TW.frame_hw(640, 480, 384 * 512) == (384, 512)             # the measured setting
    assert TW.frame_hw(1920, 1080, 384 * 512) == (320, 576)
    assert TW.frame_hw(1080, 1920, 384 * 512) == (576, 320)            # a phone held upright


def test_a_rotated_phone_video_is_read_upright(tmp_path):
    """Phones store portrait video as landscape pixels plus a display rotation;
    the model must see what the person saw."""
    av = pytest.importorskip("av")
    from cyclopsdiary import tower as TW
    raw, rot = tmp_path / "raw.mp4", tmp_path / "rot.mp4"
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=1",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(raw)], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-display_rotation", "90", "-i", str(raw), "-c", "copy", str(rot)],
                       check=True)
        ref = subprocess.run(["ffmpeg", "-v", "error", "-i", str(rot), "-frames:v", "1", "-f", "rawvideo",
                              "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("ffmpeg cannot make the rotated test video")
    k = TW.rotation_of(str(rot))
    with av.open(str(rot)) as c:
        frame = next(c.decode(video=0)).to_ndarray(format="rgb24")
    up = TW.upright(frame, k)
    assert up.shape == (320, 240, 3)
    want = np.frombuffer(ref, np.uint8).reshape(320, 240, 3)             # ffmpeg's own autorotated frame
    assert np.abs(up.astype(int) - want.astype(int)).mean() < 2.0


# ---- the query layer: one example in, events out (query.py), and the worker (stream.py)

def _phases(n, phases, rng, vocab=60, k=24, lift=6.0):
    """Word lists where phase g of an event lifts words 4g..4g+3 over one fixed background, so the two phases
    are equally distinct and only their order tells a put-down (hand, table) from a take (table, hand)."""
    ids, lps = [], []
    base = np.random.default_rng(99).normal(size=vocab)
    base[:10] -= 3.0
    for j in range(n):
        logit = rng.normal(size=vocab)
        logit[:10] -= 3.0
        for a, b, g in phases:
            if a <= j < b:
                logit = base.copy()
                logit[4 * g:4 * g + 4] += lift
        lp = logit - np.log(np.exp(logit).sum())
        top = np.argsort(-lp)[:k]
        ids.append(top)
        lps.append(lp[top])
    return ids, lps


def _footage2(db, fid, source, started, phases, n=30, seed=0, role=None):
    from cyclopsdiary import ingest as IN
    ids, lps = _phases(n, phases, np.random.default_rng(seed))
    src = db.sources.find_one({"_id": source})
    f = dict(_id=fid, workspace=src["workspace"], source=source, person=src["person"], started_at=started,
             status="ready", pin=PIN, steps=n, duration_s=float(n))
    if role:
        f["role"] = role
    db.footage.insert_one(f)
    IN.write_steps(db, f, [(float(j), float(j + 1), np.zeros(4, np.float32), ids[j], lps[j]) for j in range(n)])
    return f


PUT_A = [(5, 7, 0), (7, 9, 1)]                          # alex puts the keys down at 5-9 s
PUT_B, TAKE_B = [(4, 6, 0), (6, 8, 1)], [(20, 22, 1), (22, 24, 0)]


def _ask(db, direction, scope="all", **kw):
    from cyclopsdiary import query as QY
    q = QY.enqueue(db, "home", "fa", 5, 9, direction=direction, scope=scope, **kw)
    QY.answer(db, QY.claim(db, q["_id"]))
    return QY.events(db, q["_id"])


def test_the_example_forward_finds_the_same_act_and_reversed_finds_it_undone(db):
    _people(db)
    _footage2(db, "fa", "alex-glasses", _t(0), PUT_A, seed=1)
    _footage2(db, "fb", "bea-phone", _t(3600), PUT_B + TAKE_B, seed=2)
    put, take = _ask(db, "forward")[0], _ask(db, "reverse")[0]
    assert put["footage"] == "fb" and _tiou((put["t0"], put["t1"]), (4.0, 8.0)) >= 0.5
    assert take["footage"] == "fb" and _tiou((take["t0"], take["t1"]), (20.0, 24.0)) >= 0.5
    assert take["kind"] == "reverse" and take["rank"] == 1 and take["source"] == "bea-phone"
    assert take["observed_at"] == _t(3600) + timedelta(seconds=take["t0"])


def test_scope_others_and_after_keep_the_answers_off_the_examples_recording_and_past(db):
    _people(db)
    _footage2(db, "fa", "alex-glasses", _t(0), PUT_A + [(15, 17, 1), (17, 19, 0)], seed=1)   # he takes them back
    _footage2(db, "f0", "bea-phone", _t(-3600), [(10, 12, 1), (12, 14, 0)], seed=3)          # an hour before
    _footage2(db, "fb", "bea-phone", _t(3600), TAKE_B, seed=2)
    assert {e["footage"] for e in _ask(db, "reverse", "others")} <= {"f0", "fb"}
    after = _ask(db, "reverse", "after")
    assert after and all(e["observed_at"] >= _t(9) for e in after)
    assert "f0" not in {e["footage"] for e in after}


def test_an_example_clip_is_searched_but_never_an_answer(db):
    _people(db)
    _footage2(db, "fa", "alex-glasses", _t(0), PUT_A, seed=1)
    _footage2(db, "fx", "bea-phone", _t(60), PUT_B, seed=4, role="example")
    _footage2(db, "fb", "bea-phone", _t(3600), PUT_B, seed=2)
    got = _ask(db, "forward")
    assert got and "fx" not in {e["footage"] for e in got} and got[0]["footage"] == "fb"


def test_a_queued_query_is_answered_by_the_worker_into_events(db):
    from cyclopsdiary import query as QY
    from cyclopsdiary import stream as SM
    _people(db)
    _footage2(db, "fa", "alex-glasses", _t(0), PUT_A, seed=1)
    _footage2(db, "fb", "bea-phone", _t(3600), PUT_B + TAKE_B, seed=2)
    q = QY.enqueue(db, "home", "fa", 5, 9, direction="reverse", label="keys taken off the table", asked_by="alex")
    bad = QY.enqueue(db, "home", "fa", 5, 9)
    db.footage.update_one({"_id": "fa"}, {"$set": {"status": "reading"}})     # not ready: its query must fail
    n = SM.serve(db, once=True)
    assert n["failed_queries"] == 2 and db.queries.find_one({"_id": bad["_id"]})["status"] == "failed"
    db.footage.update_one({"_id": "fa"}, {"$set": {"status": "ready"}})
    db.queries.update_one({"_id": q["_id"]}, {"$set": {"status": "queued"}})
    assert SM.serve(db, once=True)["answered"] == 1
    done = db.queries.find_one({"_id": q["_id"]})
    assert done["status"] == "done" and done["events"] == db.events.count_documents({"query": q["_id"]}) > 0
    first = QY.events(db, q["_id"])[0]
    assert first["label"] == "keys taken off the table" and first["footage"] == "fb"


def test_the_inbox_takes_a_clip_once_it_holds_still_and_only_for_a_known_camera(db, tmp_path):
    from cyclopsdiary import stream as SM
    _people(db)
    (tmp_path / "alex-glasses").mkdir()
    (tmp_path / "nobody").mkdir()
    clip = tmp_path / "alex-glasses" / "a.mp4"
    clip.write_bytes(b"x" * 10)
    (tmp_path / "nobody" / "b.mp4").write_bytes(b"x" * 10)
    box = SM.Inbox(tmp_path)
    assert box.ready(db) == []                                  # first look
    with open(clip, "ab") as f:
        f.write(b"more")                                        # still being copied
    assert box.ready(db) == []
    got = box.ready(db)
    assert [(p.name, s) for _, p, s in got] == [("a.mp4", "alex-glasses")]
    box.done.add(got[0][0])
    assert box.ready(db) == []


def test_a_cameras_clock_offset_is_kept_and_replaced(db):
    from cyclopsdiary import workspace as WS
    _people(db)
    assert WS.add_source(db, "bea-phone", "bea", "phone", clock_offset_s=-3.5)["clock_offset_s"] == -3.5
    assert WS.add_source(db, "bea-phone", "bea", "phone", clock_offset_s=1.25)["clock_offset_s"] == 1.25
    assert "clock_offset_s" not in WS.add_source(db, "alex-glasses", "alex", "glasses")


def test_each_phone_is_timed_from_when_it_started_recording():
    from cyclopsdiary import ingest as IN
    t, where = IN.capture_start("PXL_20260926_151010310.mp4", {"creation_time": "2026-09-26T15:10:30.000000Z"})
    assert t == datetime(2026, 9, 26, 15, 10, 10, 310000, tzinfo=timezone.utc) and where == "pixel file name"
    t, where = IN.capture_start("IMG_2658.MOV", {"com.apple.quicktime.creationdate": "2026-09-26T11:09:59-0400",
                                                 "creation_time": "2026-09-26T15:10:05.000000Z"})
    assert t == datetime(2026, 9, 26, 15, 9, 59, tzinfo=timezone.utc) and where == "apple creationdate"
    assert IN.capture_start("clip.mp4", {}) == (None, None)


def test_the_database_refuses_a_malformed_query_row(db):
    from pymongo.errors import WriteError
    from cyclopsdiary import query as QY
    _people(db)
    _footage2(db, "fa", "alex-glasses", _t(0), PUT_A, seed=1)
    QY.enqueue(db, "home", "fa", 5, 9, direction="reverse")                      # the query layer's own rows pass
    for bad in ({"workspace": "home", "status": "queued"},                        # no example, no direction
                {"workspace": "home", "example": {"footage": "fa", "t0": 9, "t1": 5}, "direction": "reverse",
                 "scope": "all", "top": 10, "status": "queued", "asked_at": _t(0)},   # ends before it starts
                {"workspace": "home", "example": {"footage": "fa", "t0": 5, "t1": 9}, "direction": "sideways",
                 "scope": "all", "top": 10, "status": "queued", "asked_at": _t(0)}):
        with pytest.raises(WriteError) as e:
            db.queries.insert_one(bad)
        assert e.value.code == 121


def test_a_claim_left_by_a_dead_worker_is_queued_again_and_a_live_ones_is_not(db):
    import socket
    from cyclopsdiary import query as QY
    _people(db)
    _footage2(db, "fa", "alex-glasses", _t(0), PUT_A, seed=1)
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    dead, live = QY.enqueue(db, "home", "fa", 5, 9), QY.enqueue(db, "home", "fa", 5, 9)
    for q, pid in ((dead, gone.pid), (live, os.getpid())):
        db.queries.update_one({"_id": q["_id"]}, {"$set": {"status": "running",
                                                           "worker": f"{socket.gethostname()}:{pid}"}})
    assert QY.requeue_stale(db) == 1
    assert db.queries.find_one({"_id": dead["_id"]})["status"] == "queued"
    assert db.queries.find_one({"_id": live["_id"]})["status"] == "running"


def test_ask_answers_at_once_and_the_timeline_counts_a_moment_once(db):
    from cyclopsdiary import query as QY
    _people(db)
    _footage2(db, "fa", "alex-glasses", _t(0), PUT_A, seed=1)
    _footage2(db, "fb", "bea-phone", _t(3600), PUT_B + TAKE_B, seed=2)
    a = QY.ask(db, "home", "fa", 5, 9, direction="reverse", label="keys off the table")
    b = QY.ask(db, "home", "fa", 5, 9, direction="reverse", scope="others", label="Keys off the table")
    assert a["status"] == b["status"] == "done"
    top = QY.events(db, a["_id"])[0]
    got = QY.timeline(db, "home", label="keys", kind="reverse", apart_only=False)
    same = [m for m in got if (m["footage"], m["t0"], m["t1"]) == (top["footage"], top["t0"], top["t1"])]
    assert len(same) == 1 and same[0]["found_by"] == 2
    assert [m["observed_at"] for m in got] == sorted((m["observed_at"] for m in got), reverse=True)
    assert QY.timeline(db, "home", kind="forward", apart_only=False) == []


def test_a_strands_session_round_trips_through_mongodb(db):
    from cyclopsdiary import agentmemory as AM
    _people(db)
    AM.session_create(db, dict(session_id="s1", session_type="AGENT", created_at="t0", updated_at="t0"), "home", "alex")
    with pytest.raises(ValueError):
        AM.session_create(db, dict(session_id="s1", session_type="AGENT"), "home", "alex")
    assert AM.session_read(db, "s1") == dict(session_id="s1", session_type="AGENT", created_at="t0", updated_at="t0")
    agent = dict(agent_id="default", state={"k": 1}, conversation_manager_state={}, _internal_state={},
                 created_at="t0", updated_at="t0")
    AM.agent_create(db, "s1", agent)
    assert AM.agent_update(db, "s1", dict(agent, state={"k": 2}, created_at="t9", updated_at="t9"))
    assert AM.agent_read(db, "s1", "default") == dict(agent, state={"k": 2}, updated_at="t9")
    assert not AM.agent_update(db, "s1", dict(agent, agent_id="nobody"))
    for i, text in enumerate(["where are my keys", "on the hall table at 09:00", "thanks"]):
        AM.message_create(db, "s1", "default", dict(message={"role": "user" if i % 2 == 0 else "assistant",
                                                             "content": [{"text": text}]},
                                                    message_id=i, redact_message=None, created_at="t", updated_at="t"))
    assert [m["message_id"] for m in AM.messages(db, "s1", "default", limit=2, offset=1)] == [1, 2]
    assert AM.message_update(db, "s1", "default", dict(AM.message_read(db, "s1", "default", 2),
                                                       redact_message={"role": "user", "content": []}))
    assert AM.message_read(db, "s1", "default", 2)["redact_message"] == {"role": "user", "content": []}


def test_notes_are_the_persons_own_and_recall_finds_notes_and_old_lines_by_their_words(db):
    from cyclopsdiary import agentmemory as AM
    _people(db)
    AM.remember(db, "home", "alex", "Spare key", "lives in the blue bowl by the door")
    AM.remember(db, "home", "bea", "spare key", "bea keeps hers in her bag")
    with pytest.raises(ValueError):
        AM.remember(db, "home", "carol", "x", "y")
    AM.session_create(db, dict(session_id="s1", session_type="AGENT"), "home", "alex")
    AM.message_create(db, "s1", "default", dict(message={"role": "user", "content": [{"text": "I left the bowl in the kitchen"}]},
                                                message_id=0))
    got = AM.recall(db, "home", "alex", text="bowl")
    assert [n["note"] for n in got["notes"]] == ["lives in the blue bowl by the door"]
    assert [s["text"] for s in got["said"]] == ["I left the bowl in the kitchen"]
    assert [n["person"] for n in AM.recall(db, "home", "bea", subject="SPARE KEY")["notes"]] == ["bea"]
    assert AM.recall(db, "home", "bea", text="kitchen")["said"] == []           # alex's conversation, not bea's


def test_the_context_says_what_the_workspace_holds(db):
    from cyclopsdiary import agentmemory as AM
    from cyclopsdiary import query as QY
    _people(db)
    _footage2(db, "fa", "alex-glasses", _t(0), PUT_A, seed=1)
    _footage2(db, "fb", "bea-phone", _t(3600), PUT_B + TAKE_B, seed=2)
    _footage2(db, "fx", "bea-phone", _t(60), PUT_B, seed=4, role="example")
    QY.ask(db, "home", "fa", 5, 9, direction="reverse", label="keys off the table")
    AM.remember(db, "home", "alex", "keys", "usually on the hall table")
    ctx = AM.context(db, "home", "alex")
    assert [(c["source"], c["kind"], c["recordings"], c["steps"]) for c in ctx["cameras"]] == \
        [("alex-glasses", "glasses", 1, 30), ("bea-phone", "phone", 1, 30)]       # the example clip is not a recording
    assert ctx["notes"][0]["note"] == "usually on the hall table" and ctx["open_queries"] == 0
    text = AM.render(ctx)
    assert "alex-glasses (glasses, alex)" in text and "keys: usually on the hall table" in text


def test_the_tools_are_served_over_mcp(db):
    import asyncio
    import json
    from cyclopsdiary import mcp_server
    _people(db)
    _footage2(db, "fa", "alex-glasses", _t(0), PUT_A, seed=1)
    _footage2(db, "fb", "bea-phone", _t(3600), PUT_B + TAKE_B, seed=2)
    shared, alex = mcp_server.build(db, "home"), mcp_server.build(db, "home", "alex")
    names = lambda m: {t.name for t in asyncio.run(m.list_tools())}                   # noqa: E731
    assert names(shared) == {"footage", "find_moments", "moments", "timeline", "context"}
    assert names(alex) == names(shared) | {"remember", "recall", "my_objects", "add_object", "where_is",
                                           "confirm_sighting", "reject_sighting",
                                           "ask", "find_object", "object_belief", "at_place", "between"}

    def call(m, name, **args):
        out = asyncio.run(m.call_tool(name, args))
        blocks = out[0] if isinstance(out, tuple) else out
        return json.loads(blocks[0].text)

    got = call(alex, "find_moments", footage="fa", start_s=5, end_s=9, direction="reverse", label="keys off")
    assert got["status"] == "done" and got["moments"][0]["footage"] == "fb"
    assert call(alex, "moments", query_id=got["query"])["moments"] == got["moments"]
    assert call(alex, "remember", subject="keys", note="taken off the table", query_id=got["query"])["query"] == got["query"]
    assert call(alex, "recall", subject="keys")["notes"][0]["note"] == "taken off the table"


def test_last_seen_follows_the_move_to_the_end_of_the_recording_that_saw_it(db, tmp_path):
    from cyclopsdiary import lastseen as LS
    _people(db)
    _footage2(db, "fa", "alex-glasses", _t(0), PUT_A, seed=1)
    _footage2(db, "fb", "bea-phone", _t(3600), PUT_B + TAKE_B, seed=2)
    for fid in ("fa", "fb"):
        (tmp_path / f"{fid}.mp4").write_bytes(b"")
        db.footage.update_one({"_id": fid}, {"$set": {"path": str(tmp_path / f"{fid}.mp4")}})
    obs = dict(object="keys", person="alex", source="alex-glasses", start=_t(5).isoformat(), end=_t(9).isoformat())
    f, t0, t1 = LS.example(db, obs["source"], obs["start"], obs["end"])
    assert (f["_id"], t0, t1) == ("fa", 5.0, 9.0)
    with pytest.raises(ValueError):
        LS.example(db, "bea-phone", _t(10).isoformat(), _t(12).isoformat())    # that camera was not recording
    r = LS.run(db, obs)                                                       # reverse, after: the keys taken away
    assert r["moved"]["footage"] == "fb" and _tiou((r["moved"]["t0"], r["moved"]["t1"]), (20.0, 24.0)) >= 0.5
    assert (r["last"]["t0"], r["last"]["t1"]) == (r["moved"]["t0"], 30.0)
    assert r["last"]["observed_at"] == _t(3630) and r["query"]["label"] == "keys moved"
    page = LS.page(r, tmp_path / "demo" / "lastseen.html").read_text()
    assert 'src="media/fb.mp4#t=' in page and (tmp_path / "demo" / "media" / "fb.mp4").resolve() == tmp_path / "fb.mp4"


# ---- live cameras: a phone's recording streamed in chunks, read as it arrives (live.py, camserver.py)

class _Model:
    """Stands in for the world model's reader: one row per atom; the atom's count is its word."""
    def __init__(self):
        self.atoms = []

    def open(self):
        pass

    def push(self, frames, idx, fps):
        self.atoms.append(list(idx))
        self.last_top = (np.arange(4, dtype=np.int32) + len(self.atoms), np.zeros(4, np.float32))
        return np.ones(4) / 2.0


def _recording(mime, seconds=4.0, fps=10, w=320, h=240):
    """A phone-like recording made here: VP8 WebM (Android Chrome) or fragmented H.264 MP4 (iPhone Safari)."""
    import io
    from fractions import Fraction
    import av
    buf = io.BytesIO()
    webm = mime.startswith("video/webm")
    c = av.open(buf, "w", format="webm" if webm else "mp4",
                options={} if webm else {"movflags": "frag_keyframe+empty_moov+default_base_moof"})
    s = c.add_stream("libvpx" if webm else "libx264", rate=fps, options={"g": str(fps)})
    s.width, s.height, s.pix_fmt = w, h, "yuv420p"
    rng = np.random.default_rng(0)
    for i in range(int(seconds * fps)):
        fr = av.VideoFrame.from_ndarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), format="rgb24")
        fr.pts, fr.time_base = i, Fraction(1, fps)
        for p in s.encode(fr):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()
    return buf.getvalue()


def _chunks(data, n=7):
    cut = sorted({int(len(data) * k / n) for k in range(1, n)})
    return [data[a:b] for a, b in zip([0] + cut, cut + [len(data)])]


def _live(db, tmp_path, mime, model):
    from cyclopsdiary import live as LV
    from cyclopsdiary import tower
    return LV.LiveSession(db, "bea-phone", _t(0), mime, root=tmp_path,
                          make_read=lambda hw: tower.Read(None, hw, model=model))


@pytest.mark.parametrize("mime", ["video/webm;codecs=vp8", "video/mp4;codecs=avc1.42E01E"])
def test_a_live_recording_is_kept_as_sent_and_read_while_it_arrives(db, tmp_path, mime):
    import hashlib
    import time
    _people(db)
    data, model = _recording(mime), _Model()
    s = _live(db, tmp_path, mime, model)
    parts = _chunks(data)
    for p in parts[:5]:
        s.feed(p)
    for _ in range(50):                                            # steps land before the phone stops
        if db.steps.count_documents({"footage": s.footage["_id"]}):
            break
        time.sleep(0.1)
    assert db.steps.count_documents({"footage": s.footage["_id"]}) > 0
    assert db.footage.find_one({"_id": s.footage["_id"]})["status"] == "live"
    for p in parts[5:]:
        s.feed(p)
    done = s.stop()
    f = db.footage.find_one({"_id": s.footage["_id"]})
    assert Path(f["path"]).read_bytes() == data                         # the footage is what the phone sent
    assert f["status"] == "ready" and f["sha256"] == hashlib.sha256(data).hexdigest() and f["bytes"] == len(data)
    assert (f["width"], f["height"]) == (320, 240) and f["pin"]["frame_hw"] and f["source"] == "bea-phone"
    steps = list(db.steps.find({"footage": f["_id"]}).sort("i", 1))
    assert len(steps) == f["steps"] == done["steps"] == 4 == len(model.atoms)   # 4 s at 2 fps, 2 frames a step
    assert [st["t0"] for st in steps] == [0.0, 1.0, 2.0, 3.0]
    assert all(st["observed_at"] == _t(st["t0"]) for st in steps)


def test_a_live_recording_cut_off_midway_keeps_what_arrived(db, tmp_path):
    _people(db)
    mime = "video/webm;codecs=vp8"
    data = _recording(mime)
    s = _live(db, tmp_path, mime, _Model())
    head = data[: int(len(data) * 0.6)]
    s.feed(head)
    s.stop()                                                            # the socket dropped: no stop message
    f = db.footage.find_one({"_id": s.footage["_id"]})
    assert f["status"] == "ready" and 0 < f["steps"] < 4 and Path(f["path"]).read_bytes() == head


def test_the_camera_service_asks_for_its_token_and_streams_a_phone_into_the_memory(db, tmp_path):
    from starlette.testclient import TestClient
    from cyclopsdiary import camserver, tower
    _people(db)
    mime = "video/webm;codecs=vp8"
    data = _recording(mime)
    app = camserver.app(db, "s3cret", make_read=lambda hw: tower.Read(None, hw, model=_Model()), root=tmp_path)
    c = TestClient(app)
    assert c.get("/").status_code == 403 and c.get("/sources?t=wrong").status_code == 403
    page = c.get("/?t=s3cret")
    assert page.status_code == 200 and "getUserMedia" in page.text and "MediaRecorder" in page.text
    assert {s["_id"] for s in c.get("/sources?t=s3cret").json()} == {"alex-glasses", "bea-phone"}
    with c.websocket_connect("/ingest?t=s3cret") as ws:
        ws.send_json({"source": "bea-phone", "started_at": _t(0).timestamp() * 1000, "mime": mime})
        fid = ws.receive_json()["footage"]
        for p in _chunks(data):
            ws.send_bytes(p)
        ws.send_json({"stop": True})
        while "done" not in (m := ws.receive_json()):
            pass
    f = db.footage.find_one({"_id": fid})
    assert m["done"]["steps"] == f["steps"] == 4 and f["status"] == "ready" and Path(f["path"]).read_bytes() == data


# ---- private objects (objects.py): named, found by the model's own words, tracked across cameras

class FakeTok:
    """The model's tokenizer, for five words: keys (20), three places and a colour."""
    words = {"keys": 20, "table": 21, "chair": 22, "floor": 23, "blue": 24}

    def encode(self, text, add_special_tokens=False):
        return [self.words.get(text.strip().lower(), 59)]

    def decode(self, ids):
        return " " + {v: k for k, v in self.words.items()}.get(int(ids[0]), f"w{int(ids[0])}")


def _scene(n, spans, rng, vocab=60, k=24, lift=6.0):
    """Word lists where, inside span (a, b, place), the keys' word and that place's word are lifted."""
    ids, lps = [], []
    for j in range(n):
        logit = rng.normal(size=vocab)
        logit[20:24] -= 3.0
        for a, b, place in spans:
            if a <= j < b:
                logit[20] += lift
                logit[place] += lift - 1.0
        lp = logit - np.log(np.exp(logit).sum())
        top = np.argsort(-lp)[:k]
        ids.append(top)
        lps.append(lp[top])
    return ids, lps


def _footage3(db, fid, source, started, spans, n=20, seed=0):
    from cyclopsdiary import ingest as IN
    ids, lps = _scene(n, spans, np.random.default_rng(seed))
    src = db.sources.find_one({"_id": source})
    f = dict(_id=fid, workspace=src["workspace"], source=source, person=src["person"], started_at=started,
             status="ready", pin=PIN, steps=n, duration_s=float(n))
    db.footage.insert_one(f)
    IN.write_steps(db, f, [(float(j), float(j + 1), np.zeros(4, np.float32), ids[j], lps[j]) for j in range(n)])
    return f


def test_a_name_keeps_its_only_words_a_description_is_searched_and_a_colour_is_never_a_place(db):
    from cyclopsdiary import objects as OB
    tok = FakeTok()
    assert OB.name_words(tok, "my keys") == [[[20]]]                             # "my" dropped
    assert OB.name_words(tok, "can") == [[[59]]]                                 # a can is a can, not a stop word
    _people(db)
    _footage3(db, "fa", "alex-glasses", _t(0), [(3, 6, 22), (3, 6, 24)], seed=1)  # the keys on a blue chair
    w = OB.add(db, "alex", "keys", tok, describe="keys chair")
    tr = OB.get(db, "alex", "keys")
    assert tr["describe"] == "keys chair" and tr["words"] == [[[20], [22]]]      # one phrase, both words
    assert "blue" not in w["last"]["place"] + w["last"]["end_place"]
    OB.add(db, "alex", "keys", tok)                                              # named again: still described
    assert OB.get(db, "alex", "keys")["describe"] == "keys chair"


def test_an_object_is_found_by_the_models_own_words_for_it_and_stays_its_owners(db):
    from cyclopsdiary import objects as OB
    _people(db)
    _footage3(db, "fa", "alex-glasses", _t(0), [(3, 6, 21)], seed=1)          # the keys on the table
    _footage3(db, "fb", "bea-phone", _t(600), [(10, 14, 22)], seed=2)          # later, on the chair
    w = OB.add(db, "alex", "keys", FakeTok())
    assert [(s["footage"], s["t0"], s["t1"], s["place"][0]) for s in w["trail"]] == \
        [("fa", 3.0, 6.0, "table"), ("fb", 10.0, 14.0, "chair")]
    assert w["last"]["footage"] == "fb" and w["last"]["end_place"][0] == "chair"
    assert "keys" not in w["last"]["place"]                                     # its own word is not its place
    assert OB.objects(db, "bea") == []                                          # bea sees none of alex's
    with pytest.raises(ValueError):
        OB.where(db, "bea", "keys")
    OB.add(db, "bea", "keys", FakeTok())                                        # her own keys, her own sightings
    assert {s["owner"] for s in db.sightings.find({"tracker": OB.get(db, "alex", "keys")["_id"]})} == {"alex"}
    assert [o["name"] for o in OB.objects(db, "bea")] == ["keys"]


def test_confirming_adds_an_example_rejecting_hides_a_span_and_new_footage_is_tracked(db):
    from cyclopsdiary import objects as OB
    _people(db)
    tok = FakeTok()
    _footage3(db, "fa", "alex-glasses", _t(0), [(3, 6, 21)], seed=1)
    _footage3(db, "fb", "bea-phone", _t(600), [(10, 14, 22)], seed=2)
    OB.add(db, "alex", "keys", tok)
    t = OB.trail(db, "alex", "keys")
    assert OB.confirm(db, "alex", t[0]["_id"])
    assert OB.get(db, "alex", "keys")["examples"] == [dict(footage="fa", t0=3.0, t1=6.0)]
    assert OB.reject(db, "alex", t[-1]["_id"])
    OB.track(db, "alex", "keys", tok)                                           # found again by name and example
    assert all(s["footage"] != "fb" for s in OB.trail(db, "alex", "keys"))      # the rejected span stays hidden
    _footage3(db, "fc", "bea-phone", _t(1200), [(2, 5, 23)], seed=3)            # a new clip: keys on the floor
    assert set(OB.track_workspace(db, "home", tok)) == {"alex/keys"}
    last = OB.last_known(db, "alex", "keys")
    assert last["footage"] == "fc" and last["end_place"][0] == "floor"


# ---- the router (router.py): the four everyday questions, answered from MongoDB with no model call

def test_the_rules_pick_the_tool_for_the_everyday_questions_and_leave_the_rest():
    from cyclopsdiary import router as RT
    assert RT.rules("Where are my keys?") == ("object_belief", dict(name="my keys"))
    assert RT.rules("have you seen the remote") == ("object_belief", dict(name="the remote"))
    assert RT.rules("what's on the chair?") == ("at_place", dict(place="the chair"))
    assert RT.rules("what objects do I have") == ("find_object", dict(name=""))
    tool, span = RT.rules("what happened between 3pm and 3:30 pm")
    assert tool == "between" and span["end"] - span["start"] == timedelta(minutes=30)
    _, span = RT.rules("what happened in the last 10 minutes")
    assert span["end"] - span["start"] == timedelta(minutes=10)
    assert RT.rules("tell me a joke") is None and RT.rules("what happened to the roman empire") is None
    mine = {"key": "keys", "car key": "car keys", "wallet": "wallet"}                  # names(db, person)
    assert RT.rules("any idea where I dropped my keys earlier?", mine) == ("object_belief", dict(name="keys"))
    assert RT.rules("which camera saw my car keys last?", mine) == ("object_belief", dict(name="car keys"))
    assert RT.rules("when did I last see my wallet", mine) == ("object_belief", dict(name="wallet"))
    assert RT.rules("where are my keys and wallet?", mine) is None                     # two objects: the model
    assert RT.rules("where is my passport?", mine) == ("object_belief", dict(name="my passport"))   # "I don't know"
    for said in ("I put my keys on the table", "remember my keys live on the hall table?", "what color are my keys?"):
        assert RT.rules(said, mine) is None                                         # a statement, a note, no cue


def test_the_router_answers_from_the_trail_worded_by_how_long_ago_and_only_for_the_owner(db, monkeypatch):
    from cyclopsdiary import objects as OB
    from cyclopsdiary import router as RT
    _people(db)
    _footage3(db, "fa", "alex-glasses", _t(0), [(3, 6, 21)], seed=1)          # the keys on the table
    _footage3(db, "fb", "bea-phone", _t(600), [(10, 14, 22)], seed=2)          # later, on the chair
    OB.add(db, "alex", "keys", FakeTok())
    monkeypatch.setattr(RT, "_now", lambda: _t(614 + 60))                       # a minute after the chair
    r = RT.ask(db, "alex", "Where are my keys?", use_model=False)
    assert (r["how"], r["tool"], r["result"]["status"]) == ("rules", "object_belief", "seen")
    last = r["result"]["last"]
    assert (last["camera"], last["near"][0], last["clip"]["footage"], last["clip"]["t0"], last["clip"]["t1"]) == \
        ("bea-phone", "chair", "fb", 10.0, 14.0)
    assert "near the chair" in r["answer"] and "Probably" not in r["answer"]
    monkeypatch.setattr(RT, "_now", lambda: _t(614 + 3600))                    # an hour on: believed, not seen
    assert RT.ask(db, "alex", "where are my keys", use_model=False)["answer"].endswith("Probably still there.")
    assert [o["name"] for o in RT.ask(db, "alex", "what's on the chair", use_model=False)["result"]["objects"]] == ["keys"]
    assert RT.ask(db, "alex", "what's on the table", use_model=False)["result"]["objects"] == []    # it moved
    assert RT.ask(db, "bea", "where are my keys", use_model=False)["result"]["status"] == "unknown"   # alex's alone
    got = RT.between(db, "alex", _t(0), _t(3600))["happened"]
    assert [(h["source"], h["what"], h["footage"]) for h in got] == [("sighting", "keys", "fa"), ("sighting", "keys", "fb")]
    assert RT.ask(db, "alex", "tell me a joke", use_model=False)["answer"] is None


def test_kept_reads_are_dropped_on_a_change_and_a_read_racing_a_change_is_not_kept():
    from cyclopsdiary import router as RT
    k, n = RT._Kept(), []
    read = lambda: n.append(1) or len(n)                                       # noqa: E731 -- counts the reads
    assert (k.get("a", read), k.get("a", read)) == (1, 2)                       # no change stream: every answer reads
    k.live = True
    assert (k.get("a", read), k.get("a", read)) == (3, 3)                       # kept
    k.changed()
    assert k.get("a", read) == 4                                                # a change drops it

    def racing():
        k.changed()                                                             # a change lands during the read
        return read()
    assert (k.get("b", racing), k.get("b", read)) == (5, 6)                     # so that read was not kept


def test_a_change_stream_keeps_the_routers_reads_until_the_trail_changes(monkeypatch):
    import time
    rs = os.environ.get("CYCLOPSDIARY_TEST_RS_URI")
    if not rs:
        pytest.skip("CYCLOPSDIARY_TEST_RS_URI unset: change streams need a replica set")
    from cyclopsdiary import atlas
    from cyclopsdiary import objects as OB
    from cyclopsdiary import router as RT
    c = pymongo.MongoClient(rs, serverSelectionTimeoutMS=3000, tz_aware=True)
    db = c[f"cyclopsdiary_test_{uuid.uuid4().hex[:8]}"]
    atlas.ensure(db)
    monkeypatch.setattr(RT, "_kept", RT._Kept())
    try:
        _people(db)
        _footage3(db, "fa", "alex-glasses", _t(0), [(3, 6, 21)], seed=1)
        _footage3(db, "fb", "bea-phone", _t(600), [(10, 14, 22)], seed=2)
        tok = FakeTok()
        OB.add(db, "alex", "keys", tok)
        assert RT.watch(db)
        near = lambda: RT.ask(db, "alex", "where are my keys", use_model=False)["result"]["last"]["near"][0]  # noqa: E731
        assert near() == "chair"
        v = RT._kept.version
        assert near() == "chair" and RT._kept.data                              # answered from memory
        _footage3(db, "fc", "bea-phone", _t(1200), [(2, 5, 23)], seed=3)        # a new clip: keys on the floor
        OB.track_workspace(db, "home", tok)
        for _ in range(50):                                                     # the stream's lag
            if RT._kept.version != v:
                break
            time.sleep(0.1)
        assert RT._kept.version != v and near() == "floor"
    finally:
        c.drop_database(db.name)
        c.close()


# ---- the demo stage: its API over MongoDB (stage.py) and how the tracker finds an object (searchview.py)

class _Tok:
    def decode(self, ids):
        return " ".join(f"w{int(i)}" for i in ids)


def _keys(db):
    """alex tracks "keys" by name: the synthetic words 4 and 5 (phase 1: the keys on the table)."""
    from cyclopsdiary import objects as OB
    _people(db)
    _footage2(db, "fa", "alex-glasses", _t(0), PUT_A, seed=1)
    _footage2(db, "fb", "bea-phone", _t(3600), PUT_B + TAKE_B, seed=2)
    db.trackers.insert_one(dict(workspace="home", owner="alex", name="keys", words=[[[4, 5]]], aliases=[],
                                created_at=_t(0)))
    OB.track(db, "alex", "keys", _Tok())


def test_the_stage_shows_how_the_tracker_found_the_object_step_by_step(db):
    from cyclopsdiary import objects as OB
    from cyclopsdiary import searchview as SV
    _keys(db)
    got = SV.explain_object(db, "alex", "keys", _Tok())
    assert got["object"] == "keys" and got["words"] == [["w4", "w5"]] and got["cut"] is not None
    assert [(ln["footage"], ln["source"]) for ln in got["lanes"]] == [("fa", "alex-glasses"), ("fb", "bea-phone")]
    assert len(got["scores"]) == 60 and all(len(ln["steps"]) == 30 for ln in got["lanes"])
    present = {(ln["footage"], st["t0"]) for ln in got["lanes"] for st in ln["steps"] if st["present"]}
    sighted = {(s["footage"], t) for s in OB.trail(db, "alex", "keys") for t in np.arange(s["t0"], s["t1"], 1.0)}
    assert present == sighted and present                       # the stage draws exactly what the tracker decided
    assert got["last"]["footage"] == "fb"
    with pytest.raises(ValueError):
        SV.explain_object(db, "bea", "keys", _Tok())            # bea has no keys: alex's stay alex's


def test_the_stage_api_serves_state_media_and_a_recording_as_it_grows(db, tmp_path):
    import threading
    import time
    from starlette.testclient import TestClient
    from cyclopsdiary import stage as ST
    _keys(db)
    clip = tmp_path / "fb.webm"
    clip.write_bytes(b"0123456789" * 10)
    db.footage.update_one({"_id": "fb"}, {"$set": {"path": str(clip), "mime": "video/webm;codecs=vp8",
                                                   "status": "live"}})
    with TestClient(ST.app(db, ST.Hub(), _Tok())) as c:
        st = c.get("/api/state?person=alex").json()
        assert {x["_id"] for x in st["cameras"]} == {"alex-glasses", "bea-phone"}
        assert [o["name"] for o in st["objects"]] == ["keys"] and st["trails"]["keys"]
        assert c.get("/api/state?person=bea").json()["objects"] == []           # private per person
        r = c.get("/media/fb", headers={"Range": "bytes=0-9"})
        assert r.status_code == 206 and r.content == b"0123456789"
        assert c.get("/media/nope").status_code == 404
        assert c.get("/api/object?person=alex&name=keys").json()["lanes"]

        def grow():
            time.sleep(0.4)
            with open(clip, "ab") as f:
                f.write(b"abcdefghij")
            time.sleep(0.4)
            db.footage.update_one({"_id": "fb"}, {"$set": {"status": "ready"}})

        threading.Thread(target=grow, daemon=True).start()
        with c.websocket_connect("/video/fb") as ws:
            assert ws.receive_json()["mime"] == "video/webm;codecs=vp8"
            got = b""
            while True:
                m = ws.receive()
                if m.get("bytes"):
                    got += m["bytes"]
                elif m.get("text"):
                    assert json.loads(m["text"]) == {"end": True}
                    break
        assert got == clip.read_bytes() == b"0123456789" * 10 + b"abcdefghij"


def test_a_change_in_mongodb_reaches_the_feed_and_sightings_only_their_owner(db):
    from starlette.testclient import TestClient
    from cyclopsdiary import stage as ST
    _keys(db)
    tw = ST.TrackerWords(db)
    step = {"operationType": "insert", "ns": {"coll": "steps"}, "fullDocument": {
        "_id": "fb:6", "footage": "fb", "source": "bea-phone", "person": "bea", "i": 6, "t0": 6.0, "t1": 7.0,
        "observed_at": _t(3606), "ids": np.array([9, 5, 4], np.int32).tobytes()}}
    ev = ST.shape(step, _Tok(), tw)
    assert ev["kind"] == "step" and ev["words"][:3] == ["w9", "w5", "w4"] and ev["ranks"] == {"alex/keys": 1}
    seen = {"operationType": "update", "ns": {"coll": "sightings"}, "fullDocument": {
        "_id": "s1", "owner": "alex", "source": "bea-phone", "footage": "fb", "t0": 6.0, "t1": 8.0}}
    ev2 = ST.shape(seen, _Tok(), tw)
    assert ST.visible(ev2, "alex") and not ST.visible(ev2, "bea")
    assert ST.visible(ev, "bea") and ST.visible(ev, "bea")["ranks"] == {}           # bea sees the step, not alex's keys
    hub = ST.Hub()
    with TestClient(ST.app(db, hub, _Tok())) as c:
        with c.websocket_connect("/feed?person=alex") as ws:
            assert ws.receive_json()["kind"] == "hello"
            hub.publish("feed", ev2)
            hub.publish("feed", ev)
            assert ws.receive_json()["kind"] == "sighting"
            assert ws.receive_json()["ranks"] == {"alex/keys": 1}


def test_the_stage_reads_the_codec_a_player_needs_from_the_recording_itself(tmp_path):
    from cyclopsdiary import stage as ST
    webm, mp4 = tmp_path / "a.webm", tmp_path / "b.mp4"
    webm.write_bytes(_recording("video/webm")[:4096])                  # the first chunk is enough
    mp4.write_bytes(_recording("video/mp4")[:4096])
    assert ST.codec(webm, "video/webm") == 'video/webm; codecs="vp8"'
    assert re.fullmatch(r'video/mp4; codecs="avc1\.[0-9A-F]{6}"', ST.codec(mp4, "video/mp4"))   # profile, flags, level
    assert ST.codec(webm, "video/x-unknown") == "video/x-unknown"      # nothing known: the row's own mime
    assert ST._mime({"path": "x/IMG_1.MOV"}) == "video/mp4"             # a .mov plays as ISO media


def test_the_stage_lists_a_recordings_steps_with_only_the_askers_objects_ranked(db):
    from starlette.testclient import TestClient
    from cyclopsdiary import stage as ST
    _keys(db)
    with TestClient(ST.app(db, ST.Hub(), _Tok())) as c:
        alex = c.get("/api/steps?footage=fb&person=alex").json()
        bea = c.get("/api/steps?footage=fb&person=bea").json()
    assert [s["t0"] for s in alex] == [float(j) for j in range(30)] and all(len(s["words"]) == 6 for s in alex)
    assert alex[6]["ranks"]["alex/keys"] == 0                   # phase 1 at 6-8 s: words 4-7 lead the step
    assert all(s["ranks"] == {} for s in bea)                   # bea sees the words, never alex's keys
