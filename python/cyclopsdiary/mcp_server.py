"""CyclopsDiary's tools over MCP (stdio): `bin/cyclopsdiary mcp --workspace WS --person P`.

One server per person: notes and recall are private to the person the server was started for, as
trackers are. The Strands agent in agent/ starts it as an MCP server; so can Claude Code or any other
MCP client. Nothing may print to stdout here: stdout is the protocol.

It runs under mcp 1.x (the model's environment, which can also run the search) or mcp 2.x (the agent's
.venv-agent, where find_moments leaves the answering to `serve`). mcp 1.x logs only errors, so a 2.x
client's opening `server/discover` probe, which 1.x does not know, is not printed as a warning.
"""
from __future__ import annotations

from typing import Literal

try:
    from mcp.server.fastmcp import FastMCP as _Server
    _QUIET = dict(log_level="ERROR")
except ModuleNotFoundError:                                  # mcp 2.x renamed it
    from mcp.server.mcpserver import MCPServer as _Server
    _QUIET = {}

from . import tools

INSTRUCTIONS = """\
CyclopsDiary: a shared visual memory. Every camera in the workspace was read by a world model, one step
per second, into MongoDB. Ask by example: give a span of a recording (footage id and seconds from
`footage`) and get back the moments like it (forward) or the example undone (reverse: a key put on the
table -> the key taken away), each with its camera, person, world time and score; `apart` marks the
moments that stand out from the rest. Keep what the person tells you with `remember`; look back with
`recall` and `timeline`; `context` says what the workspace holds now. A person's objects are their own:
`add_object` names one (the world model's own words for it find it in every camera), `where_is` gives its
trail across the cameras and where it was last seen, with the model's words for what is near it."""

_tok = None


def _tokenizer():
    """The world model's tokenizer, loaded on first use (the model's own words; not the model)."""
    global _tok
    if _tok is None:
        from . import lastseen, tower
        tower.use_stack()
        _tok = lastseen.tokenizer()
    return _tok


def build(db, workspace: str, person: str | None = None):
    m = _Server("cyclopsdiary", instructions=INSTRUCTIONS, **_QUIET)

    @m.tool()
    def footage() -> dict:
        """The workspace's recordings, oldest first: footage id, camera (source), person, start time, length
        in seconds. An example for find_moments is a footage id plus start and end seconds within it."""
        return {"recordings": tools.footage(db, workspace)}          # a dict: a list would go out one item a block

    @m.tool()
    def find_moments(footage: str, start_s: float, end_s: float,
                     direction: Literal["forward", "reverse"] = "forward",
                     scope: Literal["all", "others", "after"] = "all", top: int = 10,
                     label: str | None = None) -> dict:
        """Find the moments in every camera's footage that match an example span of one recording.
        direction: forward finds moments like the example; reverse finds the example undone (the example of a
        key put on the table, reversed, finds the key being taken away). scope: all (every recording, the
        example's own span excluded), others (other recordings only), after (only moments after the example).
        label names what is asked ("keys off the table") so later timeline calls can find it.
        Returns the ranked moments with footage id, seconds, camera, person, world time (observed_at), score,
        and apart (true when the moment stands out from the rest)."""
        return tools.find_moments(db, workspace, footage, start_s, end_s, direction=direction, scope=scope,
                                  top=top, label=label, person=person)

    @m.tool()
    def moments(query_id: str) -> dict:
        """An earlier find_moments query and the moments it found, by the query id it returned."""
        return tools.moments(db, workspace, query_id)

    @m.tool()
    def timeline(label: str | None = None, kind: Literal["forward", "reverse"] | None = None,
                 since: str | None = None, until: str | None = None, apart_only: bool = True,
                 limit: int = 20) -> dict:
        """The moments every earlier query found, newest first in world time, each counted once. Filter by
        label words (e.g. "keys"), kind (forward or reverse), and ISO 8601 times since/until. The latest
        reverse moment for a label is the last time that thing was undone (e.g. the keys last taken away)."""
        return {"moments": tools.timeline(db, workspace, label=label, kind=kind, since=since, until=until,
                                          apart_only=apart_only, limit=limit)}

    @m.tool()
    def context() -> str:
        """What the workspace holds now: its cameras and recordings, the latest moments that stood apart,
        queries still open or failed, and the person's latest notes."""
        return tools.context(db, workspace, person)

    if person is not None:
        @m.tool()
        def remember(subject: str, note: str, query_id: str | None = None) -> dict:
            """Keep a note for this person across sessions: something they said or you concluded ("spare key:
            lives in the blue bowl"). query_id links the find_moments answer that backs it."""
            return tools.remember(db, workspace, person, subject, note, query_id=query_id)

        @m.tool()
        def recall(text: str | None = None, subject: str | None = None, limit: int = 10) -> dict:
            """This person's notes, by subject or by words (the latest when neither is given), and, when words
            are given, the earlier lines of their conversations that contain them."""
            return tools.recall(db, workspace, person, text=text, subject=subject, limit=limit)

        @m.tool()
        def my_objects() -> dict:
            """This person's own objects, each with where it was last seen (camera, time, the model's words near it)."""
            return {"objects": tools.my_objects(db, person)}

        @m.tool()
        def add_object(name: str, aliases: list[str] | None = None, footage: str | None = None,
                       start_s: float | None = None, end_s: float | None = None, describe: str | None = None) -> dict:
            """Start tracking one of this person's objects by name ("keys"; aliases like "key"). The world model's
            own words for the name find it in every camera's footage; give footage + start_s + end_s too when a
            span shows this person's particular one, and describe ("keys library card") when the name alone also
            names other things. Returns its trail and where it was last seen."""
            return tools.add_object(db, person, name, _tokenizer(), aliases=aliases or (), footage=footage,
                                    start_s=start_s, end_s=end_s, describe=describe)

        @m.tool()
        def where_is(name: str) -> dict:
            """Where this person's object was last seen: tracked again over every camera now. Returns the trail
            (each sighting: camera, person, world time from/until, footage seconds to show, the words near it)
            and last_seen."""
            return tools.where_is(db, person, name, _tokenizer())

        @m.tool()
        def confirm_sighting(sighting_id: str) -> dict:
            """The person says a sighting is their object: it is believed, and its span becomes an example."""
            return tools.confirm_sighting(db, person, sighting_id)

        @m.tool()
        def reject_sighting(sighting_id: str) -> dict:
            """The person says a sighting is not their object: it leaves the trail."""
            return tools.reject_sighting(db, person, sighting_id)

        # ---- the router and the agent's four read-only tools (design doc section 7; router.py). Each result
        # carries `answer`, the sentence worded from what was found, and the MongoDB queries behind it.

        import threading
        from . import router as _router
        if _router.watch(db):                                  # answers from memory, kept current by a change stream
            threading.Thread(target=_router.warm, args=(db, person), name="router-warm", daemon=True).start()

        def _said(r: dict) -> dict:
            from . import router
            return tools.plain(dict(r, answer=router.say(r)))

        @m.tool()
        def ask(question: str, use_model: bool = True) -> dict:
            """The router: a question that is one of the four everyday ones (where is X, what's at a place,
            what happened between two times, which objects) answered at once from MongoDB. answer is None when
            it is none of them. use_model false: rules only (a caller with its own model picks the rest)."""
            from . import router
            r = router.ask(db, person, question, use_model=use_model)
            last = (r.get("result") or {}).get("last") or {}
            return tools.plain(dict(answer=r["answer"], how=r["how"], tool=r.get("tool"), ms=r["ms"],
                                    clip=last.get("clip")))

        @m.tool()
        def find_object(name: str = "") -> dict:
            """Which of this person's objects answer to a name (all of them when the name is empty), and where
            each was last seen."""
            from . import router
            return _said(router.find_object(db, person, name))

        @m.tool()
        def object_belief(name: str) -> dict:
            """Where one of this person's objects is believed to be now: last seen where (the world model's
            words near it), when, on which camera, with the clip (footage file and seconds) that shows it.
            status is seen, believed (seen a while ago: probably still there), not_seen or unknown."""
            from . import router
            return _said(router.object_belief(db, person, name))

        @m.tool()
        def at_place(place: str) -> dict:
            """Which of this person's objects were last seen at a place (table, chair, counter ...)."""
            from . import router
            return _said(router.at_place(db, person, place))

        @m.tool()
        def between(start: str, end: str) -> dict:
            """What happened between two times (ISO 8601 with the time zone): sightings of this person's
            objects and the moments the cameras found, in time order."""
            from datetime import datetime
            from . import router
            when = [datetime.fromisoformat(x.replace("Z", "+00:00")) for x in (start, end)]
            when = [w if w.tzinfo else w.astimezone() for w in when]
            return _said(router.between(db, person, *when))

    return m


def main(workspace: str, person: str | None = None) -> None:
    from . import atlas
    db = atlas.database()
    atlas.ensure(db)
    build(db, workspace, person).run()                    # stdio
