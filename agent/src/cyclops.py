"""The agent's memory, context and session in MongoDB, as Strands' own interfaces.

  MongoSessionRepository  Strands' SessionRepository over agent_sessions / agent_agents / agent_messages:
                          the conversation survives restarts and any machine can carry it on.
  MongoMemoryStore        Strands' MemoryStore over the person's notes (agent_notes) and the lines of their
                          earlier conversations, found by MongoDB's text index (words, no embeddings, no
                          cosine). MemoryManager gives the agent search_memory / add_memory and folds the
                          entries that match into each model call.
  RouterModel             the model with the router around it: a fresh question the router's rules answer (the
                          four everyday ones, design doc section 7) is answered from MongoDB with no model call;
                          otherwise the model's one call picks the tool, and when that is one of the four, the
                          tool's own worded answer is the reply (no second call to word it).
  system_prompt           the configured prompt plus the workspace as it stands (agentmemory.context).

The storage itself is cyclopsdiary.agentmemory, shared with `bin/cyclopsdiary mcp`.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from strands.agent.conversation_manager import SlidingWindowConversationManager  # noqa: E402
from strands.memory import MemoryManager  # noqa: E402
from strands.memory.types import MemoryEntry, MemoryStore  # noqa: E402
from strands.models.model import Model  # noqa: E402
from strands.session.repository_session_manager import RepositorySessionManager  # noqa: E402
from strands.session.session_repository import SessionRepository  # noqa: E402
from strands.types.exceptions import SessionException  # noqa: E402
from strands.types.session import Session, SessionAgent, SessionMessage  # noqa: E402

from cyclopsdiary import agentmemory as AM  # noqa: E402
from cyclopsdiary import atlas  # noqa: E402
from cyclopsdiary import router as RT  # noqa: E402  (its rules only: no numpy here)


class MongoSessionRepository(SessionRepository):
    """Strands sessions in MongoDB, each stamped with the workspace and person it belongs to. A new message is
    written by a background thread in order, so a turn does not wait on it; any read waits for the writes."""

    def __init__(self, db, workspace: str, person: str | None = None):
        import atexit
        import queue
        import threading
        self.db, self.workspace, self.person = db, workspace, person
        self._q = queue.Queue()
        threading.Thread(target=self._write, name="session-writer", daemon=True).start()
        atexit.register(self._q.join)

    def _write(self):
        while True:
            fn, a = self._q.get()
            try:
                fn(*a)
            except Exception as e:                             # noqa: BLE001 -- logged, the conversation goes on
                print(f"session write failed: {type(e).__name__}: {e}", file=sys.stderr)
            finally:
                self._q.task_done()

    def create_session(self, session: Session, **kwargs) -> Session:
        try:
            AM.session_create(self.db, session.to_dict(), self.workspace, self.person)
        except ValueError as e:
            raise SessionException(str(e)) from None
        return session

    def read_session(self, session_id: str, **kwargs) -> Session | None:
        d = AM.session_read(self.db, session_id)
        return Session.from_dict(d) if d else None

    def create_agent(self, session_id: str, session_agent: SessionAgent, **kwargs) -> None:
        AM.agent_create(self.db, session_id, session_agent.to_dict())

    def read_agent(self, session_id: str, agent_id: str, **kwargs) -> SessionAgent | None:
        d = AM.agent_read(self.db, session_id, agent_id)
        return SessionAgent.from_dict(d) if d else None

    def update_agent(self, session_id: str, session_agent: SessionAgent, **kwargs) -> None:
        if not AM.agent_update(self.db, session_id, session_agent.to_dict()):
            raise SessionException(f"Agent {session_agent.agent_id} in session {session_id} does not exist")

    def create_message(self, session_id: str, agent_id: str, session_message: SessionMessage, **kwargs) -> None:
        self._q.put((AM.message_create, (self.db, session_id, agent_id, session_message.to_dict())))

    def read_message(self, session_id: str, agent_id: str, message_id: int, **kwargs) -> SessionMessage | None:
        self._q.join()
        d = AM.message_read(self.db, session_id, agent_id, message_id)
        return SessionMessage.from_dict(d) if d else None

    def update_message(self, session_id: str, agent_id: str, session_message: SessionMessage, **kwargs) -> None:
        self._q.join()
        if not AM.message_update(self.db, session_id, agent_id, session_message.to_dict()):
            raise SessionException(f"Message {session_message.message_id} does not exist")

    def list_messages(self, session_id: str, agent_id: str, limit: int | None = None, offset: int = 0,
                      **kwargs) -> list[SessionMessage]:
        self._q.join()
        return [SessionMessage.from_dict(d) for d in AM.messages(self.db, session_id, agent_id, limit, offset)]


class MongoMemoryStore(MemoryStore):
    """The person's notes and earlier conversations in MongoDB, found by their words."""

    def __init__(self, db, workspace: str, person: str, session_id: str | None = None, name: str = "notes",
                 max_search_results: int = 5, extract: bool = False):
        self.db, self.workspace, self.person, self.session_id = db, workspace, person, session_id
        self.name, self.max_search_results = name, max_search_results
        self.description = f"{person}'s notes and earlier conversations, kept in MongoDB"
        # extract: every few turns the agent's own model distils the conversation into notes (Strands'
        # ModelExtractor, written through add below)
        self.writable, self.extraction = True, bool(extract)

    async def search(self, query: str, options=None) -> list[MemoryEntry]:
        n = (options or {}).get("max_search_results") or self.max_search_results
        got = await asyncio.to_thread(AM.recall, self.db, self.workspace, self.person, text=query, limit=n,
                                      exclude_session=self.session_id)
        out = [MemoryEntry(content=f"{x['subject']}: {x['note']}",
                           metadata=dict(kind="note", id=str(x["_id"]), noted_at=x["created_at"].isoformat()))
               for x in got["notes"]]
        out += [MemoryEntry(content=f"{s['role']} said: {s['text']}", metadata=dict(kind="said", session=s["session"]))
                for s in got["said"]]
        return out[:n]

    async def add(self, content: str, metadata=None) -> str:
        """A note; its first line (a markdown heading from extraction, or a subject) names it."""
        head, _, body = content.strip().partition("\n")
        subject = head.lstrip("#").strip()[:200] or "note"
        doc = await asyncio.to_thread(AM.remember, self.db, self.workspace, self.person, subject,
                                      (body.strip() or content.strip())[:4000], session=self.session_id)
        return str(doc["_id"])


def _fresh_question(messages) -> str | None:
    """The person's new question: the last message is theirs and carries no tool result. Memory folded in by
    the MemoryManager is a later <memory> block and is left out."""
    if not messages or messages[-1]["role"] != "user":
        return None
    blocks = messages[-1]["content"]
    if any("toolResult" in b for b in blocks):
        return None
    words = [b["text"] for b in blocks if "text" in b and not b["text"].lstrip().startswith("<memory>")]
    return words[0].strip() if words else None


def _said(text: str, usage=None) -> list:
    """A reply of one text block, as a model's stream events."""
    return [{"messageStart": {"role": "assistant"}}, {"contentBlockStart": {"start": {}}},
            {"contentBlockDelta": {"delta": {"text": text}}}, {"contentBlockStop": {}},
            {"messageStop": {"stopReason": "end_turn"}},
            usage or {"metadata": {"usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                                   "metrics": {"latencyMs": 0}}}]


def _mcp_answer(tool, name: str, args: dict) -> str | None:
    """The `answer` of an MCP tool of `bin/cyclopsdiary mcp`; None on any failure."""
    try:
        r = tool.mcp_client.call_tool_sync(f"route-{uuid.uuid4().hex[:8]}", name, args)
        if r.get("status") != "success":
            return None
        got = r.get("structuredContent") or json.loads(r["content"][0]["text"])
        return got.get("result", got).get("answer")
    except Exception:                                          # noqa: BLE001 -- fail open to the model
        return None


def _next_provider(model: Model) -> Model:
    """The same model with OpenRouter's provider order turned by one, so a hedge goes to another provider."""
    import copy
    h = copy.copy(model)
    cfg = copy.deepcopy(getattr(model, "config", {}))
    route = ((cfg.get("params") or {}).get("extra_body") or {}).get("provider") or {}
    if len(route.get("order") or []) > 1:
        route["order"] = route["order"][1:] + route["order"][:1]
    h.config = cfg
    return h


class RouterModel(Model):
    """The configured model with the router around it. route: question -> answer text or None (rules, no
    model). tools: the four tools by name, answered from their worded result when the model picks one.
    hedge_after_s: a model call with no first event by then is sent again to the next provider, and the
    first of the two to start is kept (a hedged request, Dean & Barroso, The Tail at Scale, CACM 2013)."""

    def __init__(self, inner: Model, route, tools: dict | None = None, hedge_after_s: float | None = None):
        self.inner, self.route, self.tools = inner, route, dict(tools or {})
        self.config = getattr(inner, "config", {})
        self.hedge_after_s = hedge_after_s
        self.hedge = _next_provider(inner) if hedge_after_s else None
        self.hedged = dict(calls=0, sent=0, won=0)
        self._asked = None

    def update_config(self, **kw):
        self.inner.update_config(**kw)

    def get_config(self):
        return self.inner.get_config()

    def structured_output(self, *a, **kw):
        return self.inner.structured_output(*a, **kw)

    def answer(self, q: str, keep: bool = False) -> str | None:
        """The router's answer to this turn's question, asked once: the memory trigger asks first (keep),
        and the model call takes it from there."""
        got, self._asked = self._asked, None
        a = got[1] if got and got[0] == q and time.monotonic() - got[2] < 5 else self.route(q)
        if keep:
            self._asked = (q, a, time.monotonic())
        return a

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kw):
        q = _fresh_question(messages) if tool_specs else None     # memory extraction calls with no tools
        answer = await asyncio.to_thread(self.answer, q) if q else None
        if answer:
            for ev in _said(answer):
                yield ev
            return
        now = f"Now: {datetime.now().astimezone().isoformat(timespec='seconds')}."   # for times like "since 3pm"
        system_prompt = f"{system_prompt or ''}\n\n{now}"
        if kw.get("system_prompt_content") is not None:
            kw["system_prompt_content"] = [*kw["system_prompt_content"], {"text": now}]
        held, uses = [], []                                    # from the first tool call on, held to the end
        async for ev in self._raced(messages, tool_specs, system_prompt, kw):
            start = ev.get("contentBlockStart", {}).get("start", {})
            if "toolUse" in start:
                uses.append(dict(name=start["toolUse"]["name"], input=""))
            if not uses:
                yield ev
                continue
            d = ev.get("contentBlockDelta", {}).get("delta", {})
            if "toolUse" in d:
                uses[-1]["input"] += d["toolUse"].get("input") or ""
            held.append(ev)
        said = None
        if len(uses) == 1 and uses[0]["name"] in self.tools:
            try:
                args = json.loads(uses[0]["input"] or "{}")
            except json.JSONDecodeError:
                args = None
            if isinstance(args, dict):
                t = self.tools[uses[0]["name"]]
                said = await asyncio.to_thread(_mcp_answer, t, t.mcp_tool.name, args)
        if said is None:                                       # anything else: the agent runs the tools
            for ev in held:
                yield ev
            return
        usage = next((ev for ev in held if "metadata" in ev), None)
        for ev in _said(said, usage)[1:]:                      # the message has started already
            yield ev


    async def _raced(self, messages, tool_specs, system_prompt, kw):
        """The inner model's stream, hedged: no first event in hedge_after_s (or a failure before one), and the
        same request goes to the next provider; the arm that starts first is kept, the other cancelled."""
        self.hedged["calls"] += 1
        if not self.hedge:
            async for ev in self.inner.stream(messages, tool_specs, system_prompt, **kw):
                yield ev
            return
        q: asyncio.Queue = asyncio.Queue()

        async def pump(arm, model):
            try:
                async for ev in model.stream(messages, tool_specs, system_prompt, **kw):
                    await q.put((arm, ev, None))
                await q.put((arm, None, None))
            except Exception as e:                             # noqa: BLE001 -- handed to the reader
                await q.put((arm, None, e))

        arms, failed, winner = {0: asyncio.create_task(pump(0, self.inner))}, {}, None
        try:
            while True:
                if winner is None and len(arms) == 1:
                    try:
                        arm, ev, err = await asyncio.wait_for(q.get(), self.hedge_after_s)
                    except asyncio.TimeoutError:
                        arms[1] = asyncio.create_task(pump(1, self.hedge))
                        self.hedged["sent"] += 1
                        continue
                else:
                    arm, ev, err = await q.get()
                if winner is None:
                    if ev is None:                             # failed (or ended empty) before starting
                        failed[arm] = err
                        if len(arms) == 1:                     # the hedge takes over at once
                            arms[1] = asyncio.create_task(pump(1, self.hedge))
                            self.hedged["sent"] += 1
                        elif len(failed) == 2:
                            raise failed[0] or failed[1] or RuntimeError("the model returned nothing")
                        continue
                    winner = arm
                    self.hedged["won"] += arm
                    for a, t in arms.items():
                        if a != arm:
                            t.cancel()
                if arm != winner:
                    continue
                if err is not None:
                    raise err
                if ev is None:
                    return
                yield ev
        finally:
            for t in arms.values():
                t.cancel()


def mcp_router(ask_tool):
    """route() through the MCP server's `ask` tool, rules only (the agent's own model picks the rest); any
    failure means "not answered here" and the model takes it."""
    def route(question: str) -> str | None:
        return _mcp_answer(ask_tool, ask_tool.mcp_tool.name, {"question": question, "use_model": False})
    return route


def settings(cfg: dict) -> dict:
    c = dict(workspace="home", person=None, session="daily", window=40, extract=False, hedge_after_s=None) | \
        (cfg.get("cyclopsdiary") or {})
    if c["session"] == "daily":
        c["session"] = f"{c['workspace']}-{c['person'] or 'anyone'}-{date.today().isoformat()}"
    return c


def database():
    db = atlas.database()
    atlas.ensure(db)
    return db


def session_manager(db, c: dict) -> RepositorySessionManager:
    return RepositorySessionManager(session_id=c["session"],
                                    session_repository=MongoSessionRepository(db, c["workspace"], c["person"]))


def memory_manager(db, c: dict, answer=None) -> MemoryManager | None:
    """answer: question -> the router's answer or None (RouterModel.answer); a question it answers needs no
    memory search. Without it, the router's rules decide."""
    if not c["person"]:
        return None                                            # notes are a person's own

    def inject(ctx) -> bool:
        q = _fresh_question(ctx.messages)
        return q is not None and (answer(q, keep=True) if answer else RT.rules(q)) is None
    return MemoryManager(stores=[MongoMemoryStore(db, c["workspace"], c["person"], c["session"], extract=c["extract"])],
                         add_tool_config=True, injection={"trigger": inject})


def conversation_manager(c: dict) -> SlidingWindowConversationManager:
    return SlidingWindowConversationManager(window_size=int(c["window"]))


def system_prompt(db, cfg: dict, c: dict) -> str:
    return (cfg.get("system_prompt") or "").strip() + "\n\nThe workspace now:\n" + \
        AM.render(AM.context(db, c["workspace"], c["person"]))
