"""The agent's wiring, checked on the real database.

    .venv-agent/bin/python agent/check.py                      no model is called
    .venv-agent/bin/python agent/check.py --ask "where are the keys?"    one real turn through the model

Builds the agent from .agent.yaml in a throwaway session. Without --ask the model is a stand-in that
refuses to be called: it lists the tools, asks the router (rules, no model), calls the four tools directly
(Strands records a direct call in the conversation, so it goes through the session into MongoDB), adds and
searches a memory. With --ask the configured model (OpenRouter) answers the question. Either way it reads
the rows back and deletes what it wrote.
"""
import asyncio
import os
import sys
import time
from pathlib import Path

os.environ["CYCLOPSDIARY_NO_AGENT"] = "1"
os.environ["STRANDS_OTEL_ENABLE_CONSOLE_EXPORT"] = "false"      # no span dump on stdout
ASK = sys.argv[sys.argv.index("--ask") + 1] if "--ask" in sys.argv else None
os.chdir(Path(__file__).resolve().parent)                   # mcp_servers commands are relative to agent/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from strands.models.model import Model  # noqa: E402

from src import agent as A  # noqa: E402
from src import cyclops  # noqa: E402


class NoModel(Model):
    """The check never asks a model anything."""
    def update_config(self, **kw): pass
    def get_config(self): return {}
    def structured_output(self, *a, **kw): raise RuntimeError("the wiring check calls no model")
    def stream(self, *a, **kw): raise RuntimeError("the wiring check calls no model")


t = time.monotonic()
cfg = A.load_config()
c = cyclops.settings(cfg) | {"session": f"wiring-check-{int(time.time())}"}
cyclops.settings = lambda cfg, _c=c: _c                     # a throwaway session, deleted below
agent = A.create_agent(model=None if ASK else NoModel(), callback_handler=None)
db = cyclops.database()
print(f"agent built in {time.monotonic() - t:.1f} s; tools: {sorted(agent.tool_names)}")
print("system prompt ends:", agent.system_prompt.splitlines()[-1][:160])
if ASK:
    t = time.monotonic()
    r = agent(ASK)
    u = r.metrics.accumulated_usage
    print(f"asked {ASK!r} of {agent.model.config['model_id']}: {time.monotonic() - t:.1f} s, {r.metrics.cycle_count} cycles, "
          f"tokens in {u.get('inputTokens')} / out {u.get('outputTokens')}")
    print("tools called:", {k: v.call_count for k, v in r.metrics.tool_metrics.items()})
    print("answer:", str(r).strip())
else:
    from datetime import datetime, timedelta, timezone
    t = time.monotonic()
    print("router (rules over MCP, no model):", agent.model.answer("where are my keys?"),
          f"({time.monotonic() - t:.2f} s)")
    now = datetime.now(timezone.utc)
    for name, args in (("find_object", dict(name="")), ("object_belief", dict(name="keys")),
                       ("at_place", dict(place="chair")),
                       ("between", dict(start=(now - timedelta(days=1)).isoformat(), end=now.isoformat()))):
        t = time.monotonic()
        got = getattr(agent.tool, name)(**args)
        print(f"{name} (MCP, direct call): {got['status']} in {time.monotonic() - t:.2f} s:",
              str(got["content"][0]["text"])[:110])
    mem = cyclops.MongoMemoryStore(db, c["workspace"], c["person"], c["session"])
    nid = asyncio.run(mem.add("# wiring check\nthe check wrote this note and deletes it"))
    found = asyncio.run(mem.search("wiring check"))
    print("memory add/search:", nid, "->", [e.content for e in found][:2])
agent._session_manager.session_repository._q.join()          # the background session writes land
rows = dict(sessions=db.agent_sessions.count_documents({"_id": c["session"]}),
            agents=db.agent_agents.count_documents({"session": c["session"]}),
            messages=db.agent_messages.count_documents({"session": c["session"]}),
            notes=db.agent_notes.count_documents({"session": c["session"]}))
print("rows in MongoDB for this session:", rows)
for coll, f in (("agent_sessions", {"_id": c["session"]}), ("agent_agents", {"session": c["session"]}),
                ("agent_messages", {"session": c["session"]}), ("agent_notes", {"session": c["session"]})):
    db[coll].delete_many(f)
for q in db.queries.find({"label": "wiring check"}, {"_id": 1}):
    db.events.delete_many({"query": q["_id"]})
    db.queries.delete_one({"_id": q["_id"]})
print("deleted the check's rows")
