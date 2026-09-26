"""CyclopsDiary's agent: `adt dev` (from agent/) loads the module-level `agent` below.

Wired from .agent.yaml:
  model    provider.class / provider.kwargs (keys ending in _env read from the environment)
  tools    the Cosmos side over MCP (mcp_servers: `bin/cyclopsdiary mcp`), plus any @tool in src/tools/
  memory   MongoDB, through Strands' own interfaces (src/cyclops.py): the conversation (session manager),
           the person's notes and earlier conversations (memory manager: search_memory, add_memory, and
           matching entries folded into each model call), the workspace context in the system prompt,
           and a sliding window of the conversation for the model
"""
import importlib
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from strands import Agent

from src import cyclops
from src.mcp_tools import get_mcp_tools_sync
from src.tools import get_tools

load_dotenv(cyclops.ROOT / ".env", override=False)        # MONGODB_URI, and OPENROUTER_API_KEY once it is there
os.environ.setdefault("STRANDS_OTEL_ENABLE_CONSOLE_EXPORT", "true")


def _resolve_env(obj):
    """Replace *_env keys with the environment variable they name (recursively)."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k.endswith("_env"):
                val = os.getenv(v)
                if val is None:
                    raise RuntimeError(f"Environment variable '{v}' is not set (put it in the project's .env)")
                out[k[:-4]] = val
            else:
                out[k] = _resolve_env(v)
        return out
    if isinstance(obj, list):
        return [_resolve_env(i) for i in obj]
    return obj


def load_config() -> dict:
    p = Path(__file__).parent.parent / ".agent.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def load_model(cfg: dict):
    provider = cfg.get("provider", {})
    fqcn = provider.get("class")
    if not fqcn:
        raise ValueError("Missing 'provider.class' in .agent.yaml")
    module_path, class_name = fqcn.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)(**_resolve_env(provider.get("kwargs", {})))


FOUR = ("find_object", "object_belief", "at_place", "between")     # the agent's tools (design doc section 7)


def create_agent(model=None, router: bool = True, **kw) -> Agent:
    """model: a Strands model, or None for the one .agent.yaml names. router: answer the four everyday
    questions from MongoDB before any model call. kw: more Agent arguments."""
    cfg = load_config()
    c = cyclops.settings(cfg)
    db = cyclops.database()
    servers = [dict(s) for s in cfg.get("mcp_servers", [])]
    for s in servers:                                          # the tools answer for the agent's person
        if s.get("name") == "cyclopsdiary" and c["person"] and "--person" not in s["command"]:
            s["command"] = [*s["command"], "--person", c["person"]]
    mcp = get_mcp_tools_sync(servers)
    ask = next((t for t in mcp if t.tool_name == "ask"), None)
    four = {t.tool_name: t for t in mcp if t.tool_name in FOUR}
    model = model if model is not None else load_model(cfg)
    if router and ask is not None:
        model = cyclops.RouterModel(model, cyclops.mcp_router(ask), four, hedge_after_s=c["hedge_after_s"])
    return Agent(
        model=model,
        tools=list(four.values()) + get_tools(),
        system_prompt=cyclops.system_prompt(db, cfg, c),
        agent_id="cyclopsdiary",
        session_manager=cyclops.session_manager(db, c),
        memory_manager=cyclops.memory_manager(db, c, model.answer if isinstance(model, cyclops.RouterModel) else None),
        conversation_manager=cyclops.conversation_manager(c),
        **kw,
    )


if os.environ.get("CYCLOPSDIARY_NO_AGENT") != "1":         # the wiring check builds its own
    agent = create_agent()
