# CyclopsDiary agent

A Strands agent, scaffolded with the AWS Agent Development Toolkit (`adt init`). Its memory is MongoDB.
The Cosmos side reaches it as MCP tools.

| | |
|---|---|
| `.agent.yaml` | the model (OpenRouter: `AGENT_MODEL` and `OPENROUTER_API_KEY` from the project's `.env`), the MCP servers, and the workspace, person, session, window and extraction |
| `src/agent.py` | builds the agent. `adt dev` loads its module-level `agent` |
| `src/cyclops.py` | MongoDB behind Strands' own interfaces: `MongoSessionRepository` (the conversation, written in the background), `MongoMemoryStore` (notes and earlier conversations, found by a text index in one aggregation), the workspace context in the system prompt, and `RouterModel` (the router around the model, with a hedged request) |
| `check.py` | the wiring checked on the real database: with a stand-in model that is never called, or one real turn with `--ask` |
| `check_router_model.py` | `RouterModel` checked with fake models, no network |
| `bench_*.py` | latency, written to `.local/agent_bench/`: `bench_turns` (question to answer), `bench_phases` (where a turn's time goes), `bench_model` (OpenRouter routes), `bench_conn` (new vs kept connections), `bench_hedge` (the hedge off and on) |

```bash
python3 -m venv .venv-agent && .venv-agent/bin/pip install -r agent/requirements.txt   # plus ADT, see requirements.txt
.venv-agent/bin/python agent/check.py          # tools listed and called, session and memory rows written, then deleted
.venv-agent/bin/python agent/check.py --ask "where are the keys?"   # one real turn through OpenRouter, then deleted
cd agent && ../.venv-agent/bin/adt dev --port 8083   # chat UI: builds its UI with npm on first run
```

The agent's tools are the design doc's four, from `bin/cyclopsdiary mcp --person`: `find_object`, `object_belief`,
`at_place` and `between`, plus `search_memory` and `add_memory` from Strands' MemoryManager. A turn is answered
the fastest way that works:

1. The question is one the router's rules answer (`ask` over MCP, rules only). The answer comes from MongoDB,
   with no memory search and no model call: about 2 ms, since the MCP server keeps its reads. A change stream
   drops them when anything changes.
2. Otherwise the memory search runs (one aggregation, about 0.1 s) and the model makes one call. If it picks
   one of the four tools, that tool's worded answer is the reply, with no second call to word it.
3. Any other tool (the memory tools) runs as usual.

A model call that has not started after `hedge_after_s` (0.8 s) is sent again to the next provider in
OpenRouter's order, and the first to start is kept (a hedged request, Dean & Barroso, "The Tail at Scale").

The model is `AGENT_MODEL` on OpenRouter (deepseek/deepseek-v4.1-flash), never Claude or the Anthropic API.
Reasoning is off, and the provider order starts with Together, measured fastest (ledger L-11). Replies are one
or two sentences. Memory extraction is off because the turn it fired on waited for it. The AWS toolkit was
archived by AWS on 2026-07-06 and gets no fixes; everything except `adt dev` works without it.
