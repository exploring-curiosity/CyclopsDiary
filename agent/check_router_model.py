"""RouterModel without a network: the router's answer, a four-tool call answered from the tool's wording,
other tools left to the agent, and the hedge (slow first arm, failing first arm, fast first arm).

    .venv-agent/bin/python agent/check_router_model.py
"""
import asyncio
import json
import os
import sys
from pathlib import Path

os.environ["CYCLOPSDIARY_NO_AGENT"] = "1"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from strands.models.model import Model  # noqa: E402

from src import cyclops  # noqa: E402


class Fake(Model):
    """Streams a text reply, or one tool call, after a delay; or fails."""

    def __init__(self, delay=0.0, text=None, tool=None, fail=False, name="primary"):
        self.delay, self.text, self.tool, self.fail, self.name = delay, text, tool, fail, name
        self.config, self.calls = {}, 0

    def update_config(self, **kw): pass
    def get_config(self): return self.config
    def structured_output(self, *a, **kw): raise NotImplementedError

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kw):
        self.calls += 1
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError(f"{self.name} failed")
        yield {"messageStart": {"role": "assistant"}}
        if self.tool:
            name, args = self.tool
            yield {"contentBlockStart": {"start": {"toolUse": {"name": name, "toolUseId": "t1"}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(args)}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": self.text or self.name}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        yield {"metadata": {"usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}, "metrics": {"latencyMs": 1}}}


class FakeMCP:
    def __init__(self, answer): self.answer, self.got = answer, []

    def call_tool_sync(self, tool_use_id, name, args):
        self.got.append((name, args))
        return {"status": "success", "structuredContent": {"answer": self.answer}}


class FakeTool:
    def __init__(self, name, answer):
        self.mcp_tool = type("T", (), {"name": name})()
        self.mcp_client = FakeMCP(answer)


ASK = [{"role": "user", "content": [{"text": "which camera saw my keys?"}]}]
SPECS = [{"name": "object_belief"}]


def run(model, messages=ASK):
    async def go():
        return [ev async for ev in model.stream(messages, SPECS, "system")]
    return asyncio.run(go())


def text(events):
    return "".join(ev["contentBlockDelta"]["delta"].get("text", "") for ev in events if "contentBlockDelta" in ev)


def main():
    # the router answers: no model call
    inner = Fake(text="model")
    m = cyclops.RouterModel(inner, lambda q: "Your keys: near the chair.")
    ev = run(m)
    assert text(ev) == "Your keys: near the chair." and inner.calls == 0

    # the model picks one of the four: its worded answer is the reply, no second call
    tool = FakeTool("object_belief", "Your keys: last seen near the chair.")
    m = cyclops.RouterModel(Fake(tool=("object_belief", {"name": "keys"})), lambda q: None, {"object_belief": tool})
    ev = run(m)
    assert text(ev) == "Your keys: last seen near the chair." and tool.mcp_client.got == [("object_belief", {"name": "keys"})]
    assert [e["messageStop"]["stopReason"] for e in ev if "messageStop" in e] == ["end_turn"]

    # any other tool: left to the agent (the tool call comes through)
    m = cyclops.RouterModel(Fake(tool=("add_memory", {"content": "x"})), lambda q: None, {"object_belief": tool})
    ev = run(m)
    assert any("toolUse" in e.get("contentBlockStart", {}).get("start", {}) for e in ev)
    assert [e["messageStop"]["stopReason"] for e in ev if "messageStop" in e] == ["tool_use"]

    # the hedge: a slow first arm loses to the second, which is sent after hedge_after_s
    m = cyclops.RouterModel(Fake(delay=1.0, name="slow"), lambda q: None, hedge_after_s=0.1)
    m.hedge = Fake(delay=0.05, name="hedge")
    ev = run(m)
    assert text(ev) == "hedge" and m.hedged == dict(calls=1, sent=1, won=1), m.hedged

    # a first arm that fails at once: the hedge takes over without waiting
    m = cyclops.RouterModel(Fake(fail=True, name="down"), lambda q: None, hedge_after_s=5.0)
    m.hedge = Fake(delay=0.0, name="hedge")
    ev = run(m)
    assert text(ev) == "hedge" and m.hedged["sent"] == 1

    # both fail: the first arm's error
    m = cyclops.RouterModel(Fake(fail=True, name="down"), lambda q: None, hedge_after_s=5.0)
    m.hedge = Fake(fail=True, name="also down")
    try:
        run(m)
        raise AssertionError("no error")
    except RuntimeError as e:
        assert str(e) == "down failed"

    # a fast first arm: no hedge sent
    m = cyclops.RouterModel(Fake(delay=0.0, name="fast"), lambda q: None, hedge_after_s=0.5)
    m.hedge = Fake(name="hedge")
    ev = run(m)
    assert text(ev) == "fast" and m.hedged == dict(calls=1, sent=0, won=0) and m.hedge.calls == 0

    # the hedge goes to the next provider in OpenRouter's order
    inner = Fake()
    inner.config = {"model_id": "x", "params": {"extra_body": {"provider": {"order": ["together", "deepinfra", "baseten"]}}}}
    h = cyclops._next_provider(inner)
    assert h.config["params"]["extra_body"]["provider"]["order"] == ["deepinfra", "baseten", "together"]
    assert inner.config["params"]["extra_body"]["provider"]["order"][0] == "together"
    print("RouterModel checks pass (router answer, four-tool answer, other tools, hedge x4, provider order)")


if __name__ == "__main__":
    main()
