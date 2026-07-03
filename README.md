# ls20 Guided LLM Agent (ARC-AGI-3)

A guided LLM agent that solves **level 1** of the `ls20` (LockSmith) environment
from the [ARC-AGI-3](https://arcprize.org/) benchmark.

It pairs an LLM (which decides every action) with a small deterministic
grid-analysis layer that:

- decodes the raw integer grid into game objects (player, walls, void, rotator, keys, exit);
- computes a **key-match signal** (does the current key equal the one the exit door wants?);
- computes **collision-free navigation hints** (shortest path to the rotator, then to the exit) via BFS;
- acts as a **guardrail** so the LLM can never loop forever against a wall.

The result: level 1 goes from *never solved* by the stock guided agent to solved
in **~13 actions**.

> Scope note: this is tuned for level 1's mechanics. Later levels use different
> layouts and are not solved yet — see [OBSERVATIONS.md](OBSERVATIONS.md).

## Results

| Metric | Stock `guidedllm` | This agent (run 1) | This agent (run 2) |
| --- | --- | --- | --- |
| Levels completed | 0 | 1 | 1 |
| Actions to clear level 1 | never | 64 | **13** |

Run 1 (64 actions) revealed the exit mechanic; run 2 (13 actions) is after the
exit-navigation fix. See [OBSERVATIONS.md](OBSERVATIONS.md) for the full story.

## How it works (one screen)

1. **Decode the grid.** ARC-AGI-3 gives a 64×64 grid of integers. The tile
   meanings were reverse-engineered from game recordings (floor=3, void=4,
   wall=5, player top=12, rotator=0/1, energy=11, keys/door=9). The player is a
   5×5 block and one action moves it exactly 5 cells.
2. **Detect the key match.** The current key (HUD, rendered 2×) and the target
   key (inside the door box, rendered 1×) are both value-9 glyphs. We find them
   as connected components and normalise each to a **scale-independent 3×3
   binary glyph**, then compare.
3. **Plan the path.** BFS over the 5-cell movement lattice gives the shortest
   collision-free route — to the rotator while the key doesn't match, then to
   the exit door once it does.
4. **The twist.** The door threshold *looks* like a wall but becomes passable
   the instant you step into it carrying the matching key. Plain BFS stops one
   cell short, so the agent appends a deliberate "push into the door" move.
5. **LLM decides, guardrail rescues.** The LLM picks each action from the
   prompt (which includes the `KEY MATCH` line and `RECOMMENDED MOVE`). If it
   rams a wall, the deterministic layer substitutes the correct next move toward
   the current goal.

## Install & run (with `uv`)

This agent is a drop-in for the official
[`arcprize/ARC-AGI-3-Agents`](https://github.com/arcprize/ARC-AGI-3-Agents)
framework (it imports `arcengine`, `agents.agent.Agent`, and
`agents.templates.llm_agents.GuidedLLM` from there). Dependencies are managed by
[`uv`](https://docs.astral.sh/uv/); you do not install packages by hand — `uv`
reads the framework's `pyproject.toml`/`uv.lock` and builds the environment for
you.

### Prerequisites

- **Python 3.12+** (managed automatically by `uv` if missing).
- **`uv`** — install it once:

  ```bash
  # macOS / Linux
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # or: brew install uv        (macOS)
  # or: pipx install uv        (if you prefer pipx)
  ```

  Verify with `uv --version`.
- An **ARC API key** (from <https://arcprize.org/>) and an **OpenAI API key**
  (from <https://platform.openai.com/>).

### 1. Clone this repo and the framework

```bash
git clone <your-repo-url> arcagi-ls20-guided-agent
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
```

### 2. Install the framework's dependencies with `uv`

```bash
cd ARC-AGI-3-Agents
uv sync            # creates .venv/ and installs everything from uv.lock
```

`uv sync` is the only dependency step — it provisions Python, `arcengine`,
`openai`, and everything else the agent needs. (There is no separate install for
this repo; the agent is a single file that runs inside the framework's `uv`
environment.)

### 3. Add your API keys

```bash
cp .env.example .env
# then edit .env and set:
#   ARC_API_KEY=...
#   OPENAI_API_KEY=...
```

`.env` is git-ignored — never commit it.

### 4. Drop in the agent and register it

```bash
# from the ARC-AGI-3-Agents directory, pointing back at this repo:
cp ../arcagi-ls20-guided-agent/ls20_guided.py agents/templates/ls20_guided.py
```

Then add the import + export in `agents/__init__.py`:

```python
from .templates.ls20_guided import Ls20GuidedLLM
# ...and add "Ls20GuidedLLM" to the __all__ list
```

### 5. Run it with `uv`

```bash
# still inside ARC-AGI-3-Agents/
uv run main.py --agent=ls20guidedllm --game=ls20
```

`uv run` executes inside the synced environment (no manual `activate` needed).
When the run finishes it prints a scorecard URL you can open to watch the replay.

To confirm the agent is registered, list the available agents:

```bash
uv run main.py --help    # 'ls20guidedllm' should appear in the -a choices
```

## Configuration

Edit the class constants at the top of `ls20_guided.py`:

- `MODEL` — defaults to `gpt-4o`. For an o-series model set e.g.
  `MODEL = "o3"` and `REASONING_EFFORT = "high"`.
- `MAX_ACTIONS` — action budget per run (default 80).
- `MESSAGE_LIMIT` — LLM memory window (default 20; kept modest to avoid
  OpenAI tokens-per-minute rate limits).

## Secrets

This repo contains **no API keys**. The framework reads `ARC_API_KEY` and
`OPENAI_API_KEY` from its own `.env` file, which you create locally and which is
git-ignored. Never commit `.env`.

## Credits & acknowledgements

- **ARC Prize / [`arcprize/ARC-AGI-3-Agents`](https://github.com/arcprize/ARC-AGI-3-Agents)** —
  the ARC-AGI-3 benchmark and the agent framework this builds on. `Ls20GuidedLLM`
  subclasses their `GuidedLLM` template and reuses their game loop, `arcengine`,
  and tool-calling scaffolding.
- **[Ian Ozsvald](https://ianozsvald.com/)** — for the
  [playgroup_202607_arcagi](https://github.com/ianozsvald/playgroup_202607_arcagi)
  setup, the environment notes, and the original `GuidedLLM`-for-ls20 idea and
  prompt sketch that this work started from and corrected.

## License

The original code and docs in this repository (`ls20_guided.py`, `README.md`,
`OBSERVATIONS.md`) are released under the [MIT License](LICENSE).

The `ARC-AGI-3-Agents` framework and its templates are © ARC Prize under their
own license — see the [upstream repository](https://github.com/arcprize/ARC-AGI-3-Agents).
This repository does not vendor that code; you clone it separately.
