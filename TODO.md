# TODO / Roadmap

Status: **level 1 of `ls20` is solved reliably** (~13 actions). Everything below
is future work. Contributions welcome.

## Generalise beyond level 1

- [ ] Run past level 1 and observe where the agent breaks on levels 2–7.
- [ ] Confirm the coordinate-free key detection (`_analyze_keys`) still finds the
      current/target glyphs on other layouts (different door/rotator positions).
- [ ] Confirm the "larger bbox = current key (2×), smaller = target (1×)"
      heuristic holds when the two key glyphs differ a lot in on-cell count.
- [ ] Check whether later levels have >2 value-9 components (extra HUD/décor),
      which would break the `len(comps) < 2` / top-2 assumption.

## Exit / door mechanic

- [ ] Verify the "door threshold looks like a wall but opens on contact with the
      matching key" behaviour on every level.
- [ ] Determine whether any level shows an explicit wall→floor opening (the
      `_detect_door_opening` watcher already announces `EXIT OPENED`) vs. the
      contact mechanic — and branch navigation accordingly.
- [ ] Handle doors that open on a side other than "toward the target centroid".

## Rotator / key cycling

- [ ] Confirm the rotator always cycles in the same order and that one step onto
      it = one 90° rotation across all levels.
- [ ] Handle the case where the matching state is never reached in the expected
      number of rotations (misdetection guard).

## Energy management

- [ ] Measure how energy scales per level and whether the current "efficient
      path" is enough to finish within budget on the harder levels.
- [ ] Add an explicit low-energy fallback (e.g. prefer the shortest path even if
      the LLM disagrees) below a threshold.

## Robustness & correctness

- [ ] Add unit tests for `_decode_glyph`, `_analyze_keys`, `_bfs_path_to_rotator`,
      `_exit_path`, and `_detect_door_opening` using saved recording frames.
- [ ] Add a small fixture set of recording frames (level start, matched frame,
      door-entry frame) so tests don't need the live API.
- [ ] Guard against `_find_player` returning None mid-level (e.g. transient
      frames) without crashing the prompt builders.
- [ ] Make the "7 levels" count come from the environment metadata instead of a
      hard-coded constant.

## Model / prompt experiments

- [ ] Compare `gpt-4o` vs an o-series model (`o3`, set `REASONING_EFFORT`) on
      actions-to-solve and cost.
- [ ] Try trimming the observation prompt further to reduce token usage.
- [ ] A/B test how much the deterministic `RECOMMENDED MOVE` vs. the raw local
      map contributes to success (ablation).

## Ideas to improve performance on later levels

These are concrete strategies to raise the level-completion rate as the puzzles
get harder (more obstacles, tighter energy, trickier keys):

### Smarter, whole-episode planning
- [ ] Plan the **full** route up front: player → rotator (× N rotations needed)
      → exit, and hand the LLM the whole plan rather than one hop at a time.
- [ ] Pre-compute, for each of the 4 rotation states, how many rotator touches
      are needed to reach the match, and pick the cheapest (fewest actions).
- [ ] Cache the level layout (walls/rotator/door are static within a level) so
      BFS isn't recomputed from scratch every turn.

### Better key handling
- [ ] Detect the key's **colour** as well as its shape if later levels require a
      colour match (current detection is shape-only via value-9 glyphs).
- [ ] Track the observed rotation sequence to predict the next state, so the
      agent can stop exactly on the matching state without overshooting.
- [ ] If multiple keys/rotators exist on a level, associate each rotator with the
      door it unlocks.

### Navigation & exploration
- [ ] Add A* with a Manhattan heuristic (faster than BFS on larger open levels).
- [ ] When the door target isn't yet reachable, do **frontier exploration** to
      reveal the route instead of oscillating near the nearest wall.
- [ ] Detect and avoid dead-ends / one-way traps before committing energy to them.
- [ ] Treat newly-opened cells (`EXIT OPENED`) as first-class navigation targets
      and immediately re-plan toward them.

### Energy-aware decisions
- [ ] Estimate the energy cost of the full plan and abort/retry (RESET) early if
      it can't finish, rather than dying mid-level.
- [ ] Prefer routes that pick up any energy pickups if later levels add them.

### Guardrail & autonomy
- [ ] Make the deterministic layer able to **fully drive** a level when confidence
      is high (LLM as a fallback), reducing tokens, cost, and LLM mistakes.
- [ ] Add a loop-detector: if the agent revisits the same cell/state K times,
      force a different action or trigger exploration.
- [ ] Escalate to a stronger model only on levels where `gpt-4o` stalls.

### Learning across runs
- [ ] Persist solved per-level plans and replay them on repeat encounters.
- [ ] Log every failure with its frame so mechanics for hard levels can be
      reverse-engineered offline (the way level 1 was).
