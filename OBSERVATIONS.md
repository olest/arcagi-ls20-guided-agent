# Observations: reverse-engineering `ls20` (LockSmith)

These are the findings that made level 1 solvable, gathered by analysing game
recordings frame-by-frame. The stock `guidedllm` agent fails on `ls20` because
its assumptions about the grid are wrong; almost everything below corrects one
of those assumptions.

## 1. Tile values (verified from recordings)

The environment returns a 64×64 grid of integers. Verified meanings for `ls20`:

| Value | Meaning |
| --- | --- |
| `3` | walkable floor |
| `4` | void / background padding — **impassable** (looks empty but blocks you) |
| `5` | wall — impassable |
| `9` | player lower body **and** exit-door interior **and** key glyph displays |
| `12` | player upper body (unique to the player — used as the player marker) |
| `0`, `1` | key rotator (a 5-cell plus sign) |
| `11` | energy bar (status row near the bottom) |
| `8` | key-shape display tiles in the status area |

Common mistakes in the stock/earlier prompts: treating `4` as walls and `5` as
door interior, calling the rotator "enemies", and mistaking the energy bar for
the exit door.

## 2. Player geometry & movement lattice

- The player is a **5×5 block**: top 2 rows are value `12`, bottom 3 rows are
  value `9`. Value `12` is unique, so it's a reliable player locator.
- The grid is a 5×-upscaled render of a logical tile map: **one action moves the
  player exactly 5 cells** in a cardinal direction.
- Because step size == footprint size, movement is a clean lattice — which means
  navigation can be solved exactly with BFS over 5×5 footprint positions.

## 3. Keys and the match condition

- Each level needs a specific **key shape**. Two glyphs matter:
  - **Current key** — shown in a HUD box, rendered at **2×** scale.
  - **Target key** — shown inside the exit-door box, rendered at **1×** scale.
- Both are drawn with value `9`. Detecting them structurally (rather than by
  hard-coded coordinates): take value-9 connected components that are **not** the
  player body, normalise each to a **3×3 binary glyph** via its bounding box
  (this is scale-independent), and compare.
- Classification: the component with the larger bounding box is the current key
  (2×); the smaller is the target (1×).
- Stepping onto the rotator rotates the current key 90°, cycling through 4 states.
  The level's match is reached when a rotation makes current == target.

On the analysed recording, the match is reached after **one rotation** (frame 6).
The stock agent had no match signal, so it rotated past the match and looped.

## 4. The exit mechanic (the key empirical finding)

This was the one unknown that could only be resolved by a live run, and it's the
most surprising part:

- The exit door sits in a walled-off box. The player's floor-only reachable set
  stops at the floor cell directly outside the door — one 5-cell step short.
- **The door threshold is rendered as wall (value 5), but it becomes passable
  the moment the player steps into it while carrying the matching key.** It does
  *not* pre-open at a distance, and (on this level) no wall→floor animation
  precedes it — the "wall" cells simply accept the player when contacted.
- So the correct behaviour is: navigate to the floor cell adjacent to the door,
  then **push one more step toward the door centroid**. That contact step opens
  and enters it, completing the level.

### Evidence (frame diff around completion, run 1)

- Frame 60: player at row 20 (nearest floor below the door), key matched.
- Frame 61: player at row 15 — cells that were value `5` (wall) the previous
  frame now show the player. i.e. the threshold accepted the player on contact.
- One more UP step entered the door and the level counter incremented.

## 5. Why the first run wasted ~57 actions

Run 1 matched the key at action 6, then wandered near the door until action 64,
because the navigation hint stopped at the floor edge and never told the agent
to push *into* the (wall-looking) door. The fix: the exit path appends a
deliberate contact move toward the door centroid, and the anti-wall-ramming
guardrail is made phase-aware so it does not veto that contact move. Run 2 then
cleared level 1 in **13 actions**.

## 6. What this implies for later levels

- Later levels use different layouts (different rotator/door positions, possibly
  different key glyphs and more obstacles). The detection is coordinate-free, so
  key-match and BFS navigation should transfer, but this hasn't been verified.
- Open questions for future work: does every level use the "push into the wall
  with matching key" door mechanic, or do some show an explicit wall→floor
  opening (the code already watches for that and announces `EXIT OPENED`)? Does
  the rotator always cycle in the same order? How does energy scale per level?
