"""
Ls20GuidedLLM — a corrected guided agent for the ls20 LockSmith environment.

Fixes applied vs the upstream GuidedLLM:
  - Correct tile integers for ls20 verified from recording analysis
  - Player is a 5×5 block: top 2 rows = value 12, bottom 3 rows = value 9
  - Key rotator is a plus-sign of values 0 and 1 (NOT the player)
  - Energy bar is value 11 (rows 61-62 of grid), NOT the exit door
  - Level count corrected to 7 (win_levels=7), not 6
  - Frame-diff feedback: tells the LLM when a move was BLOCKED vs MOVED
  - Compact object-position summary replaces raw grid (reduces token load)
  - Deterministic key-match detection + BFS navigation hints
  - MESSAGE_LIMIT=20 (avoids TPM rate limits with gpt-4o)

TILE LEGEND (verified from recording analysis):
  3  = walkable floor
  4  = void / background padding (outside level boundary — not walkable)
  5  = wall (impassable)
  12 = top rows of player body (unique to player: 5 cells wide × 2 rows tall)
  9  = bottom rows of player body + exit door interior + key displays (see below)
  0,1 = key rotator (5-cell plus sign — touch to cycle key shape)
  11 = energy bar display tiles (rows 61-62 of grid, ~42 wide when full)
  8  = key shape display tiles in the status bar area

HOW TO INSTALL & RUN (drop-in for arcprize/ARC-AGI-3-Agents):
  1. Copy this file into the framework:
       cp ls20_guided.py <ARC-AGI-3-Agents>/agents/templates/ls20_guided.py
  2. Register it in <ARC-AGI-3-Agents>/agents/__init__.py:
       from .templates.ls20_guided import Ls20GuidedLLM
       # and add "Ls20GuidedLLM" to __all__
  3. Run:
       cd ARC-AGI-3-Agents
       uv run main.py --agent=ls20guidedllm --game=ls20

  Double-inheritance (GuidedLLM, Agent) means the class auto-registers as
  'ls20guidedllm' via Agent.__subclasses__().

See README.md and OBSERVATIONS.md for the full write-up.
"""

import logging
import textwrap
from collections import deque
from typing import Any, Optional

from arcengine import FrameData, GameAction
from agents.agent import Agent
from agents.templates.llm_agents import GuidedLLM

logger = logging.getLogger()


class Ls20GuidedLLM(GuidedLLM, Agent):
    """GuidedLLM with corrected ls20 tile semantics, frame-diff feedback, and compact grid summary."""

    # --- Model settings (dev defaults — cheaper for rapid iteration) ---
    # REASONING_EFFORT must be None for non-o-series models; gpt-4o rejects the param.
    # To use o3: set MODEL = "o3" and REASONING_EFFORT = "high"
    MODEL = "gpt-4o"
    REASONING_EFFORT = None

    # --- Action budget ---
    MAX_ACTIONS = 80

    # --- Memory window ---
    # Upstream default is 10 messages (~5 turns). Now that we've removed the
    # full grid dump from each message, 20 is affordable and gives ~10 turns
    # of memory — enough to track navigation without blowing the TPM limit.
    MESSAGE_LIMIT = 20

    # --- Tile integer constants (verified from recording frame analysis) ---
    TILE_FLOOR = 3
    TILE_VOID = 4
    TILE_WALL = 5
    # Player is a 5×5 block: top 2 rows = value 12, bottom 3 rows = value 9.
    # Value 12 is UNIQUE to the player (only ~10 cells in the whole grid), so
    # we use it as the player marker. Value 9 also appears in exit door structures.
    TILE_PLAYER = (12,)
    # Key rotator is a plus-sign (5 cells) made of values 0 and 1.
    # The upstream GuidedLLM incorrectly described these as enemy values.
    TILE_ROTATOR = (0, 1)
    # Energy bar: rows 61-62 of the grid filled with value 11 (up to ~84 cells when full).
    # When energy depletes, cells turn to 3 from left to right.
    TILE_ENERGY_BAR = 11

    # --- Movement geometry (verified from recording frame analysis) ---
    # The 64×64 grid is a 5×-upscaled render of a logical tile map:
    #   * the player is a 5×5 block (top 2 rows = value 12, bottom 3 rows = 9)
    #   * ONE action moves the player exactly 5 grid cells in a cardinal direction
    #   * step size == footprint size, so moves never overlap the old footprint
    STEP = 5
    SIZE = 5
    # Direction → (row delta, col delta). ACTION1=UP … ACTION4=RIGHT.
    _MOVES = {
        "ACTION1": (-STEP, 0),  # UP
        "ACTION2": (STEP, 0),   # DOWN
        "ACTION3": (0, -STEP),  # LEFT
        "ACTION4": (0, STEP),   # RIGHT
    }

    # ------------------------------------------------------------------ #
    #  Message window: keep tool_call / tool response pairs consistent    #
    # ------------------------------------------------------------------ #
    #
    # OpenAI's chat API enforces two invariants on the message history:
    #   1. Every 'tool' message must directly follow an assistant message
    #      whose tool_calls include that tool_call_id.
    #   2. Every assistant message with tool_calls must be followed by 'tool'
    #      messages answering each tool_call_id (except the very last message,
    #      whose response is produced on the next turn).
    #
    # The upstream sliding-window trim breaks both invariants, and upstream
    # also sometimes stores an assistant message containing MULTIPLE tool_calls
    # (when the model emits >1 action) while only ever answering the first —
    # which triggers "tool_call_ids did not have response messages".
    #
    # We fix this by (a) collapsing every assistant message to a single
    # tool_call before storing it, and (b) re-validating the whole list after
    # each push so it is always safe to send.

    @staticmethod
    def _role(msg: Any) -> Optional[str]:
        return msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)

    @staticmethod
    def _tool_calls(msg: Any) -> Any:
        return (
            msg.get("tool_calls") if isinstance(msg, dict)
            else getattr(msg, "tool_calls", None)
        )

    @classmethod
    def _tool_call_ids(cls, msg: Any) -> list[str]:
        tcs = cls._tool_calls(msg)
        if not tcs:
            return []
        ids = []
        for tc in tcs:
            tid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if tid is not None:
                ids.append(tid)
        return ids

    @staticmethod
    def _tool_call_id(msg: Any) -> Optional[str]:
        return (
            msg.get("tool_call_id") if isinstance(msg, dict)
            else getattr(msg, "tool_call_id", None)
        )

    @classmethod
    def _collapse_to_single_tool_call(cls, msg: Any) -> None:
        """If an assistant message carries >1 tool_call, keep only the first.

        The game only ever executes one action per turn, so extra tool_calls
        can never receive a response and would break the OpenAI history.
        """
        tcs = cls._tool_calls(msg)
        if tcs and len(tcs) > 1:
            if isinstance(msg, dict):
                msg["tool_calls"] = tcs[:1]
            else:
                try:
                    msg.tool_calls = tcs[:1]
                except Exception:
                    pass

    def push_message(self, message: Any) -> list[dict[str, Any]]:
        self._collapse_to_single_tool_call(message)
        super().push_message(message)  # append + window trim + leading-tool cleanup

        if self.MODEL_REQUIRES_TOOLS:
            self._sanitize_messages()

        return self.messages

    def _sanitize_messages(self) -> None:
        """Enforce both tool_call/tool-response invariants over the whole list."""
        msgs = self.messages

        # Pass 1: drop 'tool' messages that don't answer the immediately
        # preceding assistant tool_call (orphaned responses).
        i = 0
        while i < len(msgs):
            if self._role(msgs[i]) == "tool":
                valid = False
                if i > 0:
                    prev_ids = self._tool_call_ids(msgs[i - 1])
                    if prev_ids and self._tool_call_id(msgs[i]) in prev_ids:
                        valid = True
                if not valid:
                    msgs.pop(i)
                    continue
            i += 1

        # Pass 2: drop assistant tool_call messages whose response is missing.
        # The final message is exempt — its response comes on the next turn.
        i = 0
        while i < len(msgs):
            ids = self._tool_call_ids(msgs[i])
            if ids and i != len(msgs) - 1:
                nxt = msgs[i + 1]
                answered = (
                    self._role(nxt) == "tool"
                    and self._tool_call_id(nxt) in ids
                )
                if not answered:
                    msgs.pop(i)
                    continue
            i += 1

    # ------------------------------------------------------------------ #
    #  Compact grid summary                                                #
    # ------------------------------------------------------------------ #

    # Symbol table for the local map rendering
    _MAP_SYM: dict[int, str] = {
        3: ".",   # floor — walkable
        4: " ",   # void — IMPASSABLE (looks like empty space but you CANNOT enter it)
        5: "#",   # wall — impassable
        0: "R",   # rotator
        1: "R",   # rotator
        9: "?",   # player body or door interior
        12: "@",  # player top (your position)
        11: "-",  # energy bar
        8: "k",   # key display
    }

    # ------------------------------------------------------------------ #
    #  Pathfinding (BFS over 5×5 footprint positions, 5-cell steps)        #
    # ------------------------------------------------------------------ #
    #
    # Reactive "try a direction and see" navigation is myopic: the agent
    # repeatedly walks into the void/wall obstacle in the middle of the arena
    # and never routes around it, burning all MAX_ACTIONS without progress.
    #
    # Because movement is fully deterministic (one action = a 5-cell shift of
    # the 5×5 body onto floor tiles), we can compute an exact shortest path
    # with breadth-first search and hand the agent the correct next move.

    def _find_player(self, grid: list[list[int]]) -> Optional[tuple[int, int]]:
        """Return the (top row, left col) of the player's 5×5 footprint."""
        rows = [r for r, row in enumerate(grid) for c, v in enumerate(row) if v in self.TILE_PLAYER]
        cols = [c for row in grid for c, v in enumerate(row) if v in self.TILE_PLAYER]
        if not rows or not cols:
            return None
        return min(rows), min(cols)

    def _footprint_ok(
        self, grid: list[list[int]], r: int, c: int, walk: frozenset[int]
    ) -> bool:
        """True if the whole 5×5 footprint at (r, c) lies on walkable cells."""
        n = len(grid)
        if r < 0 or c < 0 or r + self.SIZE > n or c + self.SIZE > n:
            return False
        return all(
            grid[i][j] in walk
            for i in range(r, r + self.SIZE)
            for j in range(c, c + self.SIZE)
        )

    def _rotator_centroid(self, grid: list[list[int]]) -> Optional[tuple[int, int]]:
        cells = [
            (r, c)
            for r, row in enumerate(grid)
            for c, v in enumerate(row)
            if v in self.TILE_ROTATOR
        ]
        if not cells:
            return None
        return (
            sum(r for r, _ in cells) // len(cells),
            sum(c for _, c in cells) // len(cells),
        )

    def _bfs_path_to_rotator(self, grid: list[list[int]]) -> Optional[list[str]]:
        """
        Shortest sequence of action names moving the player so its footprint
        covers the rotator centroid. Intermediate steps must be pure floor;
        the destination may overlap the rotator tiles (you step onto it).
        Returns None if there is no player, no rotator, or no path.
        """
        player = self._find_player(grid)
        target = self._rotator_centroid(grid)
        if player is None or target is None:
            return None

        floor = frozenset({self.TILE_FLOOR})
        step_walk = frozenset({self.TILE_FLOOR, *self.TILE_ROTATOR})

        def covers(r: int, c: int) -> bool:
            return r <= target[0] < r + self.SIZE and c <= target[1] < c + self.SIZE

        start = player
        # Already on the rotator?
        if covers(*start):
            return []

        parent: dict[tuple[int, int], Optional[tuple[int, int, str]]] = {start: None}
        q: deque[tuple[int, int]] = deque([start])
        goal: Optional[tuple[int, int]] = None
        while q:
            r, c = q.popleft()
            for name, (dr, dc) in self._MOVES.items():
                nr, nc = r + dr, c + dc
                if (nr, nc) in parent:
                    continue
                # A step is valid if it lands on floor, OR it covers the rotator
                # (the final "step onto the rotator" move).
                lands_on_rotator = covers(nr, nc) and self._footprint_ok(grid, nr, nc, step_walk)
                if self._footprint_ok(grid, nr, nc, floor) or lands_on_rotator:
                    parent[(nr, nc)] = (r, c, name)
                    if covers(nr, nc):
                        goal = (nr, nc)
                        q.clear()
                        break
                    q.append((nr, nc))
        if goal is None:
            return None

        moves: list[str] = []
        cur: Optional[tuple[int, int]] = goal
        while parent[cur] is not None:
            pr, pc, name = parent[cur]  # type: ignore[misc]
            moves.append(name)
            cur = (pr, pc)
        moves.reverse()
        return moves

    def _move_is_blocked(self, grid: list[list[int]], action_name: str) -> bool:
        """True if the given movement action would not change the grid (wall/void)."""
        if action_name not in self._MOVES:
            return False
        player = self._find_player(grid)
        if player is None:
            return False
        dr, dc = self._MOVES[action_name]
        walk = frozenset({self.TILE_FLOOR, *self.TILE_ROTATOR})
        return not self._footprint_ok(grid, player[0] + dr, player[1] + dc, walk)

    # ------------------------------------------------------------------ #
    #  Generic BFS: reachable set + path to the cell nearest a target     #
    # ------------------------------------------------------------------ #

    def _bfs_reachable(
        self, grid: list[list[int]]
    ) -> tuple[dict[tuple[int, int], Any], Optional[tuple[int, int]]]:
        """BFS all floor-only footprint positions reachable from the player.

        Returns (parent_map, start). parent_map maps position -> (pr, pc, move)
        for reconstruction; the start maps to None.
        """
        player = self._find_player(grid)
        if player is None:
            return {}, None
        floor = frozenset({self.TILE_FLOOR})
        parent: dict[tuple[int, int], Any] = {player: None}
        q: deque[tuple[int, int]] = deque([player])
        while q:
            r, c = q.popleft()
            for name, (dr, dc) in self._MOVES.items():
                nr, nc = r + dr, c + dc
                if (nr, nc) not in parent and self._footprint_ok(grid, nr, nc, floor):
                    parent[(nr, nc)] = (r, c, name)
                    q.append((nr, nc))
        return parent, player

    @staticmethod
    def _reconstruct(parent: dict[tuple[int, int], Any], goal: tuple[int, int]) -> list[str]:
        moves: list[str] = []
        cur: Optional[tuple[int, int]] = goal
        while parent[cur] is not None:
            pr, pc, name = parent[cur]
            moves.append(name)
            cur = (pr, pc)
        moves.reverse()
        return moves

    def _bfs_path_toward(
        self, grid: list[list[int]], target: tuple[int, int]
    ) -> Optional[list[str]]:
        """Shortest path to the reachable floor position whose footprint centre
        is closest to `target`. Returns [] if already there, None if no player."""
        parent, start = self._bfs_reachable(grid)
        if start is None:
            return None
        tr, tc = target
        half = self.SIZE // 2

        def dist(p: tuple[int, int]) -> int:
            return abs(p[0] + half - tr) + abs(p[1] + half - tc)

        best = min(parent.keys(), key=dist)
        if best == start:
            return []
        return self._reconstruct(parent, best)

    # ------------------------------------------------------------------ #
    #  Key-match detection (current key vs target key at the exit door)   #
    # ------------------------------------------------------------------ #
    #
    # The current key is shown as a value-9 glyph in a HUD box, rendered at 2×.
    # The target key is a value-9 glyph inside the exit-door box, rendered at 1×.
    # Both normalise to a 3×3 binary glyph; the level is solved once they match.
    #
    # Detection is structural (no hard-coded coordinates): find the value-9
    # connected components that are NOT the player body, normalise each to 3×3,
    # and classify the one with the larger bounding box as the current key (2×)
    # and the smaller as the target (1×).

    def _player_cells(self, grid: list[list[int]]) -> set[tuple[int, int]]:
        p = self._find_player(grid)
        if p is None:
            return set()
        pr, pc = p
        return {
            (r, c)
            for r in range(pr, pr + self.SIZE)
            for c in range(pc, pc + self.SIZE)
        }

    def _components_9(
        self, grid: list[list[int]], exclude: set[tuple[int, int]]
    ) -> list[list[tuple[int, int]]]:
        """4-connected components of value-9 cells, skipping `exclude` cells."""
        n = len(grid)
        seen: set[tuple[int, int]] = set(exclude)
        comps: list[list[tuple[int, int]]] = []
        for r in range(n):
            for c in range(len(grid[r])):
                if grid[r][c] == 9 and (r, c) not in seen:
                    comp: list[tuple[int, int]] = []
                    dq: deque[tuple[int, int]] = deque([(r, c)])
                    seen.add((r, c))
                    while dq:
                        cr, cc = dq.popleft()
                        comp.append((cr, cc))
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = cr + dr, cc + dc
                            if (
                                0 <= nr < n
                                and 0 <= nc < len(grid[nr])
                                and (nr, nc) not in seen
                                and grid[nr][nc] == 9
                            ):
                                seen.add((nr, nc))
                                dq.append((nr, nc))
                    comps.append(comp)
        return comps

    @staticmethod
    def _decode_glyph(comp: list[tuple[int, int]]) -> tuple[str, str, str]:
        """Normalise a value-9 component to a 3×3 binary glyph via its bbox.

        Works for any render scale (1× or 2×) as long as the glyph touches all
        four edges of its bounding box (true for these key shapes)."""
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        cset = set(comp)
        glyph = []
        for lr in range(3):
            row = ""
            for lc in range(3):
                rr0, rr1 = r0 + lr * h // 3, r0 + (lr + 1) * h // 3
                cc0, cc1 = c0 + lc * w // 3, c0 + (lc + 1) * w // 3
                on = any(
                    (rr, cc) in cset
                    for rr in range(rr0, max(rr1, rr0 + 1))
                    for cc in range(cc0, max(cc1, cc0 + 1))
                )
                row += "X" if on else "."
            glyph.append(row)
        return tuple(glyph)  # type: ignore[return-value]

    def _analyze_keys(self, grid: list[list[int]]) -> Optional[dict[str, Any]]:
        """Return current/target 3×3 glyphs, whether they match, and the target
        (exit-door) centroid. None if the two key glyphs can't be identified."""
        comps = self._components_9(grid, self._player_cells(grid))
        comps = [c for c in comps if len(c) >= 3]  # drop tiny noise
        if len(comps) < 2:
            return None

        def bbox_area(c: list[tuple[int, int]]) -> int:
            rs = [r for r, _ in c]
            cs = [x for _, x in c]
            return (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)

        comps.sort(key=bbox_area, reverse=True)
        current_comp, target_comp = comps[0], comps[1]
        cur = self._decode_glyph(current_comp)
        tgt = self._decode_glyph(target_comp)
        trs = [r for r, _ in target_comp]
        tcs = [c for _, c in target_comp]
        target_centroid = (sum(trs) // len(trs), sum(tcs) // len(tcs))
        return {
            "current": cur,
            "target": tgt,
            "match": cur == tgt,
            "target_centroid": target_centroid,
        }

    @staticmethod
    def _fmt_glyph(glyph: tuple[str, str, str]) -> str:
        return "/".join(glyph)

    def _detect_door_opening(
        self, prev_grid: list[list[int]], curr_grid: list[list[int]]
    ) -> Optional[tuple[int, int]]:
        """If any wall(5) turned into floor(3) between frames, return the
        centroid of the newly-opened cells (the exit door opening)."""
        opened = [
            (r, c)
            for r in range(len(curr_grid))
            for c in range(len(curr_grid[r]))
            if prev_grid[r][c] == self.TILE_WALL and curr_grid[r][c] == self.TILE_FLOOR
        ]
        if not opened:
            return None
        return (
            sum(r for r, _ in opened) // len(opened),
            sum(c for _, c in opened) // len(opened),
        )

    # ------------------------------------------------------------------ #
    #  Phase selection: rotate-until-match, then head to the exit          #
    # ------------------------------------------------------------------ #

    def _cardinal_toward(
        self, r: int, c: int, target: tuple[int, int]
    ) -> Optional[str]:
        """Cardinal action that moves the footprint centre at (r,c) toward target."""
        half = self.SIZE // 2
        ddr, ddc = target[0] - (r + half), target[1] - (c + half)
        if abs(ddr) >= abs(ddc) and ddr != 0:
            return "ACTION1" if ddr < 0 else "ACTION2"
        if ddc != 0:
            return "ACTION3" if ddc < 0 else "ACTION4"
        return None

    def _exit_path(
        self, grid: list[list[int]], target: tuple[int, int]
    ) -> Optional[list[str]]:
        """Path to the exit door: route to the nearest floor cell, then append a
        'push into the door' move toward the target.

        The door threshold is rendered as wall (value 5) but becomes passable the
        moment the player arrives carrying the matching key, so plain floor-only
        BFS stops one step short. We add the contact move so the agent actually
        steps into the opening instead of wandering (this was ~57 wasted actions)."""
        path = self._bfs_path_toward(grid, target)
        if path is None:
            return None
        player = self._find_player(grid)
        if player is None:
            return path
        r, c = player
        for m in path:
            dr, dc = self._MOVES[m]
            r, c = r + dr, c + dc
        enter = self._cardinal_toward(r, c, target)
        if enter is not None:
            path = path + [enter]
        return path

    def _phase_and_path(
        self, grid: list[list[int]]
    ) -> tuple[str, Optional[list[str]], Optional[dict[str, Any]]]:
        """Decide the current phase and the recommended path.

        - 'exit'   : keys match -> path toward the exit-door target.
        - 'rotate' : keys don't match (or unknown) -> path toward the rotator.
        """
        keys = self._analyze_keys(grid)
        if keys and keys["match"]:
            return "exit", self._exit_path(grid, keys["target_centroid"]), keys
        return "rotate", self._bfs_path_to_rotator(grid), keys

    _ACTION_LABEL = {
        "ACTION1": "ACTION1 (UP)",
        "ACTION2": "ACTION2 (DOWN)",
        "ACTION3": "ACTION3 (LEFT)",
        "ACTION4": "ACTION4 (RIGHT)",
    }

    def _extract_notable_cells(self, grid: list[list[int]]) -> str:
        """
        Scan the grid and return:
        1. Object positions (player, rotator, energy remaining)
        2. A 15×15 local ASCII map centred on the player

        The local map is the most actionable piece of information: it shows
        exactly which cells are floor ('.'), void (' '), wall ('#'), rotator
        ('R'), and player body ('@') in the immediate neighbourhood.

        IMPORTANT: void (' ') cells are IMPASSABLE — any move that would put
        any part of the 5×5 player body into a void or wall cell is BLOCKED.
        """
        player_cells: list[tuple[int, int]] = []
        rotator_cells: list[tuple[int, int]] = []
        energy_bar_count: int = 0

        for r, row in enumerate(grid):
            for c, val in enumerate(row):
                if val in self.TILE_PLAYER:
                    player_cells.append((r, c))
                elif val in self.TILE_ROTATOR:
                    rotator_cells.append((r, c))
                elif val == self.TILE_ENERGY_BAR:
                    energy_bar_count += 1

        def centroid(cells: list[tuple[int, int]]) -> tuple[int, int]:
            r_vals = [r for r, _ in cells]
            c_vals = [c for _, c in cells]
            return sum(r_vals) // len(r_vals), sum(c_vals) // len(c_vals)

        lines: list[str] = []

        if player_cells:
            pr, pc = centroid(player_cells)
            lines.append(f"Player centre: row {pr}, col {pc}")
        else:
            lines.append("Player: NOT FOUND in grid")
            pr, pc = 32, 36  # fallback

        if rotator_cells:
            rr, rc = centroid(rotator_cells)
            lines.append(f"Key rotator centre: row {rr}, col {rc}")
            lines.append(f"  → to reach rotator: go {'UP' if rr < pr else 'DOWN'} {abs(rr-pr)} rows, "
                         f"{'LEFT' if rc < pc else 'RIGHT'} {abs(rc-pc)} cols")

        energy_pct = round(energy_bar_count / 84 * 100) if energy_bar_count else 0
        lines.append(f"Energy remaining: {energy_bar_count}/84 ({energy_pct}%)")

        # --- Local 15×15 ASCII map centred on the player ---
        half = 7
        map_lines = ["Local map (15×15, centred on @=player, R=rotator, ' '=void/IMPASSABLE, #=wall, .=floor):"]
        nrows = len(grid)
        ncols = len(grid[0]) if grid else 0
        for dr in range(-half, half + 1):
            row_str = []
            for dc in range(-half, half + 1):
                gr, gc = pr + dr, pc + dc
                if 0 <= gr < nrows and 0 <= gc < ncols:
                    v = grid[gr][gc]
                    sym = self._MAP_SYM.get(v, str(v))
                else:
                    sym = "X"  # out of bounds
                row_str.append(sym)
            map_lines.append("  " + "".join(row_str))
        lines.extend(map_lines)

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Observation prompt (sent after each action, before choosing next)  #
    # ------------------------------------------------------------------ #

    def build_func_resp_prompt(self, latest_frame: FrameData) -> str:
        """
        Override to add:
          1. Frame-diff: MOVED vs BLOCKED, computed from self.frames (instance var).
          2. Compact object-position summary (replaces the full raw grid to avoid
             hitting TPM rate limits — the 64x64 grid is ~4000 tokens per message).

        Timing note: when this method is called, the result of the last action has
        already been appended to self.frames (self.frames[-1]). The frame BEFORE
        that action is self.frames[-2]. Comparing [-1] vs [-2] tells us whether
        the last action changed the grid (MOVED) or not (BLOCKED).
        latest_frame is the same as self.frames[-1] — the current observation.
        """
        # --- Frame-diff ---
        # self.frames[-1] = state after the LAST action (= latest_frame)
        # self.frames[-2] = state BEFORE the last action
        if len(self.frames) < 2:
            effect = "RESET (first observation — no previous frame to compare)"
        else:
            prev_frame = self.frames[-2]
            curr_frame = self.frames[-1]
            if not prev_frame.frame or not curr_frame.frame:
                effect = "RESET (first observation — no previous frame to compare)"
            elif curr_frame.frame == prev_frame.frame:
                effect = (
                    "BLOCKED — the grid is unchanged. "
                    "Your last move hit a wall. Choose a DIFFERENT direction."
                )
            else:
                effect = "MOVED — the grid changed after your last action."

        # --- Compact summary (use last grid layer) ---
        current_grid = latest_frame.frame[-1] if latest_frame.frame else []
        notable = self._extract_notable_cells(current_grid)

        # --- Key-match signal + phase-aware pathfinding recommendation ---
        key_line = "KEY MATCH: UNKNOWN (could not read the key displays)."
        recommend = ""
        opened_line = ""
        if current_grid:
            phase, path, keys = self._phase_and_path(current_grid)

            if keys is not None:
                cur = self._fmt_glyph(keys["current"])
                tgt = self._fmt_glyph(keys["target"])
                if keys["match"]:
                    key_line = (
                        f"KEY MATCH: YES — your key ({cur}) matches the exit door. "
                        f"STOP touching the rotator. Head to the exit and step in."
                    )
                else:
                    key_line = (
                        f"KEY MATCH: NO — current key is {cur}, exit needs {tgt}. "
                        f"Touch the rotator to cycle the key one step."
                    )

            if path is None:
                recommend = "No collision-free route computed right now."
            elif len(path) == 0:
                if phase == "rotate":
                    recommend = (
                        "You are ON the rotator. If KEY MATCH is NO, step OFF (any open "
                        "direction) and back ON to cycle once more. If KEY MATCH is YES, "
                        "stop cycling and head to the exit."
                    )
                else:
                    recommend = (
                        "You are as close to the exit as the current layout allows. "
                        "Try stepping toward the exit door / into any newly opened gap."
                    )
            else:
                nxt = self._ACTION_LABEL.get(path[0], path[0])
                preview = " → ".join(
                    self._ACTION_LABEL.get(m, m).split(" ")[1].strip("()")
                    for m in path[:8]
                )
                if phase == "exit":
                    recommend = (
                        f"RECOMMENDED MOVE: {nxt}\n"
                        f"  (route to the EXIT DOOR is {len(path)} moves: {preview})\n"
                        f"  The final move pushes INTO the door. The door threshold looks"
                        f" like a wall but OPENS the moment you step into it with the"
                        f" matching key — so keep following this route even if a move seems"
                        f" to hit a wall near the door. Just call the RECOMMENDED MOVE."
                    )
                else:
                    recommend = (
                        f"RECOMMENDED MOVE: {nxt}\n"
                        f"  (shortest collision-free route to the rotator is {len(path)} moves: {preview})\n"
                        f"  This route avoids all void/wall tiles. Prefer it unless you have a reason not to."
                    )

            # Door-opening watcher: did a wall just turn into floor?
            if len(self.frames) >= 2 and self.frames[-2].frame and self.frames[-1].frame:
                opening = self._detect_door_opening(
                    self.frames[-2].frame[-1], self.frames[-1].frame[-1]
                )
                if opening is not None:
                    opened_line = (
                        f"EXIT OPENED: new walkable gap appeared near row {opening[0]}, "
                        f"col {opening[1]} — navigate there and step in to finish the level."
                    )

        nav_block = key_line + "\n\n" + recommend
        if opened_line:
            nav_block += "\n\n" + opened_line

        return textwrap.dedent(f"""
# State: {latest_frame.state.name}
# Levels completed: {latest_frame.levels_completed} / 7

# Last action effect: {effect}

# Notable object positions (row, col):
{notable}

# Navigation:
{nav_block}

# TURN:
Reply with a few sentences of plain-text strategy observation to inform your next action.
        """).strip()

    # ------------------------------------------------------------------ #
    #  System / rules prompt (sent every turn as the user message)        #
    # ------------------------------------------------------------------ #

    def build_user_prompt(self, latest_frame: FrameData) -> str:
        return textwrap.dedent("""
# CONTEXT: You are playing ls20 (a LockSmith puzzle game).

## Controls
- ACTION1: move UP
- ACTION2: move DOWN
- ACTION3: move LEFT
- ACTION4: move RIGHT
- RESET: restart the current level
- ACTION5, ACTION6: unused in this game — do not call them

## Goal
Complete 7 levels. Each level: match the key shape, then step into the exit door.

## Tile values (verified from game recordings)
- 3  = walkable floor — you can move here
- 4  = void / background padding — impassable
- 5  = wall — you CANNOT move through this
- 12 = YOUR PLAYER (top rows of your 5×5 body). The "Player centre" above is you.
- 9  = player lower body AND exit door interior (value 9 appears in multiple places)
- 0, 1 = key rotator — a plus-sign shape. Touch it to cycle the key shape.
         Step away and back onto it to cycle again.
- 11 = energy bar tiles (status bar at bottom of grid, NOT the exit door)
- 8  = key shape display (in status bar area)

## Key rules
- You have 7 levels to complete. "Levels completed" above shows your progress.
- Your energy is shown by the energy bar (count of 11-tiles). When it hits 0: GAME_OVER.
  Each action consumes energy. Navigate efficiently!
- The key rotator (values 0/1, plus shape) cycles the key shape when you step on it.
  Step away, then back on to rotate again.
- Once your key matches the exit door, walk into the door to complete the level.
- The exit door is typically found in a walled-off section of the level.
- Walls are value 5. You CANNOT walk through them.

## The KEY MATCH signal (READ THIS EVERY TURN)
The Navigation section reports "KEY MATCH: YES" or "KEY MATCH: NO":
- KEY MATCH: NO  -> your key is wrong. Touch the rotator ONCE (step onto it) to
  cycle the key one step, then re-check. Each step onto the rotator rotates the key.
- KEY MATCH: YES -> your key is correct. STOP touching the rotator IMMEDIATELY.
  Do NOT cycle again (that would break the match). Follow the RECOMMENDED MOVE to
  the EXIT DOOR and step into it.
Never keep cycling the rotator once KEY MATCH is YES — that is the #1 way to fail.

## Critical movement rule
IF the "Last action effect" above says BLOCKED:
  your last move did nothing — the grid did not change.
  Do NOT repeat the same direction. Try a DIFFERENT action immediately.

## Reading the local map
The observation gives you a 15×15 ASCII map centred on your player (@):
  @  = your player (you are in the centre of the map)
  .  = walkable floor
     = void / background — IMPASSABLE, just like a wall!
  #  = wall (impassable)
  R  = key rotator (step on it to cycle key shape)
  ?  = exit door interior or player body

VOID (' ') IS IMPASSABLE. The blank/space tiles in the map are NOT open space —
they are background void tiles that block movement just like walls (#).
You CANNOT walk through void tiles even though they look like empty space.

## Strategy (follow in order)
1. FOLLOW THE "RECOMMENDED MOVE" in the Navigation section. It is a deterministic,
   collision-free shortest path (to the rotator while KEY MATCH is NO, or to the
   EXIT DOOR once KEY MATCH is YES). It already routes around every void and wall.
2. While KEY MATCH: NO — get to the rotator and cycle the key (step off, step on).
   After each rotation, check the KEY MATCH line again.
3. The MOMENT KEY MATCH turns YES — stop cycling and follow the RECOMMENDED MOVE
   toward the EXIT DOOR. If you see an "EXIT OPENED" line, go to that gap and step in.
4. The exit may only open (walls turn to floor) once you arrive carrying the correct
   key. Keep moving toward the exit door area and watch for "EXIT OPENED".
5. Watch your energy — if it drops below 20%, prioritise finishing the level.

NOTE: if your chosen move would walk into a void/wall, the system will
automatically substitute the correct detour move for you — but you should still
prefer the RECOMMENDED MOVE to avoid wasting turns.

# TURN:
Call exactly one action.
        """).strip()

    # ------------------------------------------------------------------ #
    #  Action selection: LLM decides, pathfinder rescues from dead ends   #
    # ------------------------------------------------------------------ #

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Let the LLM choose, then guard against getting stuck.

        The LLM often insists on walking straight into the central void/wall
        obstacle and never routes around it, exhausting MAX_ACTIONS. If the
        LLM's chosen move is BLOCKED (its 5×5 footprint can't enter the target
        tiles) we substitute the deterministic pathfinder's next move toward the
        current phase's goal — the rotator while the key doesn't match, or the
        exit door once it does. This makes it impossible to loop forever against
        a wall, and never drags the agent back to the rotator after a match.
        """
        action = super().choose_action(frames, latest_frame)

        grid = latest_frame.frame[-1] if latest_frame.frame else None
        if grid is None or action.name not in self._MOVES:
            return action

        if not self._move_is_blocked(grid, action.name):
            return action

        phase, path, _ = self._phase_and_path(grid)
        if not path:
            return action  # nothing better to suggest; let it be

        rescue = path[0]
        if rescue == action.name:
            return action  # already optimal (shouldn't happen if blocked)

        goal = "exit" if phase == "exit" else "rotator"
        new_action = GameAction.from_name(rescue)
        new_action.set_data({})
        new_action.reasoning = {
            "override": "pathfinder",
            "phase": phase,
            "llm_choice": action.name,
            "substituted": rescue,
            "reason": f"LLM move was blocked by void/wall; using collision-free detour toward {goal}",
        }
        logger.info(
            f"[pathfinder] LLM chose {action.name} (BLOCKED) — substituting {rescue} (toward {goal})"
        )
        return new_action
