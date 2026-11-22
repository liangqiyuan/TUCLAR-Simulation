"""Hybrid planner: HTN/A* planning with LLM belief reasoning and a shared message codebook.

Implements the architecture proposed in "Collaborative Autonomy under Limited Communication".

  Paper concept                          Here
  -------------------------------------  ----------------------------------------------------------
  Planner decides every action           AStarHTNPlanner (htn_planner.py) picks every FLY/IDLE/RECALL
  Constraints checked before acting      Planner preconditions; LLM output is validated, never executed
  Blanks in the task tree                Which targets teammates are covering; types of unvisited targets
  Task-level messages, shared codebook   Codebook: "claim(3);type(1,Joint)"
  Event-triggered claims and findings    message_for(): current claim plus every known type, sent on each
                                         decision (the sim shows the latest message to anyone in range,
                                         so a drone that comes back into range receives the repeat)
  LLM reasoning about silent teammates   _belief_prompt() / _apply_llm(): LLM returns beliefs only
  A message replaces any belief          observe(): message > observation > unexpired LLM belief > memory
  A belief lapses after a set time       HybridConfig.belief_ttl (LLM beliefs), memory_ttl (stale claims)
  Shared priority rule                   Lowest drone index keeps a contested target (AgentHandler._is_surplus,
                                         fed by fresh_claims())
  Communication vs reasoning trade-off   HybridConfig.use_messages / use_llm ablation switches

Each drone owns one HybridPlanner. Nothing here is shared between drones; everything a planner
knows arrives through the scene dict that AgentHandler builds for that drone.
"""
import math
import re
import time
from collections import Counter
from dataclasses import dataclass

from environment import REQUIRED_DRONES, TargetType
from htn_planner import UNKNOWN_SLOT_GUESS, AStarHTNPlanner

COMMS_FAILURE_TEXT = "Communication Failure."


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class HybridConfig:
    use_llm: bool = True              # False: no belief reasoning (planner + messages only)
    use_messages: bool = True         # False: ignore the codebook (planner + observation + LLM only)
    share_types: bool = True          # include type(T,K) findings in messages and accept them
    belief_ttl: int = 10              # rounds before an LLM belief lapses
    memory_ttl: int = 10              # rounds before a stale last-known claim stops counting as a belief
    stale_fallback: str = "last_known"  # no fresh info and no LLM belief: "last_known" claim or "none"
    show_target_coords: bool = False  # mirror the prompt flags so the LLM sees what other prompts show
    show_own_location: bool = False

    def label(self):
        return (f"llm={int(self.use_llm)},msg={int(self.use_messages)},types={int(self.share_types)},"
                f"ttl={self.belief_ttl},mttl={self.memory_ttl},stale={self.stale_fallback}")


# ---------------------------------------------------------------------------
# Shared codebook (the "common language" of the team)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Decoded:
    has_claim: bool
    claim: int | None                 # target index; None means the sender said it is idle
    types: dict                       # target index -> TargetType


class Codebook:
    _CLAIM = re.compile(r"claim\((\d+)\)")
    _TYPE = re.compile(r"type\((\d+),\s*(Exclusive|Joint|Sequential)\)")

    @staticmethod
    def encode(kind, target, findings):
        tokens = []
        if kind == "FLY":
            tokens.append(f"claim({target})")
        elif kind == "IDLE":
            tokens.append("idle")
        elif kind == "RECALL":
            tokens.append("recall")
        tokens += [f"type({i},{k.value})" for i, k in sorted(findings.items())]
        return ";".join(tokens)

    @classmethod
    def decode(cls, text):
        has_claim, claim, types = False, None, {}
        for token in str(text).split(";"):
            token = token.strip()
            m = cls._CLAIM.fullmatch(token)
            if m:
                has_claim, claim = True, int(m.group(1))
                continue
            if token == "idle":
                has_claim, claim = True, None
                continue
            m = cls._TYPE.fullmatch(token)
            if m:
                types[int(m.group(1))] = TargetType(m.group(2))
        return Decoded(has_claim, claim, types)


# ---------------------------------------------------------------------------
# Beliefs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Belief:
    target: int | None    # target the teammate is believed to be flying to; None = idle
    source: str           # "msg" | "obs" | "llm" | "memory" (stale last-known claim)
    round: int            # round the underlying information was obtained


BELIEF_SYSTEM_PROMPT = """
You are one drone in a team that is completing a mission together. Communication is limited.
A route planner chooses your actions. Your only job is to estimate what each SILENT teammate is doing right now, because you cannot currently see or hear them.

How the team behaves (every teammate runs the same procedure as you):
- A teammate flies to one target at a time and keeps flying to it until it is cleared. It then picks another open target.
- Target types: Exclusive needs 1 drone, Joint needs 2 drones at the same time, Sequential needs 3 different drones. If a type is unknown, assume 1 drone.
- Drones avoid targets that already have enough drones heading to them. When two drones want the same slot, the lower-numbered drone keeps it.
- A teammate whose last known target is still open has most likely kept flying to it.
- A teammate whose last known target is now cleared has most likely moved on to another open target that still needs drones.
- A teammate with nothing known could be heading to any open target that is not already covered.

Reply with a single JSON object and nothing else:
{"reason": "<one short sentence>", "beliefs": {"<teammate index>": <target index, or null if idle>}}
Include only the silent teammates you are asked about.
"""


class HybridPlanner:
    """One per drone: belief store + message codebook + A* planner, with an injected LLM callback."""

    def __init__(self, agent_index, time_budget_seconds, config=None, planner=None):
        self.me = agent_index
        self.cfg = config or HybridConfig()
        self.htn = planner or AStarHTNPlanner(time_budget_seconds)   # any planner with AStarHTNPlanner's plan() interface
        self.active = set()          # teammates currently flying (recalled drones are dropped)
        self.beliefs = {}            # teammate -> Belief currently in force
        self.last_known = {}         # teammate -> latest Belief from a message or observation
        self.lost_msg = set()        # teammates whose newest message was lost this round
        self.learned_types = {}      # target index -> TargetType, learned from teammates' findings
        self.last_used = {}          # beliefs the most recent plan was built on (for logging only)
        self.queries = 0             # LLM belief queries made
        self.last_plan_s = 0.0       # wall-clock seconds spent in the last A* search (excludes the LLM)
        self._observed_round = None

    # -- belief maintenance -------------------------------------------------
    @staticmethod
    def _dead(target, targets):
        """True for a claim on a target that no longer exists or is already cleared."""
        return isinstance(target, int) and (target < 0 or target >= len(targets) or targets[target].cleared)

    def observe(self, scene, targets):
        """Ingest this round's messages and observations. Cheap; no LLM. Call once per round."""
        rnd = scene["round"]
        self._observed_round = rnd
        views = scene["teammate_targets"]
        messages = scene.get("messages") or {} if self.cfg.use_messages else {}
        self.active = set(views)
        self.lost_msg = set()
        new = {}

        for j in self.active:
            belief = None

            text = messages.get(j)
            if text is not None:
                decoded = Codebook.decode(text)
                if self.cfg.share_types:
                    for t, kind in decoded.types.items():
                        if 0 <= t < len(targets):
                            self.learned_types[t] = kind
                if decoded.has_claim and not self._dead(decoded.claim, targets):
                    belief = Belief(decoded.claim, "msg", rnd)
                elif str(text).strip() == COMMS_FAILURE_TEXT:
                    self.lost_msg.add(j)

            if belief is None:
                seen = views[j]
                if seen is None:
                    belief = Belief(None, "obs", rnd)
                elif isinstance(seen, int) and not isinstance(seen, bool) and seen >= 0:
                    belief = Belief(seen, "obs", rnd)

            if belief is not None:
                self.last_known[j] = belief
            else:
                prev = self.beliefs.get(j)
                if (prev is not None and prev.source == "llm" and rnd - prev.round < self.cfg.belief_ttl
                        and not self._dead(prev.target, targets)):
                    belief = prev
                elif self.cfg.stale_fallback == "last_known":
                    lk = self.last_known.get(j)
                    # A last-known claim only counts while it is recent; an old one would hold a target
                    # "covered" forever for a teammate that may have moved on.
                    if (lk is not None and lk.target is not None and not self._dead(lk.target, targets)
                            and rnd - lk.round < self.cfg.memory_ttl):
                        belief = Belief(lk.target, "memory", lk.round)

            if belief is not None:
                new[j] = belief

        self.beliefs = new
        self.last_known = {j: b for j, b in self.last_known.items() if j in self.active}

    def fresh_claims(self):
        """teammate -> target for every belief backed by current information (not stale memory)."""
        return {j: b.target for j, b in self.beliefs.items() if b.source != "memory"}

    def silent(self):
        """Active teammates with no current information: the LLM's job."""
        return sorted(j for j in self.active if j not in self.beliefs or self.beliefs[j].source == "memory")

    def type_learned(self, target_index):
        return target_index in self.learned_types

    # -- LLM reasoning --------------------------------------------------------
    def _belief_prompt(self, targets, scene, silent, rnd):
        known = scene["type_known"]
        me = self.me
        lines = [f"You are agent {me}. Round {rnd}."]
        if self.cfg.show_own_location:
            loc = scene["location"]
            lines.append(f"YOUR LOCATION: X={loc[0]:.1f}m, Y={loc[1]:.1f}m")
        mine = scene.get("my_target")
        lines.append("You are currently heading to target %d." % mine
                     if isinstance(mine, int) and mine >= 0 else "You are currently idle.")

        lines.append("\nTEAMMATES WITH CURRENT INFORMATION:")
        shown = False
        for j in sorted(self.beliefs):
            b = self.beliefs[j]
            if b.source == "memory":
                continue
            where = "idle" if b.target is None else f"target {b.target}"
            how = {"msg": "message", "obs": "observed, may be wrong", "llm": f"your estimate from round {b.round}"}[b.source]
            lines.append(f"  Agent {j}: {where} ({how})")
            shown = True
        if not shown:
            lines.append("  none")

        lines.append("\nSILENT TEAMMATES (you cannot currently see or hear them):")
        for j in silent:
            lk = self.last_known.get(j)
            if lk is None:
                detail = "nothing known about them yet"
            else:
                where = "idle" if lk.target is None else f"target {lk.target}"
                how = "a message" if lk.source == "msg" else "an observation"
                detail = f"last known: {where}, from {how} {rnd - lk.round} rounds ago"
            if j in self.lost_msg:
                detail += "; their newest message was lost"
            lines.append(f"  Agent {j}: {detail}")

        claimed = {}
        for j, b in self.beliefs.items():
            if b.source != "memory" and b.target is not None:
                claimed.setdefault(b.target, []).append(j)

        lines.append("\nTARGETS:")
        for i, t in enumerate(targets):
            if t.cleared:
                lines.append(f"  Target {i}: cleared")
                continue
            kind = t.target_type.value if known(t) else (
                self.learned_types[i].value if i in self.learned_types else "type unknown")
            text = f"  Target {i} ({kind}): open"
            if kind == "Sequential" and me in t.visitors:
                text += ", you have already visited it"
            if i in claimed:
                text += ", claimed by agents " + ", ".join(str(a) for a in sorted(claimed[i]))
            lines.append(text)
            if self.cfg.show_target_coords:
                d = math.dist(scene["location"], t.get_location())
                lines.append(f"      Location: X={t.north_m:.1f}m, Y={t.east_m:.1f}m, {d:.0f} meters away")

        lines.append("\nGive your belief for each silent teammate.")
        return "\n".join(lines)

    def _apply_llm(self, reply, silent, targets, rnd):
        """Validate the LLM's beliefs; anything that is not a legal belief is dropped."""
        beliefs = reply.get("beliefs") if isinstance(reply, dict) else None
        if not isinstance(beliefs, dict):
            return
        for key, val in beliefs.items():
            try:
                j = int(key)
            except (TypeError, ValueError):
                continue
            if j not in silent:
                continue
            if val is None or (isinstance(val, str) and val.strip().lower() in ("idle", "none", "null")):
                target = None
            else:
                m = re.fullmatch(r"\s*(?:target\s*)?(\d+)\s*", str(val), re.I) if not isinstance(val, bool) else None
                if m is None:
                    continue
                target = int(m.group(1))
                if target >= len(targets) or targets[target].cleared:
                    continue
            self.beliefs[j] = Belief(target, "llm", rnd)

    # -- planning ---------------------------------------------------------------
    def plan(self, agent_index, num_agents, targets, scene, ask_llm=None):
        """Returns ("FLY", idx) | ("IDLE", None) | ("RECALL", None).
        ask_llm(system_prompt, user_prompt) -> parsed JSON dict or None.
        May raise PlanningBudgetExceeded / PlanningFailed; the caller then uses fallback()."""
        rnd = scene["round"]
        if self._observed_round != rnd:
            self.observe(scene, targets)

        silent = self.silent()
        if (self.cfg.use_llm and ask_llm is not None and silent
                and not all(t.cleared for t in targets)):
            self.queries += 1
            reply = ask_llm(BELIEF_SYSTEM_PROMPT, self._belief_prompt(targets, scene, silent, rnd))
            self._apply_llm(reply, set(silent), targets, rnd)

        self.last_used = dict(self.beliefs)
        planner_scene = dict(scene)
        planner_scene["teammate_targets"] = {j: b.target for j, b in self.beliefs.items()}
        planner_scene["type_known"] = self._known_with_findings(scene, targets)

        t0 = time.perf_counter()
        try:
            return self.htn.plan(agent_index, num_agents, targets, planner_scene)
        finally:
            self.last_plan_s = time.perf_counter() - t0

    def _known_with_findings(self, scene, targets):
        learned = {targets[i] for i in self.learned_types if i < len(targets)}
        base = scene["type_known"]
        return lambda t: base(t) or t in learned

    def fallback(self, agent_index, targets, scene):
        """Constraint-respecting greedy choice for when the A* search cannot finish in its budget:
        the nearest open target not already covered by a believed claim."""
        if all(t.cleared for t in targets):
            return "RECALL", None
        known = self._known_with_findings(scene, targets)
        claims = Counter(b.target for b in self.beliefs.values()
                         if isinstance(b.target, int) and not self._dead(b.target, targets))
        best = None
        for i, t in enumerate(targets):
            if t.cleared:
                continue
            need = REQUIRED_DRONES[t.target_type] if known(t) else UNKNOWN_SLOT_GUESS
            if t.target_type == TargetType.SEQUENTIAL and known(t) and agent_index in t.visitors:
                continue
            if claims[i] >= need:
                continue
            d = math.dist(scene["location"], t.get_location())
            if best is None or d < best[0]:
                best = (d, i)
        return ("FLY", best[1]) if best else ("IDLE", None)

    # -- messaging -----------------------------------------------------------------
    def message_for(self, kind, target, targets, scene):
        """Event-triggered broadcast: this decision's claim plus every target type this drone knows."""
        findings = {}
        if self.cfg.share_types:
            known = self._known_with_findings(scene, targets)
            findings = {i: t.target_type for i, t in enumerate(targets) if not t.cleared and known(t)}
        return Codebook.encode(kind, target, findings)
