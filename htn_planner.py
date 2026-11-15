"""Per-agent HTN planners. Each drone owns one planner and its own wall-clock budget.

Each drone has one planner; details from planners are not shared with each other.
Planners are deterministic, and will be consistent between drones.
"""
import heapq
import itertools
import math
import time
from collections import Counter

from environment import REQUIRED_DRONES

UNKNOWN_SLOT_GUESS = 1

class PlanningBudgetExceeded(Exception):
    pass


class PlanningFailed(Exception):
    pass


class _BudgetedPlanner:
    def __init__(self, time_budget_seconds: float):
        self.time_budget = time_budget_seconds
        self.started_at = None  # Time that the planner started; set on this drone's first plan() call

    # Check how much time is remaining; if no time remains, throw an exception
    def _remaining(self):
        now = time.perf_counter()
        if self.started_at is None:
            self.started_at = now
        remaining = self.time_budget - (now - self.started_at)
        if remaining <= 0:
            raise PlanningBudgetExceeded(f"over {self.time_budget:.1f}s budget")
        return remaining
    
    def _start_round(self):
        """Restart this drone's clock; call at the top of each plan()."""
        self.started_at = time.perf_counter()

class Action:
    def __init__(self, name, preconditions, add_effects, del_effects, cost=1.0):
        self.name = name
        self.preconditions = set(preconditions)
        self.add_effects = set(add_effects)
        self.del_effects = set(del_effects)
        self.cost = cost

    def is_applicable(self, state):
        return self.preconditions.issubset(state)

    def apply(self, state):
        return (state - self.del_effects) | self.add_effects

    def __repr__(self):
        return f"Action({self.name})"


class AStarPlanner:
    def __init__(self, actions, check_budget=None):
        self.actions = actions
        self.check_budget = check_budget   # called once per expanded node; may raise
        self._unit = 0.0

    def heuristic(self, state, goal):
        # Each unmet goal fact needs its own achiever action (assumes an action adds at most
        # one goal fact), so the cheapest achiever's cost is an admissible per-fact bound.
        return self._unit * len(goal - state)

    def solve(self, initial_state, goal_state):
        start, goal = frozenset(initial_state), frozenset(goal_state)
        self._unit = min((a.cost for a in self.actions if a.add_effects & goal), default=0.0)

        tie = itertools.count()   # unique tiebreaker: states and paths are never compared
        pq = [(self.heuristic(start, goal), 0.0, next(tie), start, [])]
        best = {start: 0.0}

        while pq:
            _, cost, _, state, path = heapq.heappop(pq)
            if self.check_budget:
                self.check_budget()
            if goal <= state:
                return path
            if cost > best.get(state, math.inf):
                continue   # stale entry; a cheaper route to this state was found later
            for action in self.actions:
                if action.is_applicable(state):
                    nxt = action.apply(state)
                    new_cost = cost + action.cost
                    if new_cost < best.get(nxt, math.inf):
                        best[nxt] = new_cost
                        heapq.heappush(pq, (new_cost + self.heuristic(nxt, goal), new_cost,
                                            next(tie), nxt, path + [action.name]))
        return None


def build_state(me, predicted_positions):
    # predicted_positions: dict drone_id -> target it is believed heading to
    state = {f"at_{me}_cur"}
    for p, t in predicted_positions.items():
        if p != me and t is not None:
            state.add(f"at_{p}_{t}")
    return state


def build_goal(open_targets):
    return {f"served_{t}" for t in open_targets}


def build_actions(me, n_drones, open_targets, dist, target_type, visited_by_me):
    """
    open_targets:  uncleared targets worth planning for (nodes are 'cur' plus these)
    dist(a, b):    travel cost in rounds; a, b are 'cur' or a target index
    target_type:   dict t -> 1, 2, 3, or None (unknown), from this drone's beliefs
    visited_by_me: targets this drone already visited (for k=3)
    """
    nodes = ["cur"] + list(open_targets)
    partners = [p for p in range(n_drones) if p != me]
    acts = []

    for a in nodes:
        for t in open_targets:
            if a != t:
                acts.append(Action(f"go_{me}_{a}_{t}",
                                   [f"at_{me}_{a}"], [f"at_{me}_{t}"], [f"at_{me}_{a}"],
                                   cost=dist(a, t)))

    for t in open_targets:
        k = target_type.get(t)
        here = f"at_{me}_{t}"

        if k is None:
            acts.append(Action(f"scout_{me}_{t}", [here], [f"served_{t}"], [], cost=0.5))
        elif k == 1:
            acts.append(Action(f"clear_{me}_{t}", [here],
                               [f"served_{t}", f"cleared_{t}"], [], cost=0.5))
        elif k == 2:
            # Effects are just served_t for both, so join and hold reach the same state
            # and A* keeps the cheaper one instead of searching duplicate states.
            for p in partners:
                acts.append(Action(f"join_{me}_{p}_{t}", [here, f"at_{p}_{t}"],
                                   [f"served_{t}"], [], cost=0.5))
            acts.append(Action(f"hold_{me}_{t}", [here], [f"served_{t}"], [], cost=3.0))
        elif k == 3 and t not in visited_by_me:
            acts.append(Action(f"visit_{me}_{t}", [here],
                               [f"served_{t}", f"visited_{me}_{t}"], [], cost=0.5))
    return acts


class AStarHTNPlanner(_BudgetedPlanner):
    """Plans a route over every target this drone believes still needs someone, then returns
    the first leg. Replans each time the drone is prompted, so beliefs stay current."""

    def plan(self, agent_index, num_agents, targets, scene):
        # scene: location, velocity, teammate_targets, type_known (built by AgentHandler)
        # Returns ("RECALL", None) | ("IDLE", None) | ("FLY", target_index)
        self._start_round()

        if all(t.visited for t in targets):
            return "RECALL", None

        me = agent_index
        known = scene["type_known"]
        target_type = {i: (REQUIRED_DRONES[t.target_type] if known(t) else None)
                       for i, t in enumerate(targets)}
        visited_by_me = {i for i, t in enumerate(targets) if me in t.visitors}

        # Teammate beliefs: an int is a believed target; None, -1 and "blocked" carry no claim
        predicted = {p: t for p, t in scene["teammate_targets"].items()
                     if isinstance(t, int) and t >= 0 and not targets[t].cleared}
        claims = Counter(predicted.values())

        # Open = uncleared, not a Sequential I already counted toward, not already covered
        open_targets = []
        for i, t in enumerate(targets):
            if t.cleared:
                continue
            k = target_type[i]
            if k == 3 and i in visited_by_me:
                continue
            need = k if k is not None else UNKNOWN_SLOT_GUESS
            if claims[i] >= need:
                continue
            open_targets.append(i)
        if not open_targets:
            return "IDLE", None

        pos = {"cur": tuple(scene["location"])}
        pos.update({i: targets[i].get_location() for i in open_targets})
        speed = scene["velocity"]

        def dist(a, b):
            return math.dist(pos[a], pos[b]) / speed   # rounds, same scale as action costs

        actions = build_actions(me, num_agents, open_targets, dist, target_type, visited_by_me)
        path = AStarPlanner(actions, check_budget=self._remaining).solve(
            build_state(me, predicted), build_goal(open_targets))
        if not path:
            raise PlanningFailed("no plan reaches the goal")

        # From 'cur' the only applicable actions are go_<me>_cur_<t>
        return "FLY", int(path[0].rsplit("_", 1)[1])
