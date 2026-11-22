"""Hierarchical task network (HTN) planner in the layered style of the paper's task tree (Fig. 2).

Where htn_planner.py searches a flat STRIPS state space, this planner decomposes tasks level by level:

    mission                                  goal task (the paper's rescue(hikers))
      serve(t)                               intermediate task (confirm(P1)), one per open target
        fly_to(t), clear/join/hold/visit/scout(t)     primitive actions the drone executes (fly_to(P1))
      return_home                            closes every plan

Compound tasks are refined by methods. A method applies only if its precondition holds, and a task may
have several methods; the planner searches over the alternatives (depth-first, ordered nearest-first,
with branch-and-bound and memoisation) and keeps the cheapest complete decomposition. Constraints
live at the primitive level: an action enters a plan only if its preconditions hold in the state left
by the actions before it, including the time-horizon check "enough time remains to fly home afterwards"
(the analogue of the paper's battery reserve for the flight home).

The blanks of the tree are filled from outside (see hybrid_planner.py):
  * which targets are open        teammates' claims set options aside (a covered target is not in `mission`)
  * which method serves a Joint   a believed partner at the target selects `join`, otherwise `hold`
  * target types                  known types select the method; unknown types use `scout`

plan() has the same interface and return values as AStarHTNPlanner.plan, so it is a drop-in backend.
The decomposition of the last plan is kept as an indented tree in `last_tree`.
"""
import math
from collections import Counter, namedtuple

from environment import REQUIRED_DRONES
from htn_planner import (
    UNKNOWN_SLOT_GUESS,
    PlanningBudgetExceeded,
    PlanningFailed,
    _BudgetedPlanner,
)

State = namedtuple("State", "pos cost")   # pos: "cur" | target index | "home"; cost: rounds elapsed


# ---------------------------------------------------------------------------
# Generic HTN engine
# ---------------------------------------------------------------------------
# A task is a tuple (name, *args) with hashable args. Methods and operators are registered by task name.
#   method:   fn(state, *args) -> iterable of subtask lists (each one is an alternative refinement)
#   operator: fn(state, *args) -> new State, or None if the action is not permitted in `state`
class Method:
    def __init__(self, task_name, name, fn):
        self.task_name, self.name, self.fn = task_name, name, fn


class HTNSolver:
    def __init__(self, methods, operators, heuristic=None, check_budget=None):
        self.methods = {}
        for m in methods:
            self.methods.setdefault(m.task_name, []).append(m)
        self.operators = operators
        self.heuristic = heuristic or (lambda state, tasks: 0.0)
        self.check_budget = check_budget      # called once per expanded node; may raise
        self.expanded = 0

    def solve(self, state, tasks):
        """Cheapest complete decomposition of `tasks` from `state` -> (final_state, steps) or None.
        steps is the ordered list of ("D", task, method, subtasks) and ("P", task, cost) records.
        If the budget runs out after a first plan has been found, the best plan so far is returned."""
        self._best_cost, self._best_trace = math.inf, None
        self._best_state = None
        self._memo = {}
        self.expanded = 0
        try:
            self._dfs(state, tuple(tasks), None)
        except PlanningBudgetExceeded:
            if self._best_trace is None:
                raise
        if self._best_trace is None:
            return None
        steps = []
        trace = self._best_trace
        while trace is not None:
            step, trace = trace
            steps.append(step)
        steps.reverse()
        return self._best_state, steps

    def _dfs(self, state, tasks, trace):
        if self.check_budget:
            self.check_budget()
        self.expanded += 1
        if not tasks:
            if state.cost < self._best_cost:
                self._best_cost, self._best_trace, self._best_state = state.cost, trace, state
            return
        if state.cost + self.heuristic(state, tasks) >= self._best_cost:
            return
        key = (state.pos, tasks)
        if self._memo.get(key, math.inf) <= state.cost:
            return                          # reached this exact situation at least as cheaply before
        self._memo[key] = state.cost

        task, rest = tasks[0], tasks[1:]
        name, args = task[0], task[1:]
        if name in self.operators:
            nxt = self.operators[name](state, *args)
            if nxt is not None:
                self._dfs(nxt, rest, ((("P", task, nxt.cost - state.cost)), trace))
            return
        for method in self.methods.get(name, ()):
            for subtasks in method.fn(state, *args):
                self._dfs(state, tuple(subtasks) + rest, ((("D", task, method.name, tuple(subtasks))), trace))


# ---------------------------------------------------------------------------
# Decomposition tree (for logging)
# ---------------------------------------------------------------------------
class _Node:
    def __init__(self, task):
        self.task, self.method, self.cost, self.children = task, None, None, []


def build_tree(root_task, steps):
    root = _Node(root_task)
    pending = [root]
    for step in steps:
        node = pending.pop(0)               # decomposition is depth-first, left to right
        if step[0] == "D":
            node.method = step[2]
            node.children = [_Node(t) for t in step[3]]
            pending = node.children + pending
        else:
            node.cost = step[2]
    return root


def _label(task):
    name, args = task[0], task[1:]
    shown = []
    for a in args:
        shown.append("{" + ",".join(str(x) for x in sorted(a)) + "}" if isinstance(a, frozenset) else str(a))
    return f"{name}({','.join(shown)})" if shown else name


def render_tree(node, depth=0):
    label = _label(node.task)
    if node.method:
        label += f"  [{node.method}]"
    if node.cost:
        label += f"  ({node.cost:.1f} rounds)"
    lines = ["  " * depth + label]
    children = []

    def flatten(n):                         # mission(rem) refines into serve(t), mission(rem - t): show them as siblings
        for c in n.children:
            if c.task[0] == n.task[0]:
                flatten(c)
            else:
                children.append(c)
    flatten(node)
    for c in children:
        lines += render_tree(c, depth + 1)
    return lines


# ---------------------------------------------------------------------------
# The drone's task tree
# ---------------------------------------------------------------------------
def build_domain(open_targets, kind, partner_at, visited_by_me, dist, hold_cost, horizon):
    """kind: target -> 1, 2, 3 or None (unknown). partner_at: targets a teammate is believed to be at.
    dist(a, b): travel time in rounds between "cur", "home" and target indices."""

    # -- compound task: mission(rem) ------------------------------------------------------------
    def finish(state, rem):
        if not rem:
            yield [("return_home",)]

    def next_target(state, rem):            # one alternative per target, nearest first
        for t in sorted(rem, key=lambda t: (dist(state.pos, t), t)):
            yield [("serve", t), ("mission", rem - {t})]

    # -- compound task: serve(t), refined according to what is known about t -----------------------
    def serve_exclusive(state, t):
        if kind[t] == 1:
            yield [("fly_to", t), ("clear", t)]

    def serve_joint_with_partner(state, t):
        if kind[t] == 2 and t in partner_at:
            yield [("fly_to", t), ("join", t)]

    def serve_joint_hold(state, t):
        if kind[t] == 2:
            yield [("fly_to", t), ("hold", t)]

    def serve_sequential(state, t):
        if kind[t] == 3 and t not in visited_by_me:     # a drone cannot count twice toward a Sequential
            yield [("fly_to", t), ("visit", t)]

    def serve_unknown(state, t):
        if kind[t] is None:
            yield [("fly_to", t), ("scout", t)]

    methods = [
        Method("mission", "finish", finish),
        Method("mission", "next_target", next_target),
        Method("serve", "serve_exclusive", serve_exclusive),
        Method("serve", "serve_joint_with_partner", serve_joint_with_partner),
        Method("serve", "serve_joint_hold", serve_joint_hold),
        Method("serve", "serve_sequential", serve_sequential),
        Method("serve", "serve_unknown", serve_unknown),
    ]

    # -- primitive actions, with their constraints ------------------------------------------------
    def fly_to(state, t):
        cost = state.cost + dist(state.pos, t)
        if cost + dist(t, "home") > horizon:            # must still be able to fly home afterwards
            return None
        return State(t, cost)

    def at_target(c):
        def op(state, t):
            return State(state.pos, state.cost + c) if state.pos == t else None
        return op

    def return_home(state):
        cost = state.cost + dist(state.pos, "home")
        return State("home", cost) if cost <= horizon else None

    operators = {
        "fly_to": fly_to,
        "clear": at_target(1.0),
        "join": at_target(1.0),
        "hold": at_target(hold_cost),
        "visit": at_target(1.0),
        "scout": at_target(1.0),
        "return_home": return_home,
    }

    # Admissible bound: each pending target costs at least its cheapest incoming flight plus one service step
    service = min(1.0, hold_cost)
    nodes = ["cur"] + list(open_targets)
    min_in = {t: min((dist(u, t) for u in nodes if u != t), default=0.0) + service for t in open_targets}

    def heuristic(state, tasks):
        h = 0.0
        for task in tasks:
            if task[0] == "serve":
                h += min_in[task[1]]
            elif task[0] == "mission":
                h += sum(min_in[t] for t in task[1])
        return h

    return methods, operators, heuristic


class HierarchicalHTNPlanner(_BudgetedPlanner):
    """Plans a route over every target this drone believes still needs someone by decomposing
    mission -> serve(t) -> primitives, then returns the first leg. Replans each time the drone is prompted."""

    def __init__(self, time_budget_seconds, home=(0, 0, 0), horizon_rounds=None, hold_cost=1.0):
        super().__init__(time_budget_seconds)
        self.home = tuple(home)
        self.horizon_rounds = horizon_rounds      # total rounds available; None disables the time constraint
        self.hold_cost = hold_cost
        self.last_tree = ""                       # decomposition of the most recent plan
        self.last_plan = []                       # its primitive actions, e.g. ["fly_to(3)", "clear(3)", ...]
        self.expanded = 0                         # search nodes expanded by the most recent plan

    def plan(self, agent_index, num_agents, targets, scene):
        # scene: location, velocity, teammate_targets, type_known (and optionally round)
        # Returns ("RECALL", None) | ("IDLE", None) | ("FLY", target_index)
        self._start_round()
        self.last_tree, self.last_plan = "", []

        if all(t.visited for t in targets):
            return "RECALL", None

        me = agent_index
        known = scene["type_known"]
        kind = {i: (REQUIRED_DRONES[t.target_type] if known(t) else None) for i, t in enumerate(targets)}
        visited_by_me = {i for i, t in enumerate(targets) if me in t.visitors}

        # Teammate beliefs: an int is a believed target; None, -1 and "blocked" carry no claim
        predicted = {p: t for p, t in scene["teammate_targets"].items()
                     if isinstance(t, int) and t >= 0 and not targets[t].cleared}
        claims = Counter(predicted.values())

        # Blanks filled by teammates' claims: covered targets are set aside; the rest go into the mission
        open_targets = []
        for i, t in enumerate(targets):
            if t.cleared:
                continue
            k = kind[i]
            if k == 3 and i in visited_by_me:
                continue
            need = k if k is not None else UNKNOWN_SLOT_GUESS
            if claims[i] >= need:
                continue
            open_targets.append(i)
        if not open_targets:
            return "IDLE", None

        pos = {"cur": tuple(scene["location"]), "home": self.home}
        pos.update({i: targets[i].get_location() for i in open_targets})
        speed = scene["velocity"]

        def dist(a, b):
            return 0.0 if a == b else math.dist(pos[a], pos[b]) / speed   # rounds

        horizon = math.inf
        if self.horizon_rounds is not None:
            horizon = max(0.0, self.horizon_rounds - scene.get("round", 0))

        methods, operators, heuristic = build_domain(
            open_targets, kind, set(predicted.values()), visited_by_me, dist, self.hold_cost, horizon)
        solver = HTNSolver(methods, operators, heuristic, check_budget=self._remaining)
        root = ("mission", frozenset(open_targets))
        try:
            result = solver.solve(State("cur", 0.0), [root])
        finally:
            self.expanded = solver.expanded
        if result is None:
            raise PlanningFailed("no decomposition satisfies the constraints")
        _, steps = result

        self.last_tree = "\n".join(render_tree(build_tree(root, steps)))
        self.last_plan = [_label(s[1]) for s in steps if s[0] == "P"]
        first = next(s[1] for s in steps if s[0] == "P")       # from "cur" only fly_to is applicable
        return "FLY", first[1]
