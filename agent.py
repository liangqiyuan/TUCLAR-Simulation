import csv
import json
import os
import random
import statistics
import time
from dataclasses import dataclass

import httpx
from openai import APITimeoutError, BadRequestError, OpenAI

import utils_geometry as geo
from environment import VISIT_RADIUS_METERS, Environment, TargetType
from htn_planner import (
    AStarHTNPlanner,
    PlanningBudgetExceeded,
    PlanningFailed,
)
from htn_hierarchical import HierarchicalHTNPlanner
from hybrid_planner import HybridConfig, HybridPlanner

"""
Agent:  Represents an AI agent-controlled drone, and contains relevant information such as location
        and agent history. Mostly contains functions for prompting the agents.

AgentHandler:   A class that helps the simulation "handle" agents. Parses the agent's outputs,
                computes their velocity, handles prompting. It should be emphasized that this class
                is for programming convenience only; it should NOT be used as a centralized manager
                that exchanges information to and from agents freely, as that would undermine the
                goal of the research.

"""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# LLM Settings
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_API_KEY = os.environ.get("NVIDIA_API_KEY", "EMPTY")
LLM_MODEL = os.environ.get("MODEL", "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")
LLM_TIME_LIMIT_SECONDS = 30
LLM_ENABLE_REASONING = True
LLM_REMEMBER_REASONING = True
MAX_TOKENS = 32000
TEMPERATURE = 0.0
HISTORY_TURNS = 3

# Simulation settings
MAX_ROUNDS = 500
UNKNOWN_SLOT_GUESS = 1

# CSV logs go here. Set LOG_DIR to write each experiment to its own folder.
LOG_DIR = os.environ.get("LOG_DIR", ".")
log_path = os.path.join(LOG_DIR, "log.csv")
summary_log_path = os.path.join(LOG_DIR, "summary_log.csv")

# --- Experiment configs
PROMPT_STYLE = "short"        # "short": message/action  |  "framework": state/role/task/plan/message/action
SHOW_OWN_LOCATION = False     # tell each agent its own X/Y
SHOW_TARGET_COORDS = False    # tell each agent every pending target's X/Y and distance

# Planner backend (used when PLANNER_MODE == "htn"):
#   "htn_llm"   determinisitic HTN planning, will delegate to LLM if it takes too long.
#   "llm" llm only, no deterministic planning.
#   "hybrid"    A* planner chooses every action; the LLM only forms beliefs about silent teammates;
#               drones share claims/findings through a structured message codebook (hybrid_planner.py).
PLANNER_MODE = "llm"
HTN_TIME_BUDGET_SECONDS = 15  # real-world seconds per drone, from its first HTN decision
# Which planner decides the actions in "htn_llm" and "hybrid" modes:
#   "astar"         flat STRIPS-style A* search (htn_planner.py)
#   "hierarchical"  layered task tree: mission -> serve(target) -> primitive actions (htn_hierarchical.py)
HTN_BACKEND = "astar"
HTN_ENFORCE_HORIZON = True    # hierarchical only: a plan must leave time to fly home before MAX_ROUNDS
HTN_HOLD_COST = 1.0           # hierarchical only: cost (rounds) of holding at a Joint target without a partner

# Hybrid planner settings (used when PLANNER_MODE == "hybrid"). Switching use_llm / use_messages off
# separately gives the reasoning-only, communication-only and neither ablations.
HYBRID_USE_LLM = True                 # LLM beliefs about silent teammates
HYBRID_USE_MESSAGES = True            # read and send codebook messages (claim(T), type(T,K), idle)
HYBRID_SHARE_TYPES = True             # include type(T,K) findings in messages and accept them from teammates
HYBRID_BELIEF_TTL = 10                # rounds before an LLM belief lapses
HYBRID_MEMORY_TTL = 10                # rounds before a stale last-known claim is dropped (stops phantom claims)
HYBRID_STALE_FALLBACK = "last_known"  # when no fresh info and no LLM belief: "last_known" or "none"
HYBRID_PRINT_BELIEFS = False          # print each belief query's raw LLM output and parsed reply (None = timeout or unparseable)
HYBRID_COLOR_BELIEFS = True           # color the printed beliefs: green = matches what the teammate is doing, red = wrong, yellow = invalid

# EXPERIMENTAL
#   "all"        every drone, always (original behavior)
#   "any_visit"  every drone, once any drone has visited the target
#   "own_visit"  a drone only learns a type by visiting that target itself
TARGET_TYPE_VISIBILITY = "own_visit"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_SHORT = """
Your output must be in valid JSON format. Do not output anything else.
Your JSON must have "message", and "action" components.
You can communicate to your teammates through "message", but they may or may not receive it.
"""

SYSTEM_PROMPT_FRAMEWORK = """
## Coordinating from the shared observation
Decide from the observation you and the other agents all see: reading the same scene and reasoning alike, a rule anchored to that scene leads you all to the same division of labor.

**Role.** Read the ownership: is it overlapping -- do you and the other agents all work the same targets and resources -- or is it divided, each of you already holding its own part?
  - **Overlapping** (acting alike you would contend for one target or all defer): coordinate -- settle a division of labor over the shared targets from the scene.
  - **Divided** (your role owns some stages/stations and the other agents own the rest): hold your part and execute. Take the single most useful action within your own part; do not step onto a station or task that belongs to another agent's part, even if it looks like the most useful step right now -- reading the same scene, they are already taking it, so you would only collide. If your own part has no ready step (it waits on their output), do its enabling step or start the next independent unit; wait only when nothing of yours is productive.

**Task.** Settle the division over the shared targets: read how each pending target couples you and apply its rule over the canonical order:
  1. **Joint** (succeeds only if you all act on it together): converge -- all take the same one: the first such target in that order. (Acting alone wastes it.)
  2. **Exclusive** (single-occupancy -- multiple acting on it wastes it): divide by item -- take your share by role over that order; the other agents take the complement, so you never collide.
  3. **Sequential** (ordered hand-off chain): divide by stage -- work the stage your role owns, by the chain order.

## Reply format
Reply with a single JSON object and nothing else:

{
  "state":   "<from the observation only: what you hold + which steps are doable now (inputs present); world state, never the other agents' intent>",
  "role":    "<is ownership overlapping (you and the other agents all work the same targets/resources -> coordinate) or divided (your role owns some stages/stations, the others own the rest -> execute your own), and why>",
  "task":    "<for each pending shared target, how it couples you -- Joint / Exclusive / Sequential>",
  "plan":    "<overlapping: apply the matching rule per target over the canonical order -- Joint: converge on the first such target; Exclusive: your role's share; Sequential: your stage; divided: the most useful step within your own part (enabling step or next unit if none ready), not a step that belongs to another agent's part>",
  "message": "<whatever you enter here may or may not reach your teammates. communication is unreliable in this scenario, but try your best to collaborate>",
  "action":  "<copy one action verbatim from your currently-legal actions>"
}
"""

SYSTEM_PROMPT_TOS = """
## Context
You are an agent that is working with other agents. You are working together towards the same goal, but communication is limited.

## How to decide what to do
You will be presented with tasks, some of which may require multiple teammates simultaneously to work on.
You and each of your teammates are a assigned a number, and each task is assigned a number as well.
You are to work on these in the order they are presented. However, assume that your teammates deal with these tasks in order.
For example, if you are Agent 3, and you are presented with 5 available slots for work, you are to take slot 3.
This is because slots 0-2 will have already been claimed by Agents 0-2. Even if they show as empty in the prompt, assume that the change just isn't visible to you yet.

## Reply format
Reply with a single JSON object and nothing else:

{
    "thought": "Enter your reasoning here",
    "message": "Enter a message here. Your teammates may or may not get it; communication is unreliable.",
    "action": "Your choice of action."
}
"""

SYSTEM_PROMPTS = {"short": SYSTEM_PROMPT_SHORT, "framework": SYSTEM_PROMPT_FRAMEWORK, "tos": SYSTEM_PROMPT_TOS}

HOME = [0, 0, 0]


def make_htn_planner():
    """A fresh per-drone planner for the configured HTN_BACKEND."""
    if HTN_BACKEND == "astar":
        return AStarHTNPlanner(HTN_TIME_BUDGET_SECONDS)
    if HTN_BACKEND == "hierarchical":
        return HierarchicalHTNPlanner(HTN_TIME_BUDGET_SECONDS, home=HOME,
                                      horizon_rounds=MAX_ROUNDS if HTN_ENFORCE_HORIZON else None,
                                      hold_cost=HTN_HOLD_COST)
    raise ValueError(f"unknown HTN_BACKEND {HTN_BACKEND!r}")

# Tolerates raw newlines inside JSON strings (a failure the model produces now and then)
_decoder = json.JSONDecoder(strict=False)

def extract_json_object(text, key="action"):
    """Return the last parseable JSON object containing `key`; ignores braces in reasoning text."""
    if not text:
        return None
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    for start in reversed([i for i, ch in enumerate(text) if ch == "{"]):
        try:
            obj, _ = _decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and key in obj:
            return obj
    return None

def split_reasoning(reasoning, content):
    if not reasoning and "</think>" in content:
        reasoning, content = content.split("</think>", 1)
        reasoning = reasoning.replace("<think>", "")
    return reasoning.strip(), content.strip()


def type_is_known(target, agent_index, mode=None):
    mode = mode or TARGET_TYPE_VISIBILITY
    if mode == "all":
        return True
    if mode == "any_visit":
        return bool(target.visitors)
    return agent_index in target.visitors

def compute_velocity(agent, environment, wind):
    """One round of velocity: motion toward the target plus wind.
    IDLE drones and drones landed at home are not blown around."""
    target = agent.target_index
    if target is None or target == -1:
        return [0, 0, 0]

    if target == "Home":
        distance = geo.point_distance(agent.location, HOME)
        if distance < 1:
            return [0, 0, 0]
        if distance <= agent.drone_velocity:
            return [-c for c in agent.location]       # final step: lands exactly, no wind
        scale = agent.drone_velocity / distance
        cmd = [-c * scale for c in agent.location]
    elif isinstance(target, int):
        goal = environment.target_locations[target].get_location()
        distance = geo.point_distance(agent.location, goal)
        if distance < 1e-6:
            cmd = [0, 0, 0]
        else:
            scale = min(agent.drone_velocity, distance) / distance
            cmd = [(goal[i] - agent.location[i]) * scale for i in range(3)]
    else:
        print(f"WARNING: Non-integer target index: {target}")
        return [0, 0, 0]

    return [cmd[0] + wind[0], cmd[1] + wind[1], cmd[2]]

@dataclass
class DroneParams:
    comms_quality: float = 0.8              # chance a message this drone sends gets through (0-1)
    comms_distance: float = 1000            # transmit distance in meters; -1 = unlimited
    drone_velocity: float = 100             # Distance travelled per round (before wind effects)
    failed_inference_chance: float = 0.15   # chance this drone misreads a teammate's target

# Handles LLM calls & message history
class Agent:

    def __init__(self, system_prompt: str, client, model: str, start_location: [float, float, float], agent_index=-1, params: DroneParams | None = None):
        self.history = [{"role": "system", "content": system_prompt}]
        self.recent = []  # last HISTORY_TURNS exchanges, stored compactly
        self.client = client
        self.model = model
        self.location = start_location # Distance North from center, distance East from center, altitude
        self.velocity = [0,0,0] # North velocity (m/s), East velocity (m/s), Upward velocity (m/s)
        self.scatter_pointer = None # A pointer to this agent's associated drone on the map. Assigned in main.
        self.target_index = -1
        self.last_target = -1
        self.agent_index = agent_index
        self.inaction = False
        self.distance_traveled = 0.0
        self.displayed_message = "Yet to act, no message for now."
        self.response_times = []
        self.last_stats = {}
        self.call_stats = []

        self.htn = make_htn_planner()
        self.hybrid = HybridPlanner(agent_index, HTN_TIME_BUDGET_SECONDS, HybridConfig(
            use_llm=HYBRID_USE_LLM,
            use_messages=HYBRID_USE_MESSAGES,
            share_types=HYBRID_SHARE_TYPES,
            belief_ttl=HYBRID_BELIEF_TTL,
            memory_ttl=HYBRID_MEMORY_TTL,
            stale_fallback=HYBRID_STALE_FALLBACK,
            show_target_coords=SHOW_TARGET_COORDS,
            show_own_location=SHOW_OWN_LOCATION,
        ), planner=self.htn)
        self.last_raw = {}      # raw output of the latest query_json call
        self.belief_hits = {}   # belief source -> [correct, total]; scored against ground truth, logging only
        self.delegated = False
        self.plan_times = []
        self.fallbacks = 0

        p = params or DroneParams()
        self.comms_quality = p.comms_quality
        self.comms_distance = p.comms_distance
        self.drone_velocity = p.drone_velocity
        self.failed_inference_chance = p.failed_inference_chance

    
# _create:    prompts the AI, and records various timing & token metrics of the AI's output.
#            returns the reasoning and the non-reasoning parts of the response separately.    
    def _create(self, messages):
        t0 = time.perf_counter()
        reasoning, content = [], []
        t_first = t_reason_end = None
        finish = usage = stream = None
        timed_out = False
        try:
            # Handles the AI output as a stream, so it can record reasoning vs answering time
            stream = self.client.chat.completions.create(
                model=self.model, messages=messages,
                max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
                extra_body={"chat_template_kwargs": {"enable_thinking": LLM_ENABLE_REASONING, "thinking": LLM_ENABLE_REASONING}},
                stream=True,
                stream_options={"include_usage": True},
                timeout=LLM_TIME_LIMIT_SECONDS,
            )
            for chunk in stream:
                now = time.perf_counter()
                if now - t0 > LLM_TIME_LIMIT_SECONDS:
                    timed_out = True
                    break
                if chunk.usage:
                    usage = chunk.usage
                if not chunk.choices:
                    continue
                ch = chunk.choices[0]
                d = ch.delta
                finish = ch.finish_reason or finish
                r = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
                if r:
                    t_first = t_first or now
                    reasoning.append(r)
                    t_reason_end = now
                if d.content:
                    t_first = t_first or now
                    content.append(d.content)
                    if t_reason_end is None and "</think>" in "".join(content):
                        t_reason_end = now
        except (APITimeoutError, httpx.TimeoutException):
            timed_out = True
        finally:
            if stream is not None:
                stream.close()
        t_end = time.perf_counter()

        t_first = t_first or t_end
        answer_start = t_reason_end or t_first
        self.last_stats = {
            "ttft": t_first - t0,
            "reasoning_s": (t_reason_end - t_first) if t_reason_end else 0.0,
            "answer_s": t_end - answer_start,
            "total_s": t_end - t0,
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None,
            "finish_reason": finish,
            "timed_out": timed_out,
        }
        return "".join(reasoning), "".join(content)

    # Prompt the AI using _create, apply some error & text parsing.
    def generate_response(self, message):
        messages = self.history + self.recent + [{"role": "user", "content": message}]
        started = time.perf_counter()
        try:
            reasoning, content = self._create(messages)
        except BadRequestError as e:
            if "context length" in str(e).lower() and self.recent:
                self.recent = []
                reasoning, content = self._create(self.history + [{"role": "user", "content": message}])
            else:
                raise
        self.response_times.append(time.perf_counter() - started)

        if self.last_stats["timed_out"]:
            self.last_stats["invalid_json"] = False  # timeouts are counted separately
            self.call_stats.append(self.last_stats)
            print(f"WARNING: agent {self.agent_index} exceeded {LLM_TIME_LIMIT_SECONDS}s; keeping previous course.")
            return None

        reasoning, content = split_reasoning(reasoning, content)
        parsed = extract_json_object(content)
        self.last_stats["invalid_json"] = parsed is None
        self.call_stats.append(self.last_stats)

        if parsed is not None and HISTORY_TURNS > 0:
            reply = json.dumps({
                "action": parsed.get("action", ""),
                "message": parsed.get("message", ""),
            })
            if LLM_REMEMBER_REASONING and reasoning:
                reply = f"<think>\n{reasoning}\n</think>\n{reply}"
            self.recent += [
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply},
            ]
            self.recent = self.recent[-2 * HISTORY_TURNS:]
        return f"<think>{reasoning}</think>{content}" if reasoning else content

    # One stateless LLM call that must return a JSON object containing `key` (no chat history is kept).
    # Used by the hybrid planner for belief queries; timing and token stats join call_stats.
    def query_json(self, system_prompt, message, key):
        started = time.perf_counter()
        try:
            reasoning, content = self._create([{"role": "system", "content": system_prompt},
                                               {"role": "user", "content": message}])
        except BadRequestError as e:
            print(f"WARNING: agent {self.agent_index} belief query rejected: {e}")
            return None
        self.response_times.append(time.perf_counter() - started)
        self.last_raw = {"reasoning": reasoning, "content": content}   # unprocessed model output, for printing

        if self.last_stats["timed_out"]:
            self.last_stats["invalid_json"] = False
            self.call_stats.append(self.last_stats)
            print(f"WARNING: agent {self.agent_index} belief query exceeded {LLM_TIME_LIMIT_SECONDS}s.")
            return None

        reasoning, content = split_reasoning(reasoning, content)
        parsed = extract_json_object(content, key)
        self.last_stats["invalid_json"] = parsed is None
        self.call_stats.append(self.last_stats)
        return parsed

    # Advance position according to velocity
    def step(self):
        step_distance = geo.point_distance([0, 0, 0], self.velocity)  # magnitude of velocity this round
        self.distance_traveled += step_distance
        self.location[0] += self.velocity[0]
        self.location[1] += self.velocity[1]
        self.location[2] += self.velocity[2]

    # Scalar distance between an Agent and its target.
    def distance_to_target(self, environment):
        if self.target_index is None:
            return None
        target = environment.target_locations[self.target_index]
        return geo.point_distance(self.location, target.get_location())

    # Scalar distance between an Agent and its previous target.
    def distance_to_last_target(self, environment):
        if self.last_target is None:
            return None
        target = environment.target_locations[self.last_target]
        return geo.point_distance(self.location, target.get_location())


# Handles the LLMs as a group.
class AgentHandler:

    # Initiates the AI, creates the system prompts, initializes variables
    def __init__(self, num_agents: int, scenario: int, environment: Environment, drone_params: list[DroneParams] | None = None):
        self.client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        self.model = LLM_MODEL

        self.scenario = scenario
        self.environment = environment
        self.num_rounds = 0
        self.seconds_per_round = LLM_TIME_LIMIT_SECONDS
        self.sim_start = None
        self.sim_end = None
        self.teammate_views = {}

        self.system_prompt = SYSTEM_PROMPTS[PROMPT_STYLE]

        self.scenario_system_prompts = {

            # Scenario 0 is a test scenario, N/A for the research
            0: """
                MISSION REQUIREMENTS:
                    1. You will be given a target location and a current location.
                    2. Your goal is to move within 50 meters of the target location, including altitude difference.
                    3. You may only move one direction at a time.
            """,
            1: """
                MISSION REQUIREMENTS:
                    1. There is a series of locations that you will be given.
                    2. You will have a series of teammates. All your teammates reason exactly the same as you. The prompt will tell you what they are doing, but each prompt has a chance of giving the wrong value. Be cautious of this input error.
                    3. The mission is a success once all target locations are cleared, and you have used the RECALL action to go back to start.
                    4. There are three types of target location: Sequential, which must be visited by three different agents at any point (you do not need to all visit at the same time). Joint, which must be visited by two different agents at the *same time*. And Exclusive, which must be visited by any agent.
                    5. Complete the mission as quick as possible.
                    6. Pick ONE location to visit at a time. You have 3 available actions:
                            "IDLE": Do nothing.
                            "FLY 0", "FLY 1", "FLY 2"...: Travel to the target with the given index. This is the TARGET index, not the SLOT index.
                            "RECALL": Go back to your starting location.
                    7. It takes multiple rounds for you to travel to a location. You may need to use the FLY action multiple times on the same target before you arrive. Do not assume a target has been visited until it has been cleared from the target list.

                    Example JSON output: {"message": "communicate with your teammates here", "action": "Enter your action here"}
            """
        }

        self.system_prompt += self.scenario_system_prompts[scenario]
        if scenario == 1 and TARGET_TYPE_VISIBILITY != "all":
            who = "any drone has" if TARGET_TYPE_VISIBILITY == "any_visit" else "you have"
            self.system_prompt += (f"\nNOTE: A target's type is shown as \"type unknown\" until {who} visited it, so you may not know how many drones it needs.\n")

        if scenario == 1 and self.environment.vision_obstacles:
            self.system_prompt += ("\nNOTE: Obstacles may block line of sight between drones. If line of sight to a teammate is blocked, you cannot see which target they are heading to. Radio messages are unaffected.\n")

        self.num_agents = num_agents
        self.agents = []
        if drone_params is None:
            drone_params = [DroneParams() for _ in range(num_agents)]
        if len(drone_params) != num_agents:
            raise ValueError("drone_params must have one entry per agent")
        self.drone_params = drone_params
        for idx in range(num_agents):
            self.agents.append(Agent(system_prompt=self.system_prompt, client=self.client, model=self.model, start_location=[0, 0, 1000], agent_index=idx, params=drone_params[idx]))

        self.log_path = log_path
        self.summary_log_path = summary_log_path
        self.seed = None   # scenario seed of this run, set by main.py; logged in the summary
        self.init_log()


    def get_total_distance_traveled(self):
        return sum(agent.distance_traveled for agent in self.agents)

    # Returns the average number of unnecessary vists across each target
    def get_revisit_rate(self):
        targets = self.environment.target_locations
        if not targets:
            return 0.0
        wasted_visits = sum(len(t.visitors) - t.required_drones for t in targets if len(t.visitors) > t.required_drones)
        return wasted_visits / sum(t.required_drones for t in targets)

    def get_sim_time(self):
        """Seconds from the start of the first round to the end of the last one."""
        if self.sim_start is None or self.sim_end is None:
            return 0.0
        return self.sim_end - self.sim_start

    def get_average_response_time(self):
        """Mean LLM response time over every call made by every drone (None if no calls)."""
        times = [t for agent in self.agents for t in agent.response_times]
        return sum(times) / len(times) if times else None

    def log_summary(self, success: bool, optimal=None, rule=None):
        # total_distance_traveled: Collective distance traveled by all drones
        # revisit_rate: The rate at which drones visited targets they did not need to (rough approximation of poor division of labor)
        # success: Whether or not the simulation succeeded
        # num_rounds: The number of rounds the simulation took to run (Most important for measuring agent problem solving ability)
        # comms_quality: The chance for communications within range to succeed (0-1)
        # sim_time_seconds: The total time the simulation ran, in seconds
        # avg_ttft: Average time-to-first-token for each AI call.
        # std_ttft: Std deviation of times-to-first-token.
        # avg_reasoning_s: Average time spent reasoning for each AI call.
        # std_reasoning_s: Std deviation of reasoning time.
        # avg_answer_s: Average time spent answering for each AI call.
        # std_answer_s: Std deviation of answer time.
        # avg_completion_tokens: Average tokens used for each AI call.
        # std_completion_tokens: Std deviation of completion tokens.
        # optimal_distance: Estimated optimal solution (system is not gaurenteed to find best possible solution)
        # optimal_rounds: The number of rounds the estimated optimal solution took to finish.
        # avg_distance: The average distance the drones would fly if they made decisions completely at random.
        # avg_rounds: The average number of rounds the drones would take to finish if they made decisions completely at random.
        # prompt_style: the prompt style used.
        # planning_style: the planning style used.
        # num_agents: the number of agents in the simulation
        # num_exclusive: The number of exclusive targets.
        # num_joint: The number of joint targets.
        # num_sequential: The number of sequential targets.
        # num_targets: The total number of targets.
        # num_vision_obstacles: Number of obstacles that block vision
        # area_size: Radius of the simulation area, in meters.
        # agent_speed: The maximum distance an agent can move in a round, in meters.
        # num_llm_calls: The number of times an agent was prompted
        # timeout_rate: The frequency that a model timed out.
        # invalid_json_rate: The frequency that a model output in an invalid format
        # failed_inference_chance: The chance that a drone will mistake which target a teammate is heading to
        # rule_avg_distance: The average distance of the simulations in the non-LLM, non-HTN setup. For comparison only.
        # rule_avg_rounds: The average number of rounds required in the simulations of the non-LLM, non-HTN setup. For comparison only.
        #TODO: Update this comment block
        header = ["total_distance_traveled", "revisit_rate", "success", "num_rounds", "comms_quality",
          "sim_time_seconds", "avg_ttft", "std_ttft", "avg_reasoning_s", "std_reasoning_s",
          "avg_answer_s", "std_answer_s", "avg_completion_tokens", "std_completion_tokens", "optimal_distance",
          "optimal_rounds", "rule_success_rate", "rule_avg_distance", "rule_avg_rounds", "rule_invalid_actions",
          "prompt_style", "planning_style", "num_agents",
          "num_exclusive", "num_joint", "num_sequential", "num_targets", "num_vision_obstacles", "area_size", "agent_speed",
          "num_llm_calls", "timeout_rate", "invalid_json_rate", "failed_inference_chance", "type_visibility",
          "rule_ns_per_drone_round", "htn_avg_plan_s", "htn_fallback_rate", "comms_distance", "schedule",
          "hybrid_config", "belief_queries", "belief_acc_msg", "belief_acc_obs", "belief_acc_llm", "belief_acc_memory", "htn_backend", "seed"]
        

        target_types = self.environment.get_target_amounts()
        row = [
            f"{self.get_total_distance_traveled():.2f}",
            f"{self.get_revisit_rate():.4f}",
            success,
            self.num_rounds,
            self._per_agent("comms_quality"),
            f"{self.get_sim_time():.2f}",
            self._avg_stat("ttft"),
            self._std_stat("ttft"),
            self._avg_stat("reasoning_s"),
            self._std_stat("reasoning_s"),
            self._avg_stat("answer_s"),
            self._std_stat("answer_s"),
            self._avg_stat("completion_tokens"),
            self._std_stat("completion_tokens"),
            "" if optimal is None else f"{optimal.distance:.2f}",
            "" if optimal is None else optimal.rounds,
            "" if rule is None else f"{rule.success_rate:.2f}",
            "" if rule is None or rule.avg_distance is None else f"{rule.avg_distance:.2f}",
            "" if rule is None or rule.avg_rounds is None else f"{rule.avg_rounds:.2f}",
            "" if rule is None else rule.invalid_actions,
            PROMPT_STYLE,
            PLANNER_MODE,
            self.num_agents,
            target_types[0],
            target_types[1],
            target_types[2],
            sum(target_types),
            self.environment.area_size_m,
            self._per_agent("drone_velocity"),
            self._num_calls(),
            self._rate_stat("timed_out"),
            self._rate_stat("invalid_json"),
            self._per_agent("failed_inference_chance"),
            TARGET_TYPE_VISIBILITY,
            "" if rule is None or rule.avg_ns_per_drone_round is None else f"{rule.avg_ns_per_drone_round:.0f}",
            self._avg_plan_s(),
            self._fallback_rate(),
            self._per_agent("comms_distance"),
            self.environment.SCHEDULE or "",
            self.agents[0].hybrid.cfg.label() if PLANNER_MODE == "hybrid" else "",
            sum(a.hybrid.queries for a in self.agents) if PLANNER_MODE == "hybrid" else "",
            self._belief_acc("msg"),
            self._belief_acc("obs"),
            self._belief_acc("llm"),
            self._belief_acc("memory"),
            HTN_BACKEND if PLANNER_MODE in ("htn_llm", "hybrid") else "",
            "" if self.seed is None else self.seed,
        ]

        os.makedirs(os.path.dirname(self.summary_log_path) or ".", exist_ok=True)

        # Columns changed since the last run? Keep the old file rather than mixing layouts.
        if os.path.exists(self.summary_log_path):
            with open(self.summary_log_path, newline="") as f:
                existing_header = next(csv.reader(f), None)
            if existing_header != header:
                backup = f"{self.summary_log_path}.{int(time.time())}.bak"
                os.rename(self.summary_log_path, backup)
                print(f"Summary log columns changed; old log moved to {backup}")

        file_exists = os.path.exists(self.summary_log_path)
        with open(self.summary_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)

    def init_log(self):
        header = [f"agent{i}" for i in range(self.num_agents)] + \
                 [f"target{i}" for i in range(len(self.environment.target_locations))]
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # Records the Agent decisions round-by-round. Not generally useful, but
    # possibly helpful for debugging.
    def log_round(self):
        agent_targets = [
            agent.target_index if agent.target_index is not None else -1
            for agent in self.agents
        ]

        target_visitors = []
        for target in self.environment.target_locations:
            visitor = getattr(target, "visited_by", -1)
            target_visitors.append(visitor)

        row = agent_targets + target_visitors
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    # Update all agents on the situation, ask for a new course of action (or continue the current one)
    def update_agents(self):

        self.environment.set_round(self.num_rounds)
        print(f"Round {self.num_rounds}: comms_down={self.environment.comms_down}, sensing_down={self.environment.sensing_down}, wind={self.environment.wind}")

        # Record when the simulation started
        if self.sim_start is None:
            self.sim_start = time.perf_counter()

        # Decide who should or should not reason this round
        self.teammate_views = {i: self._observe_teammate_targets(i) for i in range(self.num_agents) if not self.agents[i].inaction}
        if PLANNER_MODE == "hybrid" and self.scenario == 1:
            # Each drone folds this round's messages and observations into its own beliefs (no LLM here)
            for i in self.teammate_views:
                self.agents[i].hybrid.observe(self._planner_scene(i), self.environment.target_locations)
        to_prompt = [idx for idx in range(self.num_agents) if self.should_prompt(self.agents[idx])]
        print(f"Round {self.num_rounds}: prompting agents {to_prompt}")

        for idx in to_prompt:
            self.decide(idx)

        for agent in self.agents:
            t = agent.target_index
            if isinstance(t, int) and t >= 0:
                target = self.environment.target_locations[t]
                if target.cleared or (target.target_type == TargetType.SEQUENTIAL and agent.agent_index in target.visitors):
                    agent.target_index = None

        for idx in range(self.num_agents):
            if not self.agents[idx].inaction:
                self.agents[idx].last_target = self.agents[idx].target_index

        for agent in self.agents:
            self.update_velocity(agent)
            agent.step()

        self.environment.check_visits(self.agents)
        self.log_round()
        self.num_rounds += 1
        self.sim_end = time.perf_counter()

    # Choose this drone's action: HTN plan if enabled and within budget, otherwise the LLM
    def decide(self, agent_index: int):
        agent = self.agents[agent_index]
        update_message = self.generate_update_message(agent_index)  # always built and printed

        if PLANNER_MODE == "hybrid" and self.scenario == 1:
            self._decide_hybrid(agent_index)
            return

        t0 = time.perf_counter()
        if PLANNER_MODE == "htn_llm" and self.scenario == 1 and not agent.delegated:
            try:
                kind, target = agent.htn.plan(agent_index,
                                              self.num_agents,
                                              self.environment.target_locations,
                                              self._planner_scene(agent_index))
                agent.plan_times.append(time.perf_counter() - t0)
                action = f"FLY {target}" if kind == "FLY" else kind
                print(f"[HTN] drone {agent_index}: {action}")
                if getattr(agent.htn, "last_tree", ""):
                    print(agent.htn.last_tree)
                reply = {"thought": "HTN plan",
                         "message": f"Executing {action}.",
                         "action": action}
                # Record the exchange so the LLM has context if this drone is delegated later
                if HISTORY_TURNS > 0:
                    agent.recent += [
                        {"role": "user", "content": update_message},
                        {"role": "assistant", "content": json.dumps({"action": action,
                                                                     "message": reply["message"]})},
                    ]
                    agent.recent = agent.recent[-2 * HISTORY_TURNS:]
                self.parse_response(json.dumps(reply), agent_index)
                return
            except (PlanningBudgetExceeded, PlanningFailed) as e:
                agent.plan_times.append(time.perf_counter() - t0)
                agent.fallbacks += 1
                print(f"[HTN] drone {agent_index}: delegating to LLM ({e})")

        self.parse_response(agent.generate_response(update_message), agent_index)


    # Hybrid decision: beliefs (messages, observation, LLM) -> A* plan -> action + codebook message.
    # The LLM never chooses the action; if the A* search runs out of budget a greedy rule does.
    def _decide_hybrid(self, agent_index: int):
        agent = self.agents[agent_index]
        targets = self.environment.target_locations
        scene = self._planner_scene(agent_index)

        def ask(system_prompt, user_prompt):
            reply = agent.query_json(system_prompt, user_prompt, key="beliefs")
            if HYBRID_PRINT_BELIEFS:
                raw = agent.last_raw
                if raw.get("reasoning"):
                    print(f"[BELIEF QUERY] drone {agent_index} reasoning:\n{raw['reasoning']}")
                print(f"[BELIEF QUERY] drone {agent_index} raw output:\n{raw.get('content')}")
                print(f"[BELIEF QUERY] drone {agent_index} parsed: {self._format_beliefs(agent_index, reply)}")
            return reply

        try:
            kind, target = agent.hybrid.plan(agent_index, self.num_agents, targets, scene, ask)
            note = "hybrid plan"
        except (PlanningBudgetExceeded, PlanningFailed) as e:
            agent.fallbacks += 1
            kind, target = agent.hybrid.fallback(agent_index, targets, scene)
            note = f"greedy fallback ({e})"
        agent.plan_times.append(agent.hybrid.last_plan_s)

        action = f"FLY {target}" if kind == "FLY" else kind
        message = agent.hybrid.message_for(kind, target, targets, scene)
        print(f"[HYBRID] drone {agent_index}: {action} | message: {message!r} | beliefs: "
              f"{ {j: (b.target, b.source) for j, b in agent.hybrid.last_used.items()} }")
        if getattr(agent.htn, "last_tree", ""):
            print(agent.htn.last_tree)
        self._score_beliefs(agent_index)
        self.parse_response(json.dumps({"thought": note, "message": message, "action": action}), agent_index)

    # Render a belief query's parsed reply with each belief checked against what the teammate is actually
    # doing (printing only; ground truth never reaches a planner). Green = correct, red = wrong (with the
    # actual target), yellow = not a legal belief (unknown or inactive teammate, bad value).
    def _format_beliefs(self, agent_index: int, reply):
        beliefs = reply.get("beliefs") if isinstance(reply, dict) else None
        if not isinstance(beliefs, dict):
            return str(reply)
        targets = self.environment.target_locations
        colors = HYBRID_COLOR_BELIEFS
        paint = (lambda code, text: f"\033[{code}m{text}\033[0m") if colors else (lambda code, text: text)

        def live(t):
            return t if isinstance(t, int) and 0 <= t < len(targets) and not targets[t].cleared else None

        parts = []
        for key, val in beliefs.items():
            entry = f"{json.dumps(str(key))}: {json.dumps(val)}"
            try:
                j = int(key)
                mate = self.agents[j] if 0 <= j < self.num_agents and j != agent_index else None
            except (TypeError, ValueError):
                mate = None
            if mate is None or mate.inaction:
                parts.append(paint("33", entry + " (invalid)"))
                continue
            if val is None or (isinstance(val, str) and val.strip().lower() in ("idle", "none", "null")):
                claimed = None
            elif isinstance(val, int) and not isinstance(val, bool) and 0 <= val < len(targets):
                claimed = live(val)
            else:
                parts.append(paint("33", entry + " (invalid)"))
                continue
            actual = live(mate.last_target)
            if claimed == actual:
                parts.append(paint("32", entry))
            else:
                parts.append(paint("31", f"{entry} (actual: {'idle' if actual is None else actual})"))
        reason = json.dumps(reply.get("reason", ""))
        return "{" + f'"reason": {reason}, "beliefs": {{' + ", ".join(parts) + "}}"

    # Compare the beliefs a drone just planned with against what its teammates are actually doing.
    # Ground truth is read here for logging only and never reaches a planner.
    def _score_beliefs(self, agent_index: int):
        agent = self.agents[agent_index]
        targets = self.environment.target_locations

        def live(t):
            return t if isinstance(t, int) and 0 <= t < len(targets) and not targets[t].cleared else None

        for j, belief in agent.hybrid.last_used.items():
            hit = live(belief.target) == live(self.agents[j].last_target)
            stat = agent.belief_hits.setdefault(belief.source, [0, 0])
            stat[0] += int(hit)
            stat[1] += 1

    # Interpret the response, search for which target the agent wants to head toward
    def parse_response(self, response: str, agent_index: int):
        if response is None:
            print("WARNING: Response is None!")
            return
        print(response)
        print()

        agent = self.agents[agent_index]
        data = extract_json_object(response)
        if data is None:
            print(f"Invalid JSON response (tail): {response[-300:]}")
            return

        #IDLE: Set velocity to zero.
        #FLY X: Set velocity to fly to target X
        #RECALL: Set inaction to True, set velocity to go back to start.
        action = str(data.get("action", ""))
        if action == "IDLE":
            agent.target_index = None
        elif action == "RECALL":
            agent.target_index = "Home"
            agent.inaction = True
        elif action.startswith("FLY "):
            try:
                index = int(action.removeprefix("FLY "))
                if 0 <= index < len(self.environment.target_locations):
                    t = self.environment.target_locations[index]
                    if t.cleared or (t.target_type == TargetType.SEQUENTIAL and agent_index in t.visitors):
                        print(f"Agent {agent_index} chose unavailable target {index}; idling.")
                        agent.target_index = None
                    else:
                        agent.target_index = index
                else:
                    print(f"Agent {agent_index} gave out-of-range FLY index: {index}")
            except ValueError:
                print(f"Agent {agent_index} gave malformed FLY action: {action!r}")
        else:
            print(f"Agent {agent_index} gave unrecognized action: {action!r}")

        if self.environment.comms_down or random.random() >= agent.comms_quality:
            agent.displayed_message = "Communication Failure."
        else:
            agent.displayed_message = data.get("message", "")

    # Set velocity to go towards the target.
    # TODO: change velocity based on how much time the agent spent thinking?
    def update_velocity(self, agent):
        agent.velocity = compute_velocity(agent, self.environment, getattr(self.environment, "wind", (0.0, 0.0)))

    # Generate the message updating the agent on the status of the mission
    def generate_update_message(self, agent_index: int):
        agent = self.agents[agent_index]
        update_message = ""

        print(f"Message sent to Agent {agent_index}:")

        match self.scenario:
            case 0:
                target_loc = self.environment.target_locations[0].get_location()
                update_message = (
                    f"YOUR LOCATION: X={agent.location[0]:.1f}m, Y={agent.location[1]:.1f}m.\n"
                    f"TARGET LOCATION: X={target_loc[0]:.1f}m, Y={target_loc[1]:.1f}m.\n"
                )
            case 1:
                mates = ", ".join(str(j) for j in range(self.num_agents) if j != agent_index)
                if True:#PROMPT_STYLE == "tos": TODO: The agents will sometimes waste tokens trying to find out what their agent number is, maybe this will fix it
                    update_message += f"You are **agent {agent_index}**; your teammates are agents {mates}.\n"
                if SHOW_OWN_LOCATION:
                    update_message += f"YOUR LOCATION: X={agent.location[0]:.1f}m, Y={agent.location[1]:.1f}m\n"

                # Which teammates can see each other
                views = self.teammate_views.get(agent_index) or self._observe_teammate_targets(agent_index)

                # Teammates
                for idx in range(len(self.agents)):
                    if idx == agent_index or self.agents[idx].inaction:
                        continue
                   
                    # Teammate direction
                    teammate = self.agents[idx]
                    teammate_location = teammate.location
                    seen = views[idx]
                    if seen == "blocked" or seen == -1 or seen == "sensor_failure":
                        teammate_target = "Unknown"
                    elif seen is None:
                        teammate_target = "None (idle)"
                    else:
                        teammate_target = self._target_name(seen, self.environment.target_locations[seen], agent_index)
                    update_message += f"TEAMMATE {idx} IS HEADING TO TARGET: {teammate_target}\n"

                    # Teammate message
                    if teammate.comms_quality > 0.0:
                        if self.environment.comms_down:
                            update_message += f"TEAMMATE {idx} SAYS: Communication Failure.\n"
                        elif (teammate.comms_distance == -1 or
                            geo.geometric_mean_3d(*teammate_location, *agent.location) <= teammate.comms_distance):
                            update_message += f"TEAMMATE {idx} SAYS: {teammate.displayed_message}\n"
                        else:
                            update_message += f"No message received from TEAMMATE {idx}\n"

                all_targets_visited = all(t.visited for t in self.environment.target_locations)
                if not all_targets_visited:
                    update_message += "\n"

                    # Theory of Scene would encourage a specific format
                    if PROMPT_STYLE == "tos":
                        slot_number = 0  # continues across tasks, starting at 0
                        for idx, target in enumerate(self.environment.target_locations):
                            if target.visited:
                               # update_message += f"TASK {idx} COMPLETE\n"
                               pass
                            else: # TODO: EXPERIMENTAL, update TOS prompting to account for unknown target types
                                update_message += f"TASK {idx} ({target.target_type.value}): INCOMPLETE\n"
                                if SHOW_TARGET_COORDS:
                                    distance = geo.point_distance(agent.location, target.get_location())
                                    update_message += (f"    Location: X={target.north_m:.1f}m, Y={target.east_m:.1f}m, "
                                                      f"{distance:.0f} meters away\n")
                                claimants = [a.agent_index for a in self.agents
                                             if a.last_target == idx and not a.inaction]
                                for i in range(target.required_drones):
                                    holder = f"Agent {claimants[i]}" if i < len(claimants) else "Unclaimed"
                                    update_message += f"    {holder}\n"
                                    #update_message += f"    Slot {self.number_to_letters(slot_number)}: {holder}\n"
                                    slot_number += 1

                    # If Theory of Scene is not being used, use a more generic format
                    else:
                        update_message += "The following targets have yet to be visited:\n"
                        for idx, target in enumerate(self.environment.target_locations):
                            if target.visited:
                                pass
                                #update_message += f"Target {idx} has been cleared, choose another target.\n"

                            elif not (target.target_type == TargetType.SEQUENTIAL and agent_index in target.visitors):
                                update_message += f"Target {idx} ({self._type_text(target, agent_index)}): INCOMPLETE\n"
                                if SHOW_TARGET_COORDS:
                                    distance = geo.point_distance(agent.location, target.get_location())
                                    update_message += (f"    Location: X={target.north_m:.1f}m, Y={target.east_m:.1f}m, "
                                                       f"{distance:.0f} meters away\n")

                else:
                    update_message += (
                        "\nAll targets have been visited. The mission is not complete until every drone is back at "
                        "its start location, and IDLE does not move you there. You should now RECALL.\n"
                    )

        print(update_message)
        print("End AI prompt\n")
        return update_message

    # Utility for certain styles of prompt
    def number_to_letters(self, n: int) -> str:
        if n < 0:
            raise ValueError("n must be non-negative")
        result = ""
        n += 1  # shift to 1-indexed for bijective base-26
        while n > 0:
            n, remainder = divmod(n - 1, 26)
            result = chr(ord('A') + remainder) + result
        return result


    # Return an array of arrays, representing the location of all agents.
    def get_agent_locations(self):
        locations = []
        for agent in self.agents:
            locations.append(agent.location)
        return locations

    # Check if the agents all decided to go home and do nothing
    def check_stalled(self):
        return all(agent.inaction for agent in self.agents)

    # Decide whether or not the agent should reason this round
    def should_prompt(self, agent) -> bool:
        # Recalled/inactive agents are never prompted
        if agent.inaction or agent.target_index == "Home":
            return False

        # The agent is doing nothing: never assigned (-1) or chose IDLE (None)
        # Always prompt idle agents
        if agent.target_index is None or agent.target_index == -1:
            return True

        target = self.environment.target_locations[agent.target_index]

        # Arrived at its intended location, needs a new target
        if geo.point_distance(agent.location, target.get_location()) < VISIT_RADIUS_METERS:
            return True

        if target.cleared:
            return True

        # Target no longer needs this drone (already cleared, or too many drones assigned)
        return self._is_surplus(agent, target)

    # Check if there is no need for the given agent to go to the given target
    # Mostly important for if there are too many drones going to one target.
    def _is_surplus(self, agent, target) -> bool:
        me = agent.agent_index
        # A drone knows its own visit history
        if target.target_type == TargetType.SEQUENTIAL and me in target.visitors:
            return True

        # Slots come from the type only if the drone knows it; otherwise use the guess
        hybrid = PLANNER_MODE == "hybrid"
        known = type_is_known(target, me) or (hybrid and agent.hybrid.type_learned(agent.target_index))
        slots = target.required_drones if known else UNKNOWN_SLOT_GUESS

        # Rivals are teammates this drone believes are heading to the same target
        # (hybrid: beliefs built from messages, observation and LLM estimates)
        views = agent.hybrid.fresh_claims() if hybrid else self.teammate_views.get(me, {})
        rivals = [me] + [i for i, t in views.items()
                         if t == agent.target_index and not self.agents[i].inaction]
        if len(rivals) <= slots:
            return False

        # Teammate locations aren't visible to the drone, so the lowest indices keep the target
        rivals.sort()
        return me not in rivals[:slots]


    def _type_text(self, target, agent_index):
        return target.target_type.value if type_is_known(target, agent_index) else "type unknown"

    def _target_name(self, idx, target, agent_index):
        if type_is_known(target, agent_index):
            return f"{target.target_type.value} {idx}"
        return f"Target {idx}"

    def _observe_teammate_targets(self, agent_index):
        """What this drone can infer about each active teammate's target:
        a target index, None (idle), -1 (unknown), or "blocked" (no line of sight).
        Noise is applied here, once per round."""
        me = self.agents[agent_index]
        seen = {}
        for idx, mate in enumerate(self.agents):
            if idx == agent_index or mate.inaction:
                continue
            if self.environment.sensing_down:
                seen[idx] = "sensor_failure"
                continue
            if not self.environment.has_line_of_sight(me.location, mate.location):
                seen[idx] = "blocked"
                continue
            last = mate.last_target
            if random.random() < me.failed_inference_chance:
                last = random.randrange(len(self.environment.target_locations))
            seen[idx] = last
        return seen

    def _received_messages(self, agent_index):
        """What this drone hears from each active teammate: their displayed message, the failure
        notice during an outage, or None when out of range / silent. Same rules the prompt applies."""
        me = self.agents[agent_index]
        heard = {}
        for idx, mate in enumerate(self.agents):
            if idx == agent_index or mate.inaction:
                continue
            if mate.comms_quality <= 0.0:
                heard[idx] = None
            elif self.environment.comms_down:
                heard[idx] = "Communication Failure."
            elif (mate.comms_distance == -1 or
                  geo.geometric_mean_3d(*mate.location, *me.location) <= mate.comms_distance):
                heard[idx] = mate.displayed_message
            else:
                heard[idx] = None
        return heard

    def _planner_scene(self, agent_index):
        agent = self.agents[agent_index]
        views = self.teammate_views.get(agent_index) or self._observe_teammate_targets(agent_index)
        return {
            "location": tuple(agent.location),
            "velocity": agent.drone_velocity,
            "teammate_targets": views,   # same noisy, line-of-sight-filtered view the prompt shows
            "type_known": lambda t: type_is_known(t, agent_index),
            "messages": self._received_messages(agent_index),
            "round": self.num_rounds,
            "my_target": agent.last_target,
        }


# Statistical helper functions
    def _avg_stat(self, key):
        vals = [s[key] for a in self.agents for s in a.call_stats if s.get(key) is not None]
        return "" if not vals else f"{sum(vals)/len(vals):.3f}"

    def _std_stat(self, key):
        vals = [s[key] for a in self.agents for s in a.call_stats if s.get(key) is not None]
        return "" if len(vals) < 2 else f"{statistics.stdev(vals):.3f}"

    def _num_calls(self):
        return sum(len(a.call_stats) for a in self.agents)

    def _rate_stat(self, key):
        total = self._num_calls()
        if total == 0:
            return ""
        count = sum(1 for a in self.agents for s in a.call_stats if s.get(key))
        return f"{count / total:.4f}"

    def _avg_plan_s(self):
        vals = [t for a in self.agents for t in a.plan_times]
        return "" if not vals else f"{sum(vals) / len(vals):.6f}"

    # Fraction of planned-on teammate beliefs from this source that matched what the teammate was doing
    def _belief_acc(self, source):
        if PLANNER_MODE != "hybrid":
            return ""
        hits = sum(a.belief_hits.get(source, [0, 0])[0] for a in self.agents)
        total = sum(a.belief_hits.get(source, [0, 0])[1] for a in self.agents)
        return "" if not total else f"{hits / total:.4f}"

    def _fallback_rate(self):
        total = sum(len(a.plan_times) for a in self.agents)
        return "" if not total else f"{sum(a.fallbacks for a in self.agents) / total:.4f}"

    def _per_agent(self, attr):
        vals = [getattr(a, attr) for a in self.agents]
        return vals[0] if len(set(vals)) == 1 else "|".join(str(v) for v in vals)
