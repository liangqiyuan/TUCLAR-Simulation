"""Patch agent.py and hybrid_planner.py so belief queries get their own response schema.

Why: agent.py's _create forces every LLM reply into the action schema (thought/action/message), so a belief
query can never return {"beliefs": ...}; the parser gets None and no LLM belief is ever used.

Run from the folder that holds agent.py and hybrid_planner.py:   python3 fix_belief_schema.py
Safe to run twice. A copy of each file is saved as <name>.pre_belief_fix.bak first.
"""
import shutil
import sys

AGENT = [
    # 1. schema for belief queries
    ('''@dataclass
class DroneParams:''', '''def build_belief_schema(teammates):
    """Schema for a belief query: one entry per silent teammate, each a target index or null (idle).
    Needed because _create otherwise forces every reply into the action schema, which has no room for beliefs."""
    return {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "maxLength": 300},
            "beliefs": {
                "type": "object",
                "properties": {str(j): {"anyOf": [{"type": "integer"}, {"type": "null"}]} for j in teammates},
                "required": [str(j) for j in teammates],
                "additionalProperties": False,
            },
        },
        "required": ["reason", "beliefs"],
        "additionalProperties": False,
    }

@dataclass
class DroneParams:'''),
    # 2. _create takes an optional schema
    ('''    def _create(self, messages):''',
     '''    def _create(self, messages, schema=None, schema_name="agent_action"):'''),
    ('''"json_schema": {"name": "agent_action", "schema": build_schema(), "strict": True},''',
     '''"json_schema": {"name": schema_name, "schema": schema or build_schema(), "strict": True},'''),
    # 3. query_json passes it through
    ('''    def query_json(self, system_prompt, message, key):
        started = time.perf_counter()
        try:
            reasoning, content = self._create([{"role": "system", "content": system_prompt},
                                               {"role": "user", "content": message}])''',
     '''    def query_json(self, system_prompt, message, key, schema=None):
        started = time.perf_counter()
        try:
            reasoning, content = self._create([{"role": "system", "content": system_prompt},
                                               {"role": "user", "content": message}],
                                              schema=schema, schema_name="belief_query")'''),
    # 4. the hybrid decision builds the schema from the silent teammates
    ('''        def ask(system_prompt, user_prompt):
            reply = agent.query_json(system_prompt, user_prompt, key="beliefs")''',
     '''        def ask(system_prompt, user_prompt, silent):
            reply = agent.query_json(system_prompt, user_prompt, key="beliefs", schema=build_belief_schema(silent))'''),
]

HYBRID = [
    ('''            reply = ask_llm(BELIEF_SYSTEM_PROMPT, self._belief_prompt(targets, scene, silent, rnd))''',
     '''            reply = ask_llm(BELIEF_SYSTEM_PROMPT, self._belief_prompt(targets, scene, silent, rnd), silent)'''),
    ('''        ask_llm(system_prompt, user_prompt) -> parsed JSON dict or None.''',
     '''        ask_llm(system_prompt, user_prompt, silent_teammates) -> parsed JSON dict or None. The list of silent
        teammates lets the caller constrain the reply's format to exactly those entries.'''),
]


def patch(path, edits):
    with open(path) as f:
        text = f.read()
    pending = []
    for old, new in edits:
        if new in text:
            continue                                  # already applied
        if text.count(old) != 1:
            sys.exit(f"{path}: expected text not found exactly once; nothing was changed. Text:\n{old[:120]}")
        pending.append((old, new))
    if not pending:
        print(f"{path}: already patched")
        return
    shutil.copy(path, path + ".pre_belief_fix.bak")
    for old, new in pending:
        text = text.replace(old, new)
    with open(path, "w") as f:
        f.write(text)
    print(f"{path}: patched ({len(pending)} edits)")


if __name__ == "__main__":
    patch("agent.py", AGENT)
    patch("hybrid_planner.py", HYBRID)
