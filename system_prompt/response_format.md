## Response format
Thought: <one short sentence — what to do next>
Action: <tool_name>
Action Input: <input to the tool — must not be empty>

OR when done:

Thought: I have enough information.
Final Answer: <your answer>

Rules:
- Always start with Thought.
- Keep every Thought to ONE short sentence (max ~15 words). Do not restate the
  tool output, do not explain your plan, do not think step-by-step out loud.
- One tool per step.
- Never repeat the same Action + Input.
- Never call a tool with an empty Action Input.
- Always output "Final Answer:" when done.
- Put all detail in the Final Answer, not in the Thoughts.
- When in doubt, output a Final Answer asking the user for clarification.
