## Response format
Thought: <your reasoning>
Action: <tool_name>
Action Input: <input to the tool — must not be empty>

OR when done:

Thought: I have enough information.
Final Answer: <your answer>

Rules:
- Always start with Thought.
- One tool per step.
- Never repeat the same Action + Input.
- Never call a tool with an empty Action Input.
- Always output "Final Answer:" when done.
- When in doubt, output a Final Answer asking the user for clarification.
