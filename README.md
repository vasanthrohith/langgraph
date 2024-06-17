What is LangGraph?
LangGraph takes the capabilities of LangChain to the next level, allowing you to build AI coding agents that can seamlessly coordinate multiple chains or actors across various steps. Imagine orchestrating complex tasks with ease, treating Agent workflows as a cyclic Graph structure where each node is a powerful function or a LangChain Runnable object. 🌟

Key Features:
🔹 Nodes: Any function or LangChain Runnable object like a tool.
🔹 Edges: Defines the direction between nodes.
🔹 Stateful Graphs: Designed to manage and update state objects as it processes data through its nodes.

This is a simple example of how to evaluate the relevance of a document. If it is not relevant, we will call a web search agent to get the answers.