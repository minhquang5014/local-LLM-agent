"""
Entry point for the Local LLM Agent.

Usage:
    # Interactive REPL
    python main.py

    # Single task
    python main.py "Search for the latest news about Apple Silicon and summarise it"

    # Test inference only (no tools)
    python main.py --test-inference
"""

import sys
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    console = Console()
    def _print_answer(answer: str):
        console.print(Panel(Markdown(answer), title="Agent Answer", border_style="green"))
    def _print_info(msg: str):
        console.print(f"[bold cyan]{msg}[/bold cyan]")
    def _print_error(msg: str):
        console.print(f"[bold red]{msg}[/bold red]")
except ImportError:
    def _print_answer(answer: str): print(f"\n=== Answer ===\n{answer}\n")
    def _print_info(msg: str): print(msg)
    def _print_error(msg: str): print(f"ERROR: {msg}")


def run_interactive():
    _print_info("Local LLM Agent — type 'exit' or 'quit' to stop, 'clear' to reset memory.\n")
    from src.agent import run_agent
    while True:
        try:
            task = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            _print_info("\nBye.")
            break
        if not task:
            continue
        if task.lower() in ("exit", "quit"):
            _print_info("Bye.")
            break
        if task.lower() == "clear":
            from src.memory import MemoryStore
            MemoryStore().clear()
            _print_info("Memory cleared.")
            continue
        _print_info("Thinking…")
        try:
            answer = run_agent(task)
            _print_answer(answer)
        except Exception as e:
            _print_error(f"Agent error: {e}")


def run_once(task: str):
    from src.agent import run_agent
    _print_info(f"Task: {task}\n")
    answer = run_agent(task)
    _print_answer(answer)


def run_inference_test():
    _print_info("Testing inference backend…")
    from src.inference import generate
    prompt = "What is 2 + 2? Answer in one sentence."
    _print_info(f"Prompt: {prompt}")
    result = generate(prompt)
    _print_answer(result)


def main():
    parser = argparse.ArgumentParser(description="Local LLM Agent")
    parser.add_argument("task", nargs="?", help="Task to run (omit for interactive mode)")
    parser.add_argument("--test-inference", action="store_true", help="Run a quick inference test")
    args = parser.parse_args()

    if args.test_inference:
        run_inference_test()
    elif args.task:
        run_once(args.task)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
