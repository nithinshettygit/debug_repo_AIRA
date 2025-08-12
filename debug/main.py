# main.py (interactive with context, truncation, and logging)
from langchain_core.messages import HumanMessage, AIMessage
from graph import build_graph
from datetime import datetime

STOP_PHRASES = {"ok thank you", "end of topic", "exit", "quit"}
MAX_HISTORY_MESSAGES = 20  # keep last N messages (Human+AI combined)
LOG_FILE = "conversation.log"

def log_entry(user_text, agent_text):
    ts = datetime.utcnow().isoformat() + "Z"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{ts}\nUSER: {user_text}\nAGENT: {agent_text}\n{'-'*80}\n")

def truncate_history(messages, max_items=MAX_HISTORY_MESSAGES):
    """Keep only the most recent max_items messages."""
    if len(messages) <= max_items:
        return messages
    return messages[-max_items:]

def main():
    app = build_graph()
    print("Interactive agent started. Type a question (or 'exit', 'ok thank you', 'end of topic' to stop).")

    # conversation history as a list of message objects (HumanMessage / AIMessage)
    history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in STOP_PHRASES:
            print("Agent: Thank you — ending the session. 🙏")
            break

        # append human message to history
        history.append(HumanMessage(content=user_input))

        # truncate history to prevent runaway context length
        history = truncate_history(history, MAX_HISTORY_MESSAGES)

        # invoke the graph with current conversation history
        try:
            state = {"messages": history}
            result = app.invoke(state)

            # extract agent response
            agent_msg = None
            if result and "messages" in result and len(result["messages"]) > 0:
                # result['messages'] should contain AIMessage(s)
                agent_obj = result["messages"][-1]
                agent_text = getattr(agent_obj, "content", str(agent_obj))
                agent_msg = AIMessage(content=agent_text)
                print("\nAgent:", agent_text)
            else:
                agent_text = "(no response)"
                print("\nAgent: (no response received)")

            # append agent message to history and log
            if agent_msg:
                history.append(agent_msg)
                history = truncate_history(history, MAX_HISTORY_MESSAGES)
            log_entry(user_input, agent_text)

        except Exception as e:
            print(f"\nAn error occurred while invoking the graph: {e}")

if __name__ == "__main__":
    main()
