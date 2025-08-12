# main.py
from langchain_core.messages import HumanMessage
from graph import build_graph
# importing config is not necessary here because graph/agents/router import it,
# but you can import it to ensure the env check happens early if you prefer:
# import config

if __name__ == "__main__":
    app = build_graph()

    queries = [
        "What is the process of photosynthesis?",
        "What is the square root of 144?",
        "Explain Newton's third law of motion.",
        "Who was the first person to walk on the moon?",
        "Tell me a fun fact about the solar system."
    ]

    for query in queries:
        print(f"\n--- User Query: {query} ---")
        try:
            result = app.invoke({"messages": [HumanMessage(content=query)]})
            # final state's messages list
            print(f"Agent Response: {result['messages'][-1].content}")
        except Exception as e:
            print(f"An error occurred: {e}")
