# router.py
from langchain_core.prompts import ChatPromptTemplate
from config import model, extract_text

router_prompt = ChatPromptTemplate.from_template(
    """
    You are a subject classifier. Your task is to analyze the user's query
    and determine which subject it belongs to.
    Your options are: "science", "maths", "physics", or "general".
    Respond with only one of these subject names.
    User query: {query}
    """
)

def router_node(state):
    """
    Node that runs in the graph. It must return a dict representing state.
    We just pass the state through unchanged here (identity node).
    The real classification happens in route_query (conditional function).
    """
    # Return a dict in the expected shape (preserve messages)
    return {"messages": state["messages"]}

def route_query(state):
    """
    Conditional-function used by add_conditional_edges to pick the next node.
    It must return one of: "science", "maths", "physics", "general".
    """
    query = state["messages"][-1].content
    prompt = router_prompt.format(query=query)
    response = model.generate_content(prompt)
    subject_text = extract_text(response).strip().lower()

    # simple matching (covers "math" and "maths")
    if "science" in subject_text:
        return "science"
    if "math" in subject_text:   # covers both "math" and "maths"
        return "maths"
    if "physics" in subject_text:
        return "physics"
    return "general"
