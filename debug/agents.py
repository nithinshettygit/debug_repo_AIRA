# agents.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from config import model, extract_text

def _run_agent_prompt(prompt_template, query):
    prompt = ChatPromptTemplate.from_template(prompt_template)
    formatted = prompt.format(query=query)
    resp = model.generate_content(formatted)
    return extract_text(resp)

def science_agent(state):
    """Answers science-related questions. Must return dict (state)."""
    query = state["messages"][-1].content
    answer = _run_agent_prompt("You are a science expert. Answer the following question: {query}", query)
    return {"messages": [AIMessage(content=answer)]}

def maths_agent(state):
    """Solves math problems. Must return dict (state)."""
    query = state["messages"][-1].content
    answer = _run_agent_prompt("You are a math expert. Solve the following problem: {query}", query)
    return {"messages": [AIMessage(content=answer)]}

def physics_agent(state):
    """Answers physics-related questions. Must return dict (state)."""
    query = state["messages"][-1].content
    answer = _run_agent_prompt("You are a physics expert. Answer the following question: {query}", query)
    return {"messages": [AIMessage(content=answer)]}

def general_agent(state):
    """Provides general knowledge answers. Must return dict (state)."""
    query = state["messages"][-1].content
    answer = _run_agent_prompt("You are a general knowledge assistant. Answer the following query: {query}", query)
    return {"messages": [AIMessage(content=answer)]}
