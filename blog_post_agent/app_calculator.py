import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq 
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, START, MessageGraph
from langgraph.graph.state import StateGraph
load_dotenv()

#Load environment variables
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
api_key = os.getenv("GROQ_API_KEY")

#graph state
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

##llm
model = ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192",streaming=True,temperature=0)

#create graph
def create_graph():
    """Create a graph to call agents with tool, to generate blog"""

    
    @tool
    def add(a: float, b: float):
        """Adds two numbers."""
        return a + b
    
    tool_node = ToolNode([add])
    model_with_tools = model.bind_tools([add])
    def call_model(state):
        return {"messages": [model_with_tools.invoke(state["messages"])]}
    
    def should_continue(state: State):
        if state["messages"][-1].tool_calls:
            return "tools"
        else:
            return END

    graph_workflow = StateGraph(State)
    graph_workflow.add_node("blog_title_agent", call_model)
    graph_workflow.add_edge(START, "blog_title_agent")
    graph_workflow.add_node("tools", tool_node)
    graph_workflow.add_edge("tools", "blog_title_agent")
    graph_workflow.add_conditional_edges("blog_title_agent", should_continue)

    agent = graph_workflow.compile()
    return agent
    
agent=create_graph()