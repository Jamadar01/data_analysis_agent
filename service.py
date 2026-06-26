import io
import contextlib
import os
from dotenv import load_dotenv
load_dotenv()  
from openai import OpenAI
import pandas as pd
import json

from typing import Annotated,TypedDict
# from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
# from langchain_core.messages import ToolMessage
# from langgraph.graph import StateGraph,START,END 
from langgraph.prebuilt import create_react_agent
df=pd.read_csv("sample_employee_data.csv")

def run_code(code: str) -> str:
    """
    Executes the given Python code and returns the output as a string.

    Args:
        code (str): The Python code to execute.

    Returns:
        str: The output of the executed code.
    """
    
    try:
        output_buffer = io.StringIO()
        with contextlib.redirect_stdout(output_buffer):
            exec(code,{"pd":pd,"df":df})
        output = output_buffer.getvalue()
    except Exception as e:
        output = f"Error: {e}"
    return output

system_msg=f"""You are a data analysis assistant with access to a `run_code` tool that executes Python against a pandas DataFrame called `df` (already loaded).

            To answer the user's question:
            1. Call the `run_code` tool with Python code that computes the answer and prints it with print().
            2. Read the tool's output.
            3. Reply to the user with a clear, natural-language answer based on that output.
            4. If the code returns an error, call the tool again with corrected code.

            Use the existing `df` directly. If a column is a date stored as text, convert it with pd.to_datetime before comparing.

            DataFrame structure:

            Columns and types:
            {df.dtypes}

            First rows:
            {df.head().to_string()}
            """
model=ChatOpenAI(model="gpt-4o-mini")
graph=create_react_agent(model,[run_code],prompt=system_msg)


# model_with_tools=model.bind_tools([run_code])

# class State(TypedDict):
#     messages:Annotated[list,add_messages]

# def model_node(state):
#     response=model_with_tools.invoke(state["messages"])
#     return {"messages":[response]}

# def tool_node(state):
#     last=state["messages"][-1]
#     results=[]
#     for call in last.tool_calls:
#         code=call["args"]["code"]
#         output=run_code(code)
#         results.append(ToolMessage(content=output,tool_call_id=call["id"]))
#     return {"messages":results}

# def should_continue(state):
#     last = state["messages"][-1]
#     if last.tool_calls:
#         return "tools"        
#     return END                

# openai_api_key = os.getenv("OPENAI_API_KEY")
# client = OpenAI(api_key=openai_api_key)

# builder = StateGraph(State)
# builder.add_node("model", model_node)
# builder.add_node("tools", tool_node)
# builder.add_edge(START, "model")
# builder.add_conditional_edges("model", should_continue)
# builder.add_edge("tools", "model")
# graph = builder.compile()



def answer():
    user_msg=input("Ask to an agent:")
    result = graph.invoke({"messages": [("human", user_msg)]})
    return result["messages"][-1].content

print(answer())
# testing_set
# with open("golden_set.json") as f:
#     data = json.load(f)

# passes = 0
# for i in data:
#     result = answer(i["question"])
#     result_norm = result.lower().replace(",", "")
#     if all(s.lower() in result_norm for s in i["expected_contains"]):
#         passes += 1

# print("Accuracy:", passes / len(data))
        
