import os

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.messages import SystemMessage, HumanMessage, ToolMessage
from langsmith import traceable


MAX_ITERATIONS = 10
MODEL ="qwen3:1.7b"

load_dotenv()


@tool
def get_product_price(product:str)->float:
    """look up the price of a product in catalog"""
    price_catalog = {
        "laptop": 1299.99,
       "smartphone": 799.99,
       "headphones": 199.99,
       "keyboard": 99.99,
    }
    return price_catalog.get(product.lower(), 0.0)

@tool
def apply_discount(price:float, discount_tier:str)->float:
    """apply a discount to a given price based on the discount tier"""
    discount_percentages = {
        "gold": 23,
        "silver": 10,
        "bronze": 20
    }
    discount_percentage = discount_percentages.get(discount_tier.lower(), 0)
    return round(price * (1 - discount_percentage / 100), 2)



@traceable(name="Langchain agent loop")
def run_agent(question:str):

    llm = init_chat_model(f"ollama:{MODEL}",temperature=0)
    tools = [get_product_price, apply_discount]

    tool_dict = {tool.name: tool for tool in tools}

    llm_with_tools = llm.bind_tools(tools)
    messages =[
                SystemMessage(content=
                              "You are a helpful assistant that can answer questions about product prices and apply discounts."
                              "Strict rules"
                                "1. dont assume product prices or discounts, always use the tools provided to you to get the correct information."
                                "2. first get the product price using the get_product_price tool, then apply the discount using the apply_discount tool."
                                "3. if user not specify the product ask him to specify the product name."
    
                              ),
                HumanMessage(content=question)
            ]

    for i in range(1,MAX_ITERATIONS+1):
        print(f"---Iteration {i}:----")
     
        llm_response = llm_with_tools.invoke(messages)
        tool_calls = llm_response.tool_calls

        if not tool_calls:
            print(f"Agent response: {llm_response.content}")
            return llm_response.content
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args")
        tool_id = tool_call.get("id")

        print(f"[Tool selected]: {tool_name} with args: {tool_args}")
    
        tool_to_use = tool_dict.get(tool_name)
        if not tool_to_use:
            print(f"Tool {tool_name} not found.")
            raise ValueError(f"Tool {tool_name} not found.")
        tool_result = tool_to_use.invoke(tool_args)
        print(f"[Tool executed]: {tool_result}")
        messages.append(llm_response)
        messages.append(ToolMessage(content=tool_result,tool_call_id=tool_id))

if __name__ == "__main__":
    run_agent("What is the price of a laptop and apply gold discount?")