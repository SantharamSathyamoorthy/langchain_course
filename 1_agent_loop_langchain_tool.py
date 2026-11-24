from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage,ToolMessage
from langchain.tools import tool
from langchain import tracable


load_dotenv()

MAX_ITERATIONS = 10
MODEL = "llama3.2:latest"

# define a tool that can be used by the agent

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


# define the agent loop

@tracable
def run_agent(question:str):
    pass


if __name__ == "__main__":
    print("Hello langchain with .bind tools")
    question = "What is the price of a laptop with a gold discount?"
    run_agent(question)

