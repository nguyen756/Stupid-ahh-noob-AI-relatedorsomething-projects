#.\.venv\Scripts\activate

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os

load_dotenv()


@tool
def calculator(a: float, b: float) -> str:
    """Calculate the sum of two numbers."""
    print("calculator has been used")
    return f"sum of {a} and {b} is {a + b}"

@tool
def calculate_force(a: float, b: float) -> str:
    """Calculate the times of two numbers."""
    print("calculate_force  has been used")
    return f"times of {a} and {b} is {a * b}"

def main():
    groq_api_key = os.getenv("GROQ_API_KEY")
    model = ChatOpenAI( temperature=0,
    model="llama-3.1-8b-instant",
    openai_api_key=groq_api_key,
    openai_api_base="https://api.groq.com/openai/v1")
    tools=[calculator,calculate_force]
    agent_executer = create_react_agent(model, tools)
    print("Type 'exit' or 'quit' to end the conversation.")
    print("e la cuc no doi a: ")
    while True:
        user_input = input("\nyou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        print ("\nAssistant:",end="")
        #stream words by words 
        for chunk in agent_executer.stream(
            {"messages":[HumanMessage(content=user_input)]}

        ):
            if "agent" in chunk and "messages" in chunk["agent"]:
                for message in chunk["agent"]["messages"]:
                    print(message.content,end="")
        print()
if __name__ == "__main__":
    main()