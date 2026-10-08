from dotenv import load_dotenv
import os

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-120b",
    temperature=0.1
)

# Prompt Template
prompt = PromptTemplate(
    input_variables=[
        "match_situation",
        "batsman",
        "bowler"
    ],
    template="""
You are legendary cricket commentator.

Generate exciting cricket commentary in the style of Ravi Shastri.

Match Situation:
{match_situation}

Batsman on Strike:
{batsman}

Bowler Name:
{bowler}

Requirements:
- Keep commentary under 150 words
- Make it feel like live match commentary
- Add excitement and energy
- Use natural cricket commentary style
- Add famous Ravi Shastri-like expressions

"""
)

# User Inputs
match_situation = input("Enter Match Situation: ")
batsman = input("Enter Batsman Name: ")
bowler = input("Enter Bowler Name: ")

# Create Final Prompt
final_prompt = prompt.format(
    match_situation=match_situation,
    batsman=batsman,
    bowler=bowler
)

# Generate AI Response
response = llm.invoke(final_prompt)

# Print Output
print("\nAI Cricket Commentary:\n")
print(response.content)