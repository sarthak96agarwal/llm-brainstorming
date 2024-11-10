from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain import hub
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

import streamlit as st

from dotenv import load_dotenv
import os
load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

# Define your list of pages and descriptions
pages = {
    "TOP ELECUS": "This page provides the latest news about US Elections",
    "TOP MIDEAST": "This page provides news about middle east and israel hammas war",
    "TOP MTGE": "This page provides news about US mortgage market",
}

# Define a prompt template to ask the LLM to determine relevance between query and website descriptions
prompt_template = """
Given the following query and page description, please rate the relevance of the description to the query on a scale from 0 to 10:
Query: {query}
Description: {description}
Relevance (0-10):
"""

# Initialize the LLM chain with a prompt template
prompt = PromptTemplate(input_variables=["query", "description"], template=prompt_template)
llm = ChatOpenAI(model="gpt-4", temperature=0)
llm_chain = LLMChain(prompt=prompt, llm=llm)

# Function to match the query to the most relevant page description
def match_query_to_page(query):
    highest_relevance = -1
    best_match = None

    for page, description in pages.items():
        # Ask the LLM to score the relevance
        relevance_score = int(llm_chain.run({"query": query, "description": description}).strip())
        
        # If the score is higher than the previous best, update
        if relevance_score > highest_relevance:
            highest_relevance = relevance_score
            best_match = page

    # If no match is found with a high score, return None
    return best_match if highest_relevance >= 5 else None


api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)
tools=[wiki]
prompt = hub.pull("hwchase17/openai-functions-agent")
agent=create_openai_tools_agent(llm,tools,prompt)
agent_executor=AgentExecutor(agent=agent,tools=tools)


# Final function to handle user query and find website or search PDF
def handle_user_query(query, pdf_path=None):
    # First, try matching with website descriptions using the LLM
    page = match_query_to_page(query)
    
    if page:
        return f"The most relevant page for your query is: {page}"
    else:
        return agent_executor.invoke({"input":query})["output"]


# Streamlit App Layout
st.title("AI Query Agent")

# Input Query
query = st.text_input("Enter your query:")

# Button to process the query
if st.button("Submit"):
    if query:
        # Try matching with page descriptions using the LLM
        page = match_query_to_page(query)
        
        if page:
            st.write(f"The most relevant page for your query is: **{page}**")
        else:
            st.write("Could not find a matching TOP page. Searching on wiki")
            wiki_output = agent_executor.invoke({"input":query})["output"]
            st.write(f"Wikipedia returned: {wiki_output}")
    else:
        st.write("Please enter a query.")

# Display instructions or information
st.sidebar.header("About")
st.sidebar.text("This app uses an AI model to match your query with TOP page descriptions.")
st.sidebar.text("If no match is found, it can also search on wiki")


