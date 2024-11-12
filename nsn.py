import PyPDF2
import os
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationChain
import tiktoken  # Tokenizer for OpenAI models

from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Function to read PDF content
def read_pdf(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
        return text

# Function to count tokens in a string using the OpenAI tokenizer
def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")  # OpenAI uses this encoding for GPT-3
    tokens = encoding.encode(text)
    return len(tokens)

# Function to limit the context length to stay within token limits
def limit_context_length(context, max_tokens=3500):
    # If the context exceeds the maximum token limit, truncate it
    current_tokens = count_tokens(context)
    if current_tokens > max_tokens:
        # If it's too long, truncate the context to fit within the limit
        encoding = tiktoken.get_encoding("cl100k_base")
        truncated_context = encoding.decode(encoding.encode(context)[:max_tokens])
        return truncated_context
    return context

# Set up Langchain memory and model
def create_conversational_agent(pdf_text):
    # Initialize OpenAI model (replace with your preferred LLM or API key)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
    
    # Memory for the conversation
    memory = ConversationBufferMemory(memory_key="history")  # Use default memory key 'history'
    
    # Use the PDF text to create embeddings for a FAISS vector store
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # Store the text in a FAISS vector store for better searchability
    faiss_store = FAISS.from_texts([pdf_text], embeddings)
    
    # Initialize conversation chain with context passed to the prompt
    conversation_chain = ConversationChain(llm=llm, memory=memory)

    return conversation_chain, faiss_store

# Streamlit app function
def streamlit_app():
    st.title("Query a Story Demo")

    # Sidebar for PDF links
    st.sidebar.title("Available Stories")
    story_link = "https://www.goldmansachs.com/insights/articles/how-trumps-election-is-forecast-to-affect-us-stocks"
    st.sidebar.markdown(f"[How Trump’s election is forecast to affect US stocks]({story_link})", unsafe_allow_html=True)

    # Specify the path to your PDF file
    pdf_file_path = 'stories/trump.pdf'  # Adjust the file path here

    # Check if the file exists
    if not os.path.exists(pdf_file_path):
        st.error(f"File '{pdf_file_path}' not found.")
        return

    # Read PDF content
    pdf_text = read_pdf(pdf_file_path)

    # Create conversational agent
    conversation_chain, faiss_store = create_conversational_agent(pdf_text)

    # Streamlit chat-like UI
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render chat history in a scrollable container
    chat_history = st.container()

    with chat_history:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(
                    f'<div style="background-color:#DCF8C6; padding:10px 15px; border-radius:20px; margin-bottom:5px; max-width:70%; word-wrap:break-word;"><b>You:</b> {message["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="background-color:#f1f0f0; padding:10px 15px; border-radius:20px; margin-bottom:5px; max-width:70%; word-wrap:break-word;"><b>Agent:</b> {message["content"]}</div>',
                    unsafe_allow_html=True
                )

    # Make the input box always at the bottom of the screen
    st.markdown(
        """
        <style>
        .chat-input-container {
            position: fixed;
            bottom: 0;
            width: 100%;
            padding: 10px;
            background-color: white;
            z-index: 100;
        }
        .chat-input-container input {
            width: 100%;
            padding: 15px;
            font-size: 16px;
            border-radius: 10px;
            border: 1px solid #ccc;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Render the input box in a container at the bottom of the page
    with st.container():
        user_input = st.text_input("Type your message:", key="input", label_visibility="collapsed", placeholder="Ask a question about the Story...")

    if user_input:
        # Store the user's message immediately (for synchronous display)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Use FAISS store to retrieve relevant context from PDF based on the user's input
        relevant_docs = faiss_store.similarity_search(user_input, k=3)
        context = " ".join([doc.page_content for doc in relevant_docs])

        # Limit the context length to avoid exceeding token limits
        context = limit_context_length(context, max_tokens=3500)

        # Create a prompt with the relevant context from the PDF and the user input
        prompt = f"Context: {context}\n\nUser asked: {user_input}\nAnswer:"

        # Use the conversation chain to generate the response, now based on the context of the PDF
        response = conversation_chain.run(input=prompt)  # This uses the memory and chain automatically

        # Store the agent's response immediately (for synchronous display)
        st.session_state.messages.append({"role": "agent", "content": response})

        # Render both the user input and the agent's response
        st.markdown(
            f'<div style="background-color:#DCF8C6; padding:10px 15px; border-radius:20px; margin-bottom:5px; max-width:70%; word-wrap:break-word;"><b>You:</b> {user_input}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div style="background-color:#f1f0f0; padding:10px 15px; border-radius:20px; margin-bottom:5px; max-width:70%; word-wrap:break-word;"><b>Agent:</b> {response}</div>',
            unsafe_allow_html=True
        )

# Run the Streamlit app
streamlit_app()
