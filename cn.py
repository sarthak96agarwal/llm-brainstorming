import streamlit as st
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
import faiss
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set page layout to wide mode
st.set_page_config(page_title="LLM-based News Query", layout="wide")

# Initialize OpenAI API key
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Setup LLM
llm = OpenAI(temperature=0)

# Create the prompt for company extraction
company_extraction_prompt = PromptTemplate(
    input_variables=["article_text"],
    template="""
    Analyze the following news article and extract the names of any companies or organizations mentioned in the text.

    Here's the article:
    {article_text}

    Please provide a list of company names (or organization names) mentioned in the article. If no companies are mentioned, respond with "No companies mentioned".
    """
)

# Create the prompt for earnings detection
earnings_detection_prompt = PromptTemplate(
    input_variables=["article_text"],
    template="""
    Analyze the following news article and determine if it mentions earnings. 
    Specifically, identify:
    1. Whether the company is scheduled to announce earnings in the near future.
    2. Whether the company has recently announced earnings.
    3. If no earnings information is found, indicate that.

    Here's the article:
    {article_text}

    Provide a response with one of the following options:
    - "Upcoming Earnings"
    - "Recent Earnings"
    - "No earnings information found"
    """
)

# Create LangChain LLMChains for both company extraction and earnings detection
company_extraction_chain = LLMChain(
    llm=llm, 
    prompt=company_extraction_prompt
)

earnings_detection_chain = LLMChain(
    llm=llm, 
    prompt=earnings_detection_prompt
)

# Function to encode a batch of queries into vector representations
embedding_model = OpenAIEmbeddings()

def encode_queries(queries: list):
    """
    Given a list of queries, returns a list of embeddings for all the queries.
    """
    embeddings = embedding_model.embed_documents(queries)  # Process all queries in batch
    return embeddings  # Returns a list of embeddings for each query

# Create FAISS index for vector search
def create_faiss_index(corpus_embeddings: list):
    dim = len(corpus_embeddings[0])  # Embedding dimension
    index = faiss.IndexFlatL2(dim)  # Using a simple L2 distance metric
    index.add(np.array(corpus_embeddings).astype(np.float32))  # Add embeddings to the index
    return index

# Storing articles and metadata
articles = [
    "Tesla has announced their upcoming earnings report for Q3 2024, expected to be released next month.",
    "Apple's earnings for Q2 2024 showed significant growth, exceeding analysts' expectations.",
    "Amazon reported strong sales growth for Q4 2024, with a focus on cloud computing profits.",
    "Microsoft recently announced a partnership with OpenAI to integrate GPT into their cloud services.",
    "Meta's quarterly earnings were up, driven by new advertising strategies and growth in the virtual reality space."
]  # This is an example. Replace with your actual article data.

# Generate embeddings for the articles
corpus_embeddings = encode_queries(articles)  # Batch encode all articles
index = create_faiss_index(corpus_embeddings)

# Metadata storage: Keeping a reference to the actual articles for retrieval
article_metadata = [{"text": article} for article in articles]

# Perform vector search to retrieve relevant articles from FAISS
def vector_search(query: str):
    query_embedding = encode_queries([query])  # Batch processing, even for a single query
    query_embedding = np.array(query_embedding[0]).astype(np.float32).reshape(1, -1)
    
    # Perform search for the most similar vectors (top_k results)
    distances, indices = index.search(query_embedding, k=1)
    
    # Return the indices of the most similar documents
    return indices.tolist(), distances.tolist()

# Function to extract companies using LLM
def extract_companies_with_llm(article_text: str):
    companies_response = company_extraction_chain.run({"article_text": article_text})
    if companies_response.lower() == "no companies mentioned":
        return []
    else:
        companies = [company.strip() for company in companies_response.split(",")]
        return companies

# Function to detect earnings status using LLM
def detect_earnings_with_llm(article_text: str):
    earnings_status = earnings_detection_chain.run({"article_text": article_text})
    return earnings_status

# Full pipeline to process the query
def process_query(query: str):
    # Step 1: Perform vector search to retrieve relevant articles
    indices, distances = vector_search(query)
    
    final_output = []
    for idx in indices[0]:  # FAISS returns a list of indices
        # Retrieve the article text from the metadata using the index
        article_text = article_metadata[idx]['text']
        
        # Step 2: Extract companies using LLM
        companies = extract_companies_with_llm(article_text)
        
        if companies:
            # Step 3: Detect earnings status using LLM
            earnings_status = detect_earnings_with_llm(article_text)
            
            # Final Result for this article
            final_output.append({
                "article": article_text,
                "companies": companies,
                "earnings_status": earnings_status
            })
    
    # Return the results
    return final_output

# Streamlit app layout
st.title("LLM-based News Query and Earnings Detection")

# Layout structure with two columns
col1, col2 = st.columns([1, 3])  # Left column for articles, right column for search box and results

# **Left Column**: Display the articles corpus
with col1:
    st.subheader("Articles Corpus")
    for idx, article in enumerate(articles):
        st.write(f"**Article {idx+1}:**")
        st.write(article)
        st.write("---")

# **Right Column**: For search input and results
with col2:
    # Search box at the top
    query = st.text_input("Enter your query:")

    # Suggested Queries Section Below the Search Box
    st.subheader("Suggested Queries:")

    suggested_queries = [
        "What is the latest news about Apple's earnings?",
        "Which companies are reporting earnings this week?",
        "Tell me about Tesla.",
        "What's going on with meta?",
        "Give me news about Amazon's stock performance and earnings."
    ]

    # Displaying suggested queries as clickable options
    for prompt in suggested_queries:
        if st.button(f"Run query: {prompt}"):
            with st.spinner(f"Processing: {prompt}..."):
                results = process_query(prompt)
                # Display the results for the selected prompt
                st.subheader("Results")
                if results:
                    for item in results:
                        st.subheader(f"Article")
                        st.write(item["article"])
                        
                        st.subheader("Companies Mentioned:")
                        st.write(", ".join(item["companies"]))
                        
                        st.subheader("Earnings Status:")
                        st.write(item["earnings_status"])
                else:
                    st.write("No relevant articles found.")

    # Section to Display Results Based on User Input
    if query:
        with st.spinner(f"Processing: {query}..."):
            results = process_query(query)
            if results:
                st.subheader("Results")
                for item in results:
                    st.subheader(f"Article")
                    st.write(item["article"])
                    
                    st.subheader("Companies Mentioned:")
                    st.write(", ".join(item["companies"]))
                    
                    st.subheader("Earnings Status:")
                    st.write(item["earnings_status"])
            else:
                st.write("No relevant articles found.")
