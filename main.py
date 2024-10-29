import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Initialize HuggingFaceEmbeddings with model name
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
    separators=["\n\n", "\n", " "]
)

# Prepare to store all document chunks
all_chunks = []

# Process each PDF file in the 'data' directory
data_folder = './data'
for filename in os.listdir(data_folder):
    if filename.endswith('.pdf'):
        file_path = os.path.join(data_folder, filename)
        
        # Load and split the document
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        chunks = text_splitter.split_documents(pages)
        
        # Add document chunks to the list
        all_chunks.extend(chunks)

# Generate embeddings and store chunks in Chroma
vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=embedding_function,
    persist_directory="./chroma_persist"  # Make sure this directory is writable
)


# Streamlit app interface
st.title("Structured RAG PDF Document Search")
st.write("Enter your query below to retrieve relevant information from the PDF documents.")

# Accept user query
user_query = st.text_input("Enter your query:")

if user_query:
    # Generate query embedding
    query_embedding = embedding_function.embed_query(user_query)
    
    # Perform similarity search
    retrieved_docs = vectorstore.similarity_search_by_vector(query_embedding, k=2)
    
    # Display results in Streamlit
    st.write(f"### Retrieved Documents ({len(retrieved_docs)} results):")
    for idx, doc in enumerate(retrieved_docs, 1):
        st.write(f"**Document {idx}:**")
        st.write(doc.page_content)

# Define a simple structured prompt for enhanced querying
class QueryTemplate(ChatPromptTemplate):
    def __init__(self):
        super().__init__()
        self.template = """
            Context: Use the retrieved documents to summarize key points relevant to the query.
            Query: {query}
            Relevant Content:
            {context}
        """

    def format_prompt(self, query, context):
        return self.template.format(query=query, context=context)

# Add a Runnable class for formatting and displaying query responses
class QueryProcessor(RunnablePassthrough):
    def __init__(self):
        # Define a basic template for messages
        messages = [
            {"role": "system", "content": "You are an assistant skilled in retrieving key information from academic documents."},
            {"role": "user", "content": "Please summarize the key points."}
        ]
        super().__init__(messages=messages)

    def __call__(self, query, context):
        formatted_query = self.template.format_prompt(query, context)
        return formatted_query

# Initialize template and processor
query_template = QueryTemplate()
query_processor = QueryProcessor(query_template)

# Using query processor to display structured information
if user_query:
    for doc in retrieved_docs:
        processed_query = query_processor(user_query, doc.page_content)
        st.write(processed_query)
