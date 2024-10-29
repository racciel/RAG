import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

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
vectorstore = Chroma.from_documents(documents=all_chunks, embedding=embedding_function)

# Modify the query to be a single string when embedding
query = "What are the key points of Q-methodology?"

# Generate query embedding
query_embedding = embedding_function.embed_query(query)

# Perform similarity search
retrieved_docs = vectorstore.similarity_search_by_vector(query_embedding, k=2)

print(f"\nDOCUMENTS RETRIEVED: {len(retrieved_docs)}\n")

# Print retrieved documents' content
for doc in retrieved_docs:
    print(doc.page_content)
