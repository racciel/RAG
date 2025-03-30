from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
import torch
import warnings

#warnings.filterwarnings("ignore", category=UserWarning, module="langsmith.client")

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ["LANGCHAIN_TRACING_V2"] = "null"

MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"  #"ssmadha/gpt2-finetuned-scientific-articles" #"Qwen/Qwen2.5-Coder-1.5B-Instruct" # #"microsoft/phi-2" #"MaziyarPanahi/SciPhi-Self-RAG-Mistral-7B-32k-Mistral-7B-Instruct-v0.2-slerp" #"muntasir2179/TinyLlama-1.1B-rag-finetuned-v1.0"  #"TinyLlama/TinyLlama-1.1B-Chat-v1.0" #"unsloth/gemma-2b-it"#"microsoft/phi-2" #"MaziyarPanahi/SciPhi-Self-RAG-Mistral-7B-32k-Mistral-7B-Instruct-v0.2-slerp" #"Intel/dynamic_tinybert"
MAX_INPUT_LENGTH = 1024
CHUNK_SIZE = 1024
MAX_NEW_TOKENS = 150

st.title("RAG Chatbot")
docs_dir = "./documents"
os.makedirs(docs_dir, exist_ok=True)

def truncate_input(text, tokenizer, max_length=MAX_INPUT_LENGTH):
    tokens = tokenizer(text, max_length=max_length, truncation=True, return_tensors="pt")
    return tokens

def chunk_documents(documents, chunk_size=CHUNK_SIZE):
    chunks = []
    for doc in documents:
        text = doc.page_content
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
    return chunks

@st.cache_data
def load_documents():
    loader = DirectoryLoader(docs_dir)
    return loader.load()

@st.cache_resource
def initialize_index(_documents):
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(_documents, embed_model)
    return vector_store

documents = load_documents()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
all_splits = text_splitter.split_documents(documents)
vector_store = initialize_index(all_splits)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", is_decoder=True)

hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    return_full_text=False,
    pad_token_id=tokenizer.eos_token_id
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3},
    #search_type="similarity_score_threshold", 
    #search_kwargs={"score_threshold": 0.7}
)

prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant answering user's questions using this {context}."
     "If you can't find the answer to the question, just say that you don't know." 
     "Don't write nonsense which is irrelevant to the topic and the question!"
     "If your context is empty, just say that you cannot answer the question."
     #"Try to answer as short as possible and keep the answers concise."
     ),
    ("human", "\n\nQuestion: {question}\n\n"),
])

prompt_template = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    if not docs:
        return ""
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    } 
    | prompt_template 
    | llm 
    | StrOutputParser()
)

if "history" not in st.session_state:
    st.session_state.history = []

st.subheader("Conversation History")
for chat in st.session_state.history:
    print(chat)
    st.write(f"**You:** {chat['query']}")
    st.write(f"**Bot:** {chat['response']}")

st.subheader("Chat with Your Documents")
query = st.text_input("Ask a question:")

if query:
    try:
        response = qa_chain.invoke(query)
        st.session_state.history.append({"query": query, "response": response})
        st.write(response)
    except Exception as e:
        st.error(f"An error occurred: {e}")