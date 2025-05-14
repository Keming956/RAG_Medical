import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Step 1: Document Loading
def load_documents(directory_path):
    """Load text documents from a directory"""
    try:
        loader = DirectoryLoader(directory_path, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents from {directory_path}")
        return documents
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []

# Step 2: Document Splitting
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into manageable chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

# Step 3: Create Embeddings and Vector Store
def create_vectorstore(chunks, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Create embeddings and FAISS vectorstore"""
    try:
        # Initialize the embedding model (CPU-friendly)
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create and save the FAISS index
        vectorstore = FAISS.from_documents(chunks, embeddings)
        print(f"Created FAISS vectorstore with {len(chunks)} documents")
        
        return vectorstore, embeddings
    except Exception as e:
        print(f"Error creating vectorstore: {e}")
        raise

# Step 4: Save and Load FAISS Index
def save_vectorstore(vectorstore, path="faiss_index"):
    """Save the FAISS vectorstore to disk"""
    try:
        vectorstore.save_local(path)
        print(f"Saved FAISS index to {path}")
    except Exception as e:
        print(f"Error saving vectorstore: {e}")

def load_vectorstore(path="faiss_index", embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Load the FAISS vectorstore from disk"""
    try:
        # Initialize the same embeddings used for creating the index
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load the index
        vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        print(f"Loaded FAISS index from {path}")
        
        return vectorstore, embeddings
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
        raise

# Step 5: Load CPU-friendly Language Model
def load_cpu_friendly_llm(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Load a small language model that can run on CPU"""
    try:
        print(f"Loading language model: {model_id}")
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype=torch.float32,  # Use float32 on CPU
            low_cpu_mem_usage=True
        )
        
        # Create text generation pipeline
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        # Wrap the pipeline in HuggingFacePipeline for LangChain
        llm = HuggingFacePipeline(pipeline=pipe)
        print("Language model loaded successfully")
        
        return llm
    except Exception as e:
        print(f"Error loading language model: {e}")
        raise

# Step 6: Optional Reranking Function
def add_reranking(retriever, model_name="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3):
    """Add reranking capability to an existing retriever using sentence-transformers"""
    try:
        from sentence_transformers import CrossEncoder
        
        print(f"Adding reranking with model: {model_name}")
        # Load cross-encoder model
        model = CrossEncoder(model_name)
        
        # Store original retriever method
        original_get_relevant_documents = retriever.get_relevant_documents
        
        # Define new method with reranking
        def reranked_get_relevant_documents(query):
            # Get original documents
            docs = original_get_relevant_documents(query)
            
            if not docs:
                return []
            
            # Create document-query pairs for scoring
            pairs = [(doc.page_content, query) for doc in docs]
            
            # Get scores from cross-encoder
            scores = model.predict(pairs)
            
            # Sort documents by score
            scored_docs = list(zip(docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_n documents
            return [doc for doc, score in scored_docs[:top_n]]
        
        # Replace the method
        retriever.get_relevant_documents = reranked_get_relevant_documents
        print("Reranking added successfully")
        
        return retriever
    except ImportError:
        print("Warning: sentence-transformers not installed. Reranking not added.")
        print("Install with: pip install sentence-transformers")
        return retriever
    except Exception as e:
        print(f"Error adding reranking: {e}")
        return retriever  # Return original retriever on error

# Step 7: Create QA Chain
def create_qa_chain(vectorstore, llm, use_reranking=False):
    """Create a question-answering chain using LCEL syntax"""
    try:
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 10 if use_reranking else 4}
        )
        
        # Add reranking if requested
        if use_reranking:
            retriever = add_reranking(retriever)
        
        # Create template for the prompt
        template = """Answer the question based on the following context:

        Context: {context}
        
        Question: {question}
        
        Answer: """

        prompt = ChatPromptTemplate.from_template(template)
        
        # Format the documents
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        
        # Create the chain using LCEL
        qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Function to preserve the source documents
        def qa_with_sources(query):
            if isinstance(query, dict):
                if "query" in query:
                    query_str = query["query"]
                else:
                    # Try to get the first value or convert to string
                    query_str = str(next(iter(query.values())))
            else:
                query_str = str(query)
                
            docs = retriever.get_relevant_documents(query_str)
            answer = qa_chain.invoke(query_str)
            return {"result": answer, "source_documents": docs}
        
        print("QA chain created successfully")
        return qa_with_sources  # Return a function that preserves docs
    except Exception as e:
        print(f"Error creating QA chain: {e}")
        raise

# Main function to run the entire RAG pipeline
def main(query, document_directory):
    index_path = "faiss_index"
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # CPU-friendly model
    use_reranking = True  # Set to False if you don't want reranking
    
    # Step 1: Check if FAISS index exists
    if os.path.exists(index_path):
        print(f"Using existing FAISS index at {index_path}")
        vectorstore, embeddings = load_vectorstore(index_path, embedding_model)
    else:
        # Load and process documents
        print(f"Creating new FAISS index from documents in {document_directory}")
        documents = load_documents(document_directory)
        chunks = split_documents(documents)
        vectorstore, embeddings = create_vectorstore(chunks, embedding_model)
        save_vectorstore(vectorstore, index_path)
    
    # Step 2: Load language model
    llm = load_cpu_friendly_llm(llm_model)
    
    # Step 3: Create QA chain
    qa_chain = create_qa_chain(vectorstore, llm, use_reranking)
    
    # Step 4: Run a sample query
    print(f"\nProcessing query: {query}")
    
    result = qa_chain({"query": query})
    
    print("\nAnswer:")
    print(result["result"])
    print("\nSources:")
    for i, doc in enumerate(result["source_documents"]):
        print(f"Source {i+1}:")
        print(f"  From: {doc.metadata.get('source', 'Unknown')}")
        print(f"  Content: {doc.page_content[:150]}...")
        print()

    with open('answer.txt', 'w') as f:
        f.write(f"{query}\n{result['result']}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--document_directory',
                      required=True,
                      help='path to the your documents')
    parser.add_argument('--query',
                      required=True,
                      help='yours query')

    args = parser.parse_args()
    document_directory = args.document_directory
    query = args.query

    main(query, document_directory)