import os
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

# --- Part 1: Configuration & Initialization ---
# Define the folder where the FAISS index is stored
VECTOR_DB_DIR = "data/faiss_vectorstore"
FAISS_INDEX_NAME = "faiss_index"
FAISS_INDEX_PATH = os.path.join(VECTOR_DB_DIR, FAISS_INDEX_NAME)
CHUNKS_FILE = "merged_chunks_with_figures.json"

# Initialize embeddings (Must match the model used for creation)
embedding_model_name = "thenlper/gte-large"
embedding_model_kwargs = {"device": "cuda"} # Change to "cpu" if you are not using a GPU
embedding_encode_kwargs = {"normalize_embeddings": True}

embeddings = HuggingFaceBgeEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=embedding_model_kwargs,
    encode_kwargs=embedding_encode_kwargs
)

# Initialize the Cross-Encoder for reranking
reranker_model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
reranker = CrossEncoder(reranker_model_name)

# --- Part 2: Loading Data & Vector Store ---
try:
    # Load the FAISS vector store
    vectorstore = FAISS.load_local(
        folder_path=FAISS_INDEX_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ FAISS vector store loaded successfully.")
except Exception as e:
    print(f"❌ Error loading FAISS vector store: {e}")
    print("Please ensure the directory and index files exist and the path is correct.")
    exit()

try:
    # Load original merged data for direct keyword search
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        original_chunks = json.load(f)
    print(f"✅ Original data loaded from {CHUNKS_FILE}.")
except FileNotFoundError:
    print(f"❌ Error: {CHUNKS_FILE} not found. Please ensure it's in the same directory.")
    exit()

# --- Part 3: Hybrid Search Function (Corrected) ---
def hybrid_search(query: str, vectorstore: FAISS, original_chunks: list, k_semantic=20) -> list[Document]:
    """
    Performs a hybrid search combining semantic search and a direct keyword search on original data.
    """
    
    # 1. Semantic Search (Vector Search)
    semantic_results = vectorstore.similarity_search(query, k=k_semantic)
    
    # 2. Direct Keyword Search on original data
    keyword_results = []
    query_lower = query.lower()
    
    for chunk in original_chunks:
        subchapter = chunk.get("subchapter", "").lower()
        content = chunk.get("content", "").lower()
        
        if query_lower in subchapter or query_lower in content:
            # Create a Document object from the raw chunk data
            doc = Document(page_content=chunk["content"], metadata=chunk)
            keyword_results.append(doc)
    
    # 3. Combine results and remove duplicates
    combined_results = semantic_results + keyword_results
    unique_docs = {doc.metadata.get("chunk_uuid"): doc for doc in combined_results}
    final_results = list(unique_docs.values())
            
    return final_results

# --- Part 4: Main Execution Block ---
# Define a sample query for testing
query_to_test = "13.1 MAGNETIC FIELD AND FIELD LINES"

k_final_results = 3  # The number of final, reranked results to display

print(f"\n🔍 Performing Hybrid Search for: '{query_to_test}'...")

# a) Perform the robust hybrid search
candidate_results = hybrid_search(query_to_test, vectorstore, original_chunks)

print(f"Found {len(candidate_results)} candidate documents for reranking.")

if len(candidate_results) == 0:
    print("No documents found. Please check your query or data.")
else:
    # b) Prepare documents for the reranker
    documents_to_rerank = [r.page_content for r in candidate_results]
    document_pairs = [[query_to_test, doc] for doc in documents_to_rerank]

    # c) Use the reranker to score the documents
    print("\n🔄 Reranking the results...")
    scores = reranker.predict(document_pairs)

    # d) Combine and sort the results by score
    combined_results = zip(scores, candidate_results)
    sorted_results = sorted(combined_results, key=lambda x: x[0], reverse=True)

    # 5. Print the top reranked results
    print(f"✅ Reranking complete. Displaying top {k_final_results} results.")
    for idx, (score, doc) in enumerate(sorted_results[:k_final_results], 1):
        print(f"\n--- Reranked Result {idx} (Score: {score:.4f}) ---")
        print(f"Subchapter: {doc.metadata.get('subchapter', 'N/A')}")
        content_snippet = doc.page_content.replace('\n', ' ').strip()
        print(f"Content (first 300 chars): {content_snippet[:300]}...")
        
        figures_str = doc.metadata.get("figures", "")
        if figures_str:
            print("Figures:")
            print(f" - {figures_str}")
        else:
            print("Figures: None")
