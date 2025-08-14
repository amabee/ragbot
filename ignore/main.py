import argparse
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder
import numpy as np

CHROMA_PATH = "chroma"

# Improved prompt template
PROMPT_TEMPLATE = """
You are a knowledgeable assistant helping with questions about Alice's Adventures in Wonderland.

Context from the book:
{context}

Question: {question}

Instructions:
- Answer based primarily on the provided context
- Include relevant quotes when they help illustrate your answer
- If the context doesn't fully answer the question, say what you can determine and note what's missing
- Be engaging and capture the whimsical nature of the story when appropriate

Answer:
"""

def expand_query(query, model):
    """Generate alternative phrasings of the query for better search"""
    expansion_prompt = f"""
    Original question: "{query}"
    
    Generate 2 alternative ways to ask this question that might find different relevant information:
    1. Using different words with similar meaning
    2. A more specific version
    
    Return only the alternative questions, one per line.
    """
    
    try:
        response = model.invoke(expansion_prompt)
        alternatives = [line.strip() for line in response.content.strip().split('\n') if line.strip()]
        return [query] + alternatives[:2]  # Original + 2 alternatives
    except:
        return [query]  # Fallback to original query only

def multi_query_search(queries, db, k_per_query=5):
    """Search with multiple query variations and combine results"""
    all_results = []
    seen_content = set()
    
    for query in queries:
        results = db.similarity_search_with_relevance_scores(query, k=k_per_query)
        
        for doc, score in results:
            # Simple deduplication based on first 100 chars
            content_key = doc.page_content[:100]
            if content_key not in seen_content:
                seen_content.add(content_key)
                all_results.append((doc, score, query))  # Track which query found it
    
    # Sort by score and return top results
    all_results.sort(key=lambda x: x[1], reverse=True)
    return all_results

def rerank_results(query, documents, reranker_model=None, top_k=5):
    """Rerank documents using a cross-encoder model"""
    if not documents:
        return documents
        
    # Initialize reranker if not provided
    if reranker_model is None:
        print("🔄 Loading reranker model...")
        reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Prepare query-document pairs for reranking
    query_doc_pairs = []
    for doc, original_score in documents:
        query_doc_pairs.append([query, doc.page_content])
    
    # Get reranking scores
    print(f"📊 Reranking {len(query_doc_pairs)} documents...")
    rerank_scores = reranker_model.predict(query_doc_pairs)
    
    # Combine documents with new scores
    reranked_results = []
    for i, (doc, original_score) in enumerate(documents):
        rerank_score = float(rerank_scores[i])
        reranked_results.append((doc, rerank_score, original_score))
    
    # Sort by rerank score (higher is better)
    reranked_results.sort(key=lambda x: x[1], reverse=True)
    
    # Return top_k results in original format (doc, score)
    return [(doc, rerank_score) for doc, rerank_score, _ in reranked_results[:top_k]]

def debug_openai_config():
    """Debug OpenAI configuration"""
    print("🔧 Debugging OpenAI Configuration:")
    
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    print(f"API Key exists: {bool(api_key)}")
    if api_key:
        print(f"API Key starts with: {api_key[:10]}...")
        print(f"API Key length: {len(api_key)}")
    
    print(f"Base URL: {base_url}")
    
    # Check if .env file exists
    env_path = ".env"
    print(f".env file exists: {os.path.exists(env_path)}")
    
    if not api_key:
        print("❌ OPENAI_API_KEY not found!")
        print("Make sure your .env file contains:")
        print("OPENAI_API_KEY=your_key_here")
        return False
    
    return True


def main():
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")
    
    if not debug_openai_config():
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--k", type=int, default=3, help="Number of final results to return")
    parser.add_argument("--retrieve_k", type=int, default=10, help="Number of documents to retrieve before reranking")
    parser.add_argument("--threshold", type=float, default=0.25, help="Similarity threshold")
    parser.add_argument("--expand", action="store_true", help="Use query expansion")
    parser.add_argument("--rerank", action="store_true", help="Use reranking for better results")
    
    args = parser.parse_args()
    
    print(f"🔍 Searching for: '{args.query_text}'")
    
    # Use SAME embedding model as database creation
    embedding_function = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Initialize model for query expansion
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Initialize reranker if requested
    reranker = None
    if args.rerank:
        print("🚀 Initializing reranker...")
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Generate query variations if requested
    if args.expand:
        queries = expand_query(args.query_text, model)
        print(f"📝 Query variations: {queries}")
        results = multi_query_search(queries, db, k_per_query=args.retrieve_k//len(queries))
        results = [(doc, score) for doc, score, _ in results[:args.retrieve_k]]  # Remove query tracking
    else:
        # Get more results for reranking
        k_to_retrieve = args.retrieve_k if args.rerank else args.k
        results = db.similarity_search_with_relevance_scores(args.query_text, k=k_to_retrieve)
    
    if not results:
        print("❌ No results found")
        return
    
    print(f"📥 Retrieved {len(results)} initial results")
    
    # Apply reranking if requested
    if args.rerank and len(results) > 1:
        print(f"🎯 Reranking results...")
        results = rerank_results(args.query_text, results, reranker, top_k=args.k)
        print(f"✨ After reranking, top {len(results)} results:")
    else:
        # Just take top k if not reranking
        results = results[:args.k]
        print(f"📊 Top {len(results)} results (no reranking):")
    
    # Show results with scores
    for i, (doc, score) in enumerate(results, 1):
        score_type = "Rerank" if args.rerank else "Similarity"
        print(f"  {i}. {score_type} Score: {score:.4f}")
        print(f"     Preview: {doc.page_content[:80]}...")
        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
            print(f"     Source: {doc.metadata['source']}")
        print()
    
    # Apply threshold filter (if using similarity scores)
    if not args.rerank:
        good_results = [(doc, score) for doc, score in results if score >= args.threshold]
        if not good_results:
            print(f"❌ No results found above threshold {args.threshold}")
            if results:
                print(f"Best score was: {results[0][1]:.4f}")
            return
    else:
        # For reranked results, we typically don't use the same threshold
        good_results = results
    
    # Prepare context
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in good_results])
    
    # Generate response
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=args.query_text)

    print("🤖 Generating response...")
    response = model.invoke(prompt)
    response_text = response.content

    # Extract sources
    sources = [doc.metadata.get("source", "Unknown") for doc, _ in good_results]
    
    print(f"\n💬 Response:\n{response_text}")
    print(f"\n📚 Sources: {list(set(sources))}")

if __name__ == "__main__":
    main()
