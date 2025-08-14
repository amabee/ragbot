import argparse
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pathlib import Path

CHROMA_PATH = "chroma"
BOOKS_INDEX_PATH = "books_index"

# Improved prompt template
PROMPT_TEMPLATE = """
You are a friendly helper who loves stories and helping kids with their questions.

Here's some information from the book "{book_title}":
{context}

Question: {question}

Instructions:
- Answer using the information from "{book_title}" as much as possible
- Include quotes or parts of the story that help explain your answer
- If the story doesn't have the full answer, tell what you can figure out and what's missing
- Be playful, fun, and kind—make your answer feel like a story adventure!

Answer:
"""


def get_best_book_for_query(query, books_index_db, max_score_threshold=1.2):
    """
    Find the single best book that matches the query based on similarity scores
    Note: Lower scores are better (distance-based similarity)
    """
    book_results_with_scores = books_index_db.similarity_search_with_score(query, k=10)

    if not book_results_with_scores:
        print("❌ No books matched the query.")
        return None, float("inf")

    # Get the best match (lowest score = most similar)
    best_doc, best_score = book_results_with_scores[0]
    best_book = best_doc.metadata["book_name"]

    print(f"📖 Best matching book: '{best_book}' (score: {best_score:.3f})")

    # Show other potential matches for context
    print("📊 Other potential matches:")
    for doc, score in book_results_with_scores[1:4]:  # Show top 3 alternatives
        book_name = doc.metadata["book_name"]
        print(f"   - {book_name}: {score:.3f}")

    # Check if the best match is reasonable
    if best_score > max_score_threshold:
        print(
            f"⚠️  Best book match score ({best_score:.3f}) is above threshold ({max_score_threshold})"
        )
        print("💡 The query might be too vague or no relevant books found")
        return None, best_score

    return best_book, best_score


def query_single_book(
    query, book_name, per_book_dir, embeddings, k=10, max_chunk_threshold=0.8
):
    """
    Query a specific book's chunks
    Note: Lower scores are better (closer to 0 means more similar)
    """
    chroma_dir = per_book_dir / book_name
    if not chroma_dir.exists():
        print(f"❌ Book directory not found: {chroma_dir}")
        return []

    db = Chroma(persist_directory=str(chroma_dir), embedding_function=embeddings)

    # Get chunks with scores
    chunks_with_scores = db.similarity_search_with_score(query, k=k)

    # Filter by threshold and show what we're including
    relevant_chunks = []
    print(f"\n🔍 Chunk relevance scores for '{book_name}':")
    for i, (chunk, score) in enumerate(chunks_with_scores):
        status = "✅ INCLUDED" if score < max_chunk_threshold else "❌ EXCLUDED"
        print(f"   Chunk {i+1}: {score:.3f} {status}")
        if score < max_chunk_threshold:
            relevant_chunks.append(chunk)

    return relevant_chunks


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument(
        "--book_threshold",
        type=float,
        default=1.2,
        help="Maximum similarity score for book selection (lower scores = better matches, typical range 0.5-1.5)",
    )
    parser.add_argument(
        "--chunk_threshold",
        type=float,
        default=1.0,
        help="Maximum similarity score for chunk inclusion (lower scores = better matches, typical range 0.5-1.2)",
    )
    parser.add_argument(
        "--max_chunks", type=int, default=8, help="Maximum chunks to retrieve"
    )
    parser.add_argument(
        "--force_book", type=str, help="Force query against specific book name"
    )

    args = parser.parse_args()
    query_text = args.query_text
    print(f"🔍 Searching for: '{query_text}'")

    # Initialize model & embeddings
    model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    books_index_db = Chroma(
        persist_directory="books_index", embedding_function=embeddings
    )
    per_book_dir = Path("chroma_per_book")

    # Determine which book to query
    if args.force_book:
        target_book = args.force_book
        print(f"🎯 Forced to use book: '{target_book}'")
    else:
        target_book, book_score = get_best_book_for_query(
            query_text, books_index_db, args.book_threshold
        )

        if not target_book:
            print("❌ No suitable book found for the query.")
            return

    # Query the selected book
    relevant_chunks = query_single_book(
        query_text,
        target_book,
        per_book_dir,
        embeddings,
        k=args.max_chunks,
        max_chunk_threshold=args.chunk_threshold,
    )

    if not relevant_chunks:
        print(f"❌ No relevant chunks found in '{target_book}'.")
        return

    # Prepare context for the LLM
    context_text = "\n\n---\n\n".join([chunk.page_content for chunk in relevant_chunks])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(
        context=context_text, question=query_text, book_title=target_book
    )
    response = model.invoke(prompt)

    print(f"\n💬 Response (from '{target_book}'):")
    print(response.content)
    print(f"\n📚 Source: {target_book}")
    print(f"📄 Used {len(relevant_chunks)} chunks from this book")


if __name__ == "__main__":
    main()
