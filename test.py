from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_PATH = "chroma"

# Test what's in the database
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# Test search with very loose criteria
query = "Alice"
results = db.similarity_search_with_relevance_scores(query, k=5)

print(f"Query: '{query}'")
print(f"Results found: {len(results)}")

if results:
    for i, (doc, score) in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Score: {score:.4f}")
        print(f"Content (first 150 chars): {doc.page_content[:150]}")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
else:
    print("No results found at all!")
    
    # Check if database has any documents
    try:
        # Get a few random documents to see if anything is there
        all_docs = db.similarity_search("the", k=5)  # "the" should match something
        print(f"Total documents when searching for 'the': {len(all_docs)}")
        if all_docs:
            print("Sample document:")
            print(all_docs[0].page_content[:200])
    except Exception as e:
        print(f"Error accessing database: {e}")
