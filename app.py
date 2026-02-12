from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import time
import os

app = Flask(__name__)

# ==================== LOAD FREE LOCAL MODEL ====================
print("Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # FREE, runs locally
print("Model loaded successfully!")

# ==================== LOAD DOCUMENTS ====================
with open('documents.json', 'r') as f:
    documents = json.load(f)

print(f"Loaded {len(documents)} documents")

# ==================== EMBEDDING FUNCTIONS ====================

def get_embedding(text):
    """Convert text to embedding vector using FREE local model"""
    return model.encode(text)


def compute_all_embeddings():
    """Compute embeddings for all documents (run once)"""
    print("Computing embeddings for all documents...")
    
    # Extract all content
    contents = [doc['content'] for doc in documents]
    
    # Batch encode (much faster!)
    embeddings = model.encode(contents, show_progress_bar=True)
    
    # Save embeddings to file
    np.save('doc_embeddings.npy', embeddings)
    print("Embeddings saved to doc_embeddings.npy")
    return embeddings


# Load or compute embeddings
try:
    doc_embeddings = np.load('doc_embeddings.npy')
    print(f"Loaded cached embeddings: {doc_embeddings.shape}")
except FileNotFoundError:
    doc_embeddings = compute_all_embeddings()

# ==================== SEARCH FUNCTIONS ====================

def calculate_cosine_similarity(query_embedding, doc_embeddings):
    """Calculate cosine similarity between query and all documents"""
    # Reshape for sklearn
    query_embedding = query_embedding.reshape(1, -1)
    
    # Calculate similarities
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    return similarities


def initial_retrieval(query, k=8):
    """Stage 1: Fast retrieval using embeddings"""
    # Convert query to embedding
    query_embedding = get_embedding(query)
    
    # Calculate similarity with all documents
    similarities = calculate_cosine_similarity(query_embedding, doc_embeddings)
    
    # Get top k indices
    top_k_indices = np.argsort(similarities)[::-1][:k]
    
    # Format results
    results = []
    for idx in top_k_indices:
        results.append({
            'id': int(idx),
            'score': float(similarities[idx])
        })
    
    return results


def simple_rerank(query, candidates, rerank_k=5):
    """Stage 2: Simple re-ranking using query-document similarity"""
    # For free version, use a simple heuristic re-ranking
    # We'll check if query keywords appear in document
    
    reranked = []
    query_words = set(query.lower().split())
    
    for candidate in candidates:
        doc_id = candidate['id']
        doc_content = documents[doc_id]['content'].lower()
        
        # Count keyword matches
        keyword_matches = sum(1 for word in query_words if word in doc_content)
        
        # Combine original score with keyword matching
        combined_score = (candidate['score'] * 0.7) + (keyword_matches / len(query_words) * 0.3)
        
        reranked.append({
            'id': doc_id,
            'score': min(combined_score, 1.0)  # Cap at 1.0
        })
    
    # Sort by new scores
    reranked.sort(key=lambda x: x['score'], reverse=True)
    
    # Return top rerank_k
    return reranked[:rerank_k]


# ==================== API ENDPOINT ====================

@app.route('/search', methods=['POST'])
def search():
    """Main search endpoint"""
    start_time = time.time()
    
    try:
        # Get request parameters
        data = request.json
        query = data.get('query', '')
        k = data.get('k', 8)
        rerank = data.get('rerank', True)
        rerank_k = data.get('rerankK', 5)
        
        # Validate inputs
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        if k < 1 or k > len(documents):
            return jsonify({'error': f'k must be between 1 and {len(documents)}'}), 400
        
        # Stage 1: Initial retrieval
        candidates = initial_retrieval(query, k)
        
        # Stage 2: Re-ranking (if requested)
        if rerank:
            final_results = simple_rerank(query, candidates, rerank_k)
        else:
            final_results = candidates[:rerank_k]
        
        # Format response
        results = []
        for result in final_results:
            doc_id = result['id']
            results.append({
                'id': doc_id,
                'score': round(result['score'], 2),
                'content': documents[doc_id]['content'],
                'metadata': documents[doc_id]['metadata']
            })
        
        # Calculate latency
        latency = int((time.time() - start_time) * 1000)
        
        return jsonify({
            'results': results,
            'reranked': rerank,
            'metrics': {
                'latency': latency,
                'totalDocs': len(documents)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'documents': len(documents),
        'model': 'all-MiniLM-L6-v2'
    })


if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
