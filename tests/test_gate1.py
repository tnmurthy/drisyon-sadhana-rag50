from src.pipeline import chunk_text, cosine_similarity

def test_gate1_chunking_invariants():
    sample = "A" * 1200
    chunks = chunk_text(sample, chunk_size=500, overlap=50)
    assert len(chunks) >= 3, "Chunks must partition the input"
    assert len(chunks[0]) == 500

def test_gate1_vector_similarity():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]
    assert abs(cosine_similarity(v1, v2) - 1.0) < 1e-5
    assert abs(cosine_similarity(v1, v3) - 0.0) < 1e-5
