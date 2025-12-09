from sentence_transformers import SentenceTransformer, util

# Load a small, fast model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Two lyrics to test (short lines for example)
lyric_1 = """
I'm feeling lonely, I'm losing my mind
Walking through the night with no one by my side
"""

lyric_2 = """
I wander all alone, emptiness inside
Another night with nobody near me
"""

# Embed them
emb1 = model.encode(lyric_1, convert_to_tensor=True)
emb2 = model.encode(lyric_2, convert_to_tensor=True)

# Cosine similarity
similarity = util.cos_sim(emb1, emb2).item()

print(f"Cosine similarity: {similarity:.4f}")

if similarity > 0.5:
    print("→ These lyrics are semantically similar")
else:
    print("→ These lyrics are NOT similar")
