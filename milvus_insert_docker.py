from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection

# Connect first
connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Open collection
collection = Collection("documents")

# Load collection
collection.load()

# Data
text = "Milvus is a vector database"

# Convert to embedding
embedding = model.encode(text).tolist()

# Insert
collection.insert([
    [1],
    [text],
    [embedding]
])

print("Inserted successfully!")