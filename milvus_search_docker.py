from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection

# Connect to Docker Milvus
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

# Question
query = "What is a vector database?"

# Convert question to embedding
query_embedding = model.encode(query).tolist()

# Search
results = collection.search(
    data=[query_embedding],
    anns_field="embedding",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=1
)

print(results)