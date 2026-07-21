from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)

connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

# Delete old collection if it exists
if utility.has_collection("documents"):
    utility.drop_collection("documents")

fields = [
    FieldSchema(
        name="id",
        dtype=DataType.INT64,
        is_primary=True
    ),
    FieldSchema(
        name="text",
        dtype=DataType.VARCHAR,
        max_length=500
    ),
    FieldSchema(
        name="embedding",
        dtype=DataType.FLOAT_VECTOR,
        dim=384
    )
]

schema = CollectionSchema(
    fields,
    description="Test Documents"
)

collection = Collection(
    name="documents",
    schema=schema
)

# Create vector index
collection.create_index(
    field_name="embedding",
    index_params={
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
)

print("Collection and index created!")