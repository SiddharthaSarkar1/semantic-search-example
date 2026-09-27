from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

query = "refund"

documents = [
    "money refunded",
    "money back",
    "return payment",
    "cancel order",
    "funding",
    "eating pizza"
]

query_embedding = model.encode(query)
doc_embeddings  = model.encode(documents)
scores          = util.cos_sim(query_embedding, doc_embeddings)

for doc, score in zip(documents, scores[0]):
    print(f"{doc:20} → similarity: {score:.2f}")
