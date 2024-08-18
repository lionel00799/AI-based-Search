from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from search.providers.base import SearchProvider
from schemas import SearchResponse, SearchResult

class ElasticsearchSearchProvider(SearchProvider):
    def __init__(self, es_host: str, ssl_assert_fingerprint: str, username: str, passwd: str, index_name_doc: str, index_name_image: str):
        self.es = Elasticsearch(hosts=es_host, ssl_assert_fingerprint=ssl_assert_fingerprint, basic_auth=(username, passwd))
        self.index_name_doc = index_name_doc
        self.index_name_image = index_name_image

    def search(self, query: str) -> SearchResponse:

        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        model_image = SentenceTransformer("sentence-transformers/clip-ViT-B-32")
    
        # Generate embedding for the query
        query_embedding = model.encode([query])[0]
        query_embedding_image = model_image.encode([query])[0]

        response = self.es.search(
            index=self.index_name_doc,
            body={
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": 10,
                    "num_candidates": 50,
                },
                "size": 5,  # Number of results to return
            }
        )

        response_image = self.es.search(
            index=self.index_name_image,
            body={
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding_image,
                    "k": 10,
                    "num_candidates": 50,
                },
                "size": 5,  # Number of results to return
            }
        )

        hits = response['hits']['hits']
        results = [SearchResult(title="ABC", url=hit['_source']['metadata']['path'], content=hit['_source']['content'][:500]) for hit in hits]
        
        image_results = []
        hits_image = response_image['hits']['hits']
        for hit in hits_image:
            image_results.append(hit['_source']['metadata']['path'])
        return SearchResponse(results=results, images=image_results)
