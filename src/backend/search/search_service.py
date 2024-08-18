import json
import os

import redis
from dotenv import load_dotenv
from fastapi import HTTPException

from schemas import SearchResponse
from search.providers.base import SearchProvider
from search.providers.elasticsearch import ElasticsearchSearchProvider

load_dotenv()


redis_url = os.getenv("REDIS_URL")
redis_client = redis.Redis.from_url(redis_url) if redis_url else None

def get_elasticsearch_config():
    es_host = os.getenv("ELASTICSEARCH_HOST")
    ssl_assert_fingerprint = os.getenv("ELASTICSEARCH_SSL_FINGERPRINT")
    username = os.getenv("ELASTICSEARCH_USERNAME")
    passwd = os.getenv("ELASTICSEARCH_PASSWORD")
    index_name_doc = os.getenv("ELASTICSEARCH_INDEX_DOC")
    index_name_image = os.getenv("ELASTICSEARCH_INDEX_IMAGE")

    if not es_host or not ssl_assert_fingerprint or not username or not passwd or not index_name_doc or not index_name_image:
        raise HTTPException(
            status_code=500,
            detail="One or more Elasticsearch environment variables are not set. "
                   "Please set ELASTICSEARCH_HOST, ELASTICSEARCH_SSL_FINGERPRINT, "
                   "ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD, and ELASTICSEARCH_INDEX."
        )

    return es_host, ssl_assert_fingerprint, username, passwd, index_name_doc, index_name_image

def get_search_provider() -> SearchProvider:
    search_provider = os.getenv("SEARCH_PROVIDER", "elasticsearch")
    es_host, ssl_assert_fingerprint, username, passwd, index_name_doc, index_name_image = get_elasticsearch_config()
    return ElasticsearchSearchProvider(es_host, ssl_assert_fingerprint, username, passwd, index_name_doc, index_name_image)            


async def perform_search(query: str) -> SearchResponse:
    search_provider = get_search_provider()

    try:
        cache_key = f"search:{query}"
        if redis_client and (cached_results := redis_client.get(cache_key)):
            cached_json = json.loads(json.loads(cached_results.decode("utf-8")))  # type: ignore
            return SearchResponse(**cached_json)

        results = search_provider.search(query)

        if redis_client:
            redis_client.set(cache_key, json.dumps(results.model_dump_json()), ex=7200)

        return results
    except Exception:
        raise HTTPException(
            status_code=500, detail="There was an error while searching."
        )

