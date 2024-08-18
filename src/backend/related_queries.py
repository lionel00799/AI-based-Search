from llm.base import BaseLLM
from prompts import RELATED_QUESTION_PROMPT, SEARCH_RESULT_FILTER_PROMPT
from schemas import RelatedQueries, SearchResult, RelatedContents
from utils import list_parser, results_filter


async def generate_related_queries(
    query: str, search_results: list[SearchResult], llm: BaseLLM
) -> list[str]:
    context = "\n\n".join([f"{str(result)}" for result in search_results])
    context = context[:4000]
    related = llm.structured_complete(
        RelatedQueries, RELATED_QUESTION_PROMPT.format(query=query, context=context)
    )
    
    return list_parser(related.related_questions)

async def filter_related_contents(
    query: str, search_results: list[SearchResult], llm: BaseLLM
) -> list[str]:
    context = "\n\n".join([f"{str(result)}" for result in search_results])
    context = context[:4000]
    
    print(SEARCH_RESULT_FILTER_PROMPT.format(query=query, context=context))
    related = llm.structured_complete(
        RelatedContents, SEARCH_RESULT_FILTER_PROMPT.format(query=query, context=context)
    )
    
    print(related.related_contents)
    
    related_contents = list_parser(related.related_contents)
    
    return results_filter(search_results, related_contents)
