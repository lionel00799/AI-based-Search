from llm.base import BaseLLM
from prompts import RELATED_QUESTION_PROMPT, SEARCH_RESULT_FILTER_PROMPT
from schemas import RelatedQueries, SearchResult, RelatedContents
from utils import list_parser, results_filter
from query_plan import convert_to_related_queries


async def generate_related_queries(
    query: str, search_results: list[SearchResult], llm: BaseLLM
) -> list[str]:
    context = "\n\n".join([f"{str(result)}" for result in search_results])
    context = context[:4000]
    
    related_question_prompt = RELATED_QUESTION_PROMPT.format(query=query, context=context)
    
    full_response = ""
    response_gen = await llm.astream(related_question_prompt)

    async for completion in response_gen:
        full_response += completion.delta or ""
    
    return convert_to_related_queries(full_response)

async def filter_related_contents(
    query: str, search_results: list[SearchResult], llm: BaseLLM
) -> list[str]:
    context = "\n\n".join([f"{str(result)}" for result in search_results])
    context = context[:4000]
    
    related = llm.structured_complete(
        RelatedContents, SEARCH_RESULT_FILTER_PROMPT.format(query=query, context=context)
    )
    
    related_contents = list_parser(related.related_contents)
    
    return results_filter(search_results, related_contents)
