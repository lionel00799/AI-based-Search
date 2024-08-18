from llm.base import BaseLLM
from prompts import RELATED_QUESTION_PROMPT
from schemas import RelatedQueries, SearchResult


async def generate_related_queries(
    query: str, search_results: list[SearchResult], llm: BaseLLM
) -> list[str]:
    context = "\n\n".join([f"{str(result)}" for result in search_results])
    context = context[:4000]
    related = llm.structured_complete(
        RelatedQueries, RELATED_QUESTION_PROMPT.format(query=query, context=context)
    )
    
    # Step 1: Remove the brackets
    related.related_questions = related.related_questions.strip('[]')

    # Step 2: Split the string by comma
    questions_list = related.related_questions.split(', ')

    # Optional Step 3: Trim any extra whitespace
    questions_list = [question.strip() for question in questions_list]
    questions_list = [question.replace('"', "") for question in questions_list]
    questions_list = [question.replace("'", "") for question in questions_list]
    
    return questions_list
