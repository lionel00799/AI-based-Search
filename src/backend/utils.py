import os
from collections import namedtuple
from constants import ChatModel


def is_local_model(model: ChatModel) -> bool:
    return model in [
        ChatModel.LOCAL_LLAMA_3,
        ChatModel.LOCAL_GEMMA,
        ChatModel.LOCAL_MISTRAL,
        ChatModel.LOCAL_PHI3_14B,
        ChatModel.CUSTOM,
    ]


def strtobool(val: str | bool) -> bool:
    if isinstance(val, bool):
        return val
    return val.lower() in ("true", "1", "t")

def list_parser(model_response: str) -> list[str]:
    
    # Step 1: Remove the brackets
    model_response = model_response.strip('[]')

    # Step 2: Split the string by comma
    questions_list = model_response.split(', ')

    # Optional Step 3: Trim any extra whitespace
    questions_list = [question.strip() for question in questions_list]
    questions_list = [question.replace('"', "") for question in questions_list]
    questions_list = [question.replace("'", "") for question in questions_list]
    questions_list = [question.replace('`', "") for question in questions_list]
    
    return questions_list

def results_filter(searchResults: list[str], relatedContents: list[str]) -> list[str]:
    
    # Convert title list to a set for faster lookup
    # title_set = set(relatedContents)

    # Filter the original list
    filtered_list = [item for item in searchResults if item.title in relatedContents]
    return filtered_list
    

DB_ENABLED = strtobool(os.environ.get("DB_ENABLED", "true"))
PRO_MODE_ENABLED = strtobool(os.environ.get("PRO_MODE_ENABLED", "true"))
