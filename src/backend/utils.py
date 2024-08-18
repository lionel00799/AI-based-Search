import os

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


DB_ENABLED = strtobool(os.environ.get("DB_ENABLED", "true"))
PRO_MODE_ENABLED = strtobool(os.environ.get("PRO_MODE_ENABLED", "true"))
