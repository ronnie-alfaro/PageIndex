import asyncio
import json
import logging
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_LOCAL_MODEL = "local/default"
DEFAULT_LLAMA_CPP_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"

_LITELLM = None
_LAST_ERROR = None


def _load_litellm():
    global _LITELLM
    if _LITELLM is None:
        from dotenv import load_dotenv
        import litellm

        load_dotenv()
        if not os.getenv("OPENAI_API_KEY") and os.getenv("CHATGPT_API_KEY"):
            os.environ["OPENAI_API_KEY"] = os.getenv("CHATGPT_API_KEY")
        litellm.drop_params = True
        _LITELLM = litellm
    return _LITELLM


def _normalize_model(model):
    return model or DEFAULT_LOCAL_MODEL


def _provider_for(model):
    model = _normalize_model(model)
    if model.startswith(("local/", "llamacpp/")):
        return "llamacpp"
    if model.startswith("ollama/"):
        return "ollama"
    if model.startswith("litellm/"):
        return "litellm"
    if "/" not in model:
        return "llamacpp"
    return "litellm"


def _provider_model(model):
    model = _normalize_model(model)
    if model.startswith("local/"):
        name = model.removeprefix("local/")
        return os.getenv("PAGEINDEX_LLAMA_CPP_MODEL", name if name != "default" else "default")
    if model.startswith("llamacpp/"):
        name = model.removeprefix("llamacpp/")
        return os.getenv("PAGEINDEX_LLAMA_CPP_MODEL", name if name != "default" else "default")
    if model.startswith("ollama/"):
        return model.removeprefix("ollama/")
    if model.startswith("litellm/"):
        return model.removeprefix("litellm/")
    return model


def _messages(prompt, chat_history=None):
    messages = list(chat_history) if chat_history else []
    messages.append({"role": "user", "content": prompt})
    return messages


def _post_json(url, payload, timeout=None):
    timeout = timeout or float(os.getenv("PAGEINDEX_LLM_TIMEOUT", "120"))
    data = json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM endpoint returned HTTP {e.code}: {body}") from e
    except URLError as e:
        raise RuntimeError(f"Could not reach LLM endpoint at {url}: {e.reason}") from e


def _llamacpp_completion(model, prompt, chat_history=None):
    base_url = os.getenv("PAGEINDEX_LLAMA_CPP_BASE_URL", DEFAULT_LLAMA_CPP_BASE_URL).rstrip("/")
    response = _post_json(
        f"{base_url}/chat/completions",
        {
            "model": _provider_model(model),
            "messages": _messages(prompt, chat_history),
            "temperature": 0,
        },
    )
    choice = response.get("choices", [{}])[0]
    message = choice.get("message") or {}
    finish_reason = "max_output_reached" if choice.get("finish_reason") == "length" else "finished"
    return message.get("content", ""), finish_reason


def _ollama_completion(model, prompt, chat_history=None):
    base_url = os.getenv("PAGEINDEX_OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/")
    response = _post_json(
        f"{base_url}/api/chat",
        {
            "model": _provider_model(model),
            "messages": _messages(prompt, chat_history),
            "stream": False,
            "options": {"temperature": 0},
        },
    )
    message = response.get("message") or {}
    return message.get("content", ""), "finished"


def _litellm_completion(model, prompt, chat_history=None):
    litellm = _load_litellm()
    response = litellm.completion(
        model=_provider_model(model),
        messages=_messages(prompt, chat_history),
        temperature=0,
    )
    choice = response.choices[0]
    finish_reason = "max_output_reached" if choice.finish_reason == "length" else "finished"
    return choice.message.content, finish_reason


def llm_completion(model, prompt, chat_history=None, return_finish_reason=False):
    global _LAST_ERROR
    _LAST_ERROR = None
    max_retries = 3 if _provider_for(model) in {"llamacpp", "ollama"} else 10
    for i in range(max_retries):
        try:
            provider = _provider_for(model)
            if provider == "llamacpp":
                content, finish_reason = _llamacpp_completion(model, prompt, chat_history)
            elif provider == "ollama":
                content, finish_reason = _ollama_completion(model, prompt, chat_history)
            else:
                content, finish_reason = _litellm_completion(model, prompt, chat_history)
            return (content, finish_reason) if return_finish_reason else content
        except Exception as e:
            _LAST_ERROR = str(e)
            logging.debug(f"LLM error: {e}")
            if i < max_retries - 1:
                time.sleep(1)
            else:
                if return_finish_reason:
                    return "", "error"
                return ""


async def llm_acompletion(model, prompt):
    return await asyncio.to_thread(llm_completion, model, prompt)


def count_tokens(text, model=None):
    if not text:
        return 0
    provider = _provider_for(model)
    if provider in {"llamacpp", "ollama"}:
        return max(1, len(text) // 4)
    try:
        return _load_litellm().token_counter(model=_provider_model(model), text=text)
    except Exception:
        return max(1, len(text) // 4)


def last_error():
    return _LAST_ERROR
