import requests
import re

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =====================================================
# OLLAMA CONFIGURATION
# =====================================================

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

MODEL_NAME = "llama3.2"


# =====================================================
# SHARED CONNECTION SESSION
# =====================================================

_session = None


def get_ollama_session():

    global _session

    if _session is None:

        _session = requests.Session()

        retry_strategy = Retry(
            total=2,
            backoff_factor=0.3,
            status_forcelist=[
                502,
                503,
                504
            ],
            allowed_methods=[
                "POST"
            ]
        )

        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=retry_strategy
        )

        _session.mount(
            "http://",
            adapter
        )

        _session.mount(
            "https://",
            adapter
        )

    return _session


# =====================================================
# REMOVE THINKING / REASONING TAGS
# =====================================================

def clean_model_response(answer):

    if not answer:
        return ""

    answer = str(answer)

    # -------------------------------------------------
    # Remove <think>...</think>
    # -------------------------------------------------

    answer = re.sub(
        r"<think>.*?</think>",
        "",
        answer,
        flags=re.DOTALL | re.IGNORECASE
    )

    # -------------------------------------------------
    # Remove any remaining standalone think tags
    # -------------------------------------------------

    answer = re.sub(
        r"</?think>",
        "",
        answer,
        flags=re.IGNORECASE
    )

    return answer.strip()


# =====================================================
# GENERATE RESPONSE USING OLLAMA
# =====================================================

def generate_with_qwen(
    prompt,
    temperature=0.2,
    max_tokens=100,
    timeout=120,
    think=False
):

    session = get_ollama_session()

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,

        # Complete response instead of streaming
        "stream": False,

        # Keep model loaded
        "keep_alive": "10m",

        # Works with models that support thinking.
        # For non-thinking models such as llama3.2,
        # this will not cause an error.
        "think": think,

        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }

    try:

        response = session.post(
            OLLAMA_URL,
            json=payload,
            timeout=timeout
        )

        response.raise_for_status()

    except requests.exceptions.Timeout:

        print("Ollama request timed out.")

        return ""

    except requests.exceptions.ConnectionError:

        print(
            "Could not connect to Ollama. "
            "Make sure Ollama is running."
        )

        return ""

    except requests.exceptions.RequestException as error:

        print(
            "Ollama request failed:",
            error
        )

        return ""

    # =================================================
    # PARSE JSON
    # =================================================

    try:

        result = response.json()

    except ValueError:

        print(
            "Ollama returned invalid JSON."
        )

        return ""

    # Useful while debugging
    print(
        "OLLAMA RESULT:",
        result
    )

    # =================================================
    # GET RESPONSE
    # =================================================

    answer = result.get(
        "response",
        ""
    )

    # =================================================
    # SAFETY: REMOVE THINKING
    # =================================================

    answer = clean_model_response(
        answer
    )

    # =================================================
    # EMPTY RESPONSE
    # =================================================

    if not answer:

        print(
            "Ollama returned an empty response."
        )

        return ""

    return answer