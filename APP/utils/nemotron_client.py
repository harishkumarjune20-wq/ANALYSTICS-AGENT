# =====================================================
# NVIDIA NEMOTRON CLIENT
# =====================================================

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NEMOTRON_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"


def _get_nvidia_api_key():
    """Read NVIDIA API key from environment or Streamlit secrets."""

    api_key = os.getenv("NVIDIA_API_KEY")

    if api_key:
        return api_key

    try:
        import streamlit as st

        api_key = st.secrets.get("NVIDIA_API_KEY")

        if api_key:
            return api_key
    except Exception:
        pass

    return None


def generate_with_nemotron(
    prompt,
    temperature=0.2,
    max_tokens=500,
    timeout=120,
):
    """Generate analytics Insights/Summary with NVIDIA Nemotron."""

    api_key = _get_nvidia_api_key()

    if not api_key:
        raise RuntimeError(
            "NVIDIA_API_KEY is not configured. "
            "Add NVIDIA_API_KEY to your .env file or Streamlit secrets."
        )

    client = OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=api_key,
        timeout=timeout,
    )

    response = client.chat.completions.create(
        model=NEMOTRON_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are the Analytics Agent Business Analyst. "
                    "Analyze only the supplied analytics data. "
                    "Never invent numbers, dates, suppliers, routes, "
                    "causes, or trends."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        },
        stream=False,
    )

    if not response.choices:
        return ""

    content = response.choices[0].message.content

    return content.strip() if content else ""
