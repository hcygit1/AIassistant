from openai import OpenAI
import os
import sys

_USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None
_REASONING_COLOR = "\033[90m" if _USE_COLOR else ""
_RESET_COLOR = "\033[0m" if _USE_COLOR else ""


def _clean_api_key(raw: str | None) -> str:
    if not raw:
        raise SystemExit("Missing NVIDIA_API_KEY in environment")

    api_key = raw.strip()
    api_key = api_key.strip("\"'“”‘’")

    try:
        api_key.encode("ascii")
    except UnicodeEncodeError as exc:
        raise SystemExit(
            "NVIDIA_API_KEY contains non-ASCII characters. "
            "This usually means the key was copied with smart quotes or other invalid characters."
        ) from exc

    if not api_key:
        raise SystemExit("NVIDIA_API_KEY is empty after stripping quotes/whitespace")

    return api_key


def main() -> None:
    api_key = _clean_api_key(os.getenv("NVIDIA_API_KEY"))

    prompt = sys.argv[1] if len(sys.argv) > 1 else "请简单介绍一下你自己。"

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )

    completion = client.chat.completions.create(
        model="z-ai/glm-5.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
        top_p=1,
        max_tokens=16384,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": True,
                "clear_thinking": False,
            }
        },
        stream=True,
    )

    for chunk in completion:
        if not getattr(chunk, "choices", None):
            continue
        if len(chunk.choices) == 0 or getattr(chunk.choices[0], "delta", None) is None:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            print(f"{_REASONING_COLOR}{reasoning}{_RESET_COLOR}", end="")
        if getattr(delta, "content", None) is not None:
            print(delta.content, end="")

    print()


if __name__ == "__main__":
    main()
