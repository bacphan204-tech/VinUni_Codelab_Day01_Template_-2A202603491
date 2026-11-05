"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # response.usage contains input_tokens and output_tokens (prompt_tokens/completion_tokens)
    """
    # 
    from openai import OpenAI

    # Khởi tạo OpenAI Client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Bắt đầu đo thời gian
    start = time.perf_counter()

    # Gửi yêu cầu đến OpenAI
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )

    # Tính thời gian phản hồi
    latency = float(time.perf_counter() - start)

    # Lấy nội dung AI trả lời
    response_text = response.choices[0].message.content or ""

    # Lấy số token
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens
    }

    # Trả về kết quả
    return response_text, latency, usage



# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Gemini model to use (default: gemini-2.5-flash).
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        Option A (New Google GenAI SDK):
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            # Configure using types.GenerateContentConfig
            
        Option B (Legacy Google GenerativeAI SDK):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model_inst = genai.GenerativeModel(model)
            # Configure using genai.types.GenerationConfig
            
        Ensure your usage dictionary extracts 'input_tokens' and 'output_tokens' 
        from the response metadata (e.g. response.usage_metadata).
    """
    
    from google import genai
    from google.genai import types

    # 1. Khoi tao Gemini Client
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
    )

    # 2. Cau hinh tham so cho model
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens
    )

    # 3. Bat dau do thoi gian
    start = time.perf_counter()

    # 4. Goi Gemini API
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config
    )

    # 5. Tinh thoi gian phan hoi
    latency = float(time.perf_counter() - start)

    # 6. Lay noi dung phan hoi
    response_text = response.text or ""

    # 7. Lay thong tin token
    usage = {
        "input_tokens": response.usage_metadata.prompt_token_count,
        "output_tokens": response.usage_metadata.candidates_token_count
    }

    # 8. Tra ve ket qua
    return response_text, latency, usage



# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Claude model to use (default: claude-3-5-haiku).
        temperature: Sampling temperature (0.0 - 1.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum output tokens.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # response.usage contains input_tokens and output_tokens
    """
    
    import anthropic

    # 1. Khoi tao Anthropic Client
    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    # 2. Bat dau do thoi gian
    start = time.perf_counter()

    # 3. Goi Anthropic Claude API
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # 4. Tinh thoi gian phan hoi
    latency = float(time.perf_counter() - start)

    # 5. Lay noi dung AI tra loi
    response_text = response.content[0].text

    # 6. Lay thong tin token
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens
    }

    # 7. Tra ve ket qua
    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash (gemini-2.5-flash)
    with the same prompt and return a structured comparison dictionary.

    Calculate the exact USD token cost for input + output using the prices in PRICING_1M_TOKENS.

    Args:
        prompt: The user message to send to all models.

    Returns:
        A dictionary containing:
            - "gpt4o": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gpt4o_mini": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gemini_flash": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
    """
    
    # 1. Goi GPT-4o
    gpt4o_text, gpt4o_latency, gpt4o_usage = call_openai(
        prompt,
        model=OPENAI_MODEL
    )

    # 2. Goi GPT-4o-mini
    mini_text, mini_latency, mini_usage = call_openai(
        prompt,
        model=OPENAI_MINI_MODEL
    )

    # 3. Goi Gemini 2.5 Flash
    gemini_text, gemini_latency, gemini_usage = call_gemini(
        prompt,
        model=GEMINI_MODEL
    )

    # 4. Ham ho tro tinh chi phi va tao ket qua
    def make_stats(text, latency, usage, model_name):
        prices = PRICING_1M_TOKENS[model_name]

        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]

        cost = (
            input_tokens * prices["input"]
            + output_tokens * prices["output"]
        ) / 1_000_000

        return {
            "response": text,
            "latency": latency,
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }

    # 5. Tong hop ket qua so sanh
    results = {
        "gpt4o": make_stats(
            gpt4o_text,
            gpt4o_latency,
            gpt4o_usage,
            OPENAI_MODEL
        ),

        "gpt4o_mini": make_stats(
            mini_text,
            mini_latency,
            mini_usage,
            OPENAI_MINI_MODEL
        ),

        "gemini_flash": make_stats(
            gemini_text,
            gemini_latency,
            gemini_usage,
            GEMINI_MODEL
        )
    }

    # 6. Tra ve ket qua
    return results


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.

    Behaviour:
        - Streams response tokens from Gemini 2.5 Flash as they arrive.
        - Maintains the last 3 turns of conversation history for context.
        - Typing 'quit' or 'exit' ends the session.

    Hints:
        - Maintain a history list of conversation turns.
        - Check how to stream responses using client.chats or model.generate_content(..., stream=True).
        - Keep history limited to the last 3 turns to optimize context window and costs.
    """
   
    from google import genai
    from google.genai import types

    # Khoi tao Gemini Client
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
    )

    # Luu lich su hoi thoai
    history = []

    print("=== Gemini Streaming Chatbot ===")
    print("Type 'quit' or 'exit' to stop.")

    while True:
        user_input = input("\nYou: ").strip()

        # Thoat chatbot
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Tao tin nhan nguoi dung
        user_message = types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=user_input)
            ]
        )

        # Ket hop lich su
        formatted_history = history + [user_message]

        # Goi API streaming
        response_stream = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=formatted_history
        )

        print("AI: ", end="", flush=True)

        response_parts = []

        for chunk in response_stream:
            if chunk.text:
                print(chunk.text, end="", flush=True)
                response_parts.append(chunk.text)

        print()

        # Luu cau tra loi hoan chinh
        full_response = "".join(response_parts)

        model_message = types.Content(
            role="model",
            parts=[
                types.Part.from_text(text=full_response)
            ]
        )

        # Cap nhat lich su
        history.append(user_message)
        history.append(model_message)

        # Chi giu 3 luot hoi thoai gan nhat
        history = history[-6:]



# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    
    # Thu hien lan dau va cac lan thu lai
    for attempt in range(max_retries + 1):

        try:
            # Goi ham can thuc hien
            result = fn()

            # Thanh cong thi tra ve ngay
            return result

        except Exception:

            # Neu da het so lan thu lai
            if attempt == max_retries:
                raise

            # Tinh thoi gian cho theo cap so nhan
            delay = base_delay * (2 ** attempt)

            # Cho truoc khi thu lai
            time.sleep(delay)



# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    
    # Danh sach ket qua
    all_results = []

    # Duyet qua tung prompt
    for prompt in prompts:

        # So sanh cac model
        comparison = compare_models(prompt)

        # Them prompt goc vao ket qua
        result = {
            **comparison,
            "prompt": prompt
        }

        # Luu vao danh sach
        all_results.append(result)

    # Tra ve tat ca ket qua
    return all_results



# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    
    # 1. Tao tieu de bang Markdown
    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "|---|---|---|---:|---:|---:|"
    ]

    # 2. Ten hien thi cua cac model
    model_names = {
        "gpt4o": "GPT-4o",
        "gpt4o_mini": "GPT-4o-mini",
        "gemini_flash": "Gemini-Flash"
    }

    # 3. Duyet qua cac ket qua
    for result in results:

        prompt = str(result["prompt"])
        prompt = prompt.replace("|", "\\|").replace("\n", " ")

        # 4. Tao tung dong cho moi model
        for model_key, model_name in model_names.items():

            stats = result[model_key]

            response = str(stats["response"])
            response = response.replace("\n", " ")

            # Gioi han noi dung trong 50 ky tu
            if len(response) > 50:
                response = response[:47] + "..."

            response = response.replace("|", "\\|")

            latency = stats["latency"]
            input_tokens = stats["input_tokens"]
            output_tokens = stats["output_tokens"]
            cost = stats["cost"]

            # 5. Tao dong Markdown
            row = (
                f"| {prompt} | {model_name} | {response} | "
                f"{latency:.2f}s | "
                f"{input_tokens}/{output_tokens} | "
                f"${cost:.8f} |"
            )

            lines.append(row)

    # 6. Ghep cac dong thanh bang
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini 2.5 Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
