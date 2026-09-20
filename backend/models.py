import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"


def ask_gemma(prompt, temperature=0.1):
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "num_ctx": 8192,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=data,
        timeout=300,
    )

    response.raise_for_status()

    result = response.json()

    if "response" not in result:
        raise RuntimeError(f"Unexpected Ollama response: {result}")

    return result["response"]


if __name__ == "__main__":
    question = input("You: ")
    answer = ask_gemma(question)
    print("\nGemma:", answer)