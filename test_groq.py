#!/usr/bin/env python3
import llm_client
if __name__ == "__main__":
    llm_client.load_env_file()

    print("Testing Groq models...")
    for model in ["groq-gpt-oss-120b", "groq-qwen3-8-27b"]:
        print(f"\n--- {model} ---")
        try:
            resp = llm_client.call_llm(
                [{"role": "user", "content": "Say hello in one sentence."}],
                model_key=model,
                max_tokens=20,
                temperature=0.0,
            )
            print(resp)
        except Exception as e:
            print(f"Error: {e}")