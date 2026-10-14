#!/usr/bin/env python3
import llm_client
import history_story_generator

if __name__ == "__main__":
    # Load environment variables from .env file
    llm_client.load_env_file()

    topic = "The Rise of the Mali Empire"
    length = "short"

    print(f"Generating story for topic: {topic}")
    print(f"Length mode: {length}")
    print("=" * 60)

# Test default model (gemini-31-flash-lite)
print("\n1. Using default model (gemini-31-flash-lite):")
try:
    story_default = history_story_generator.generate_history_story(
        topic, model_key="gemini-31-flash-lite", length_mode=length
    )
    print(f"Title: {story_default['title']}")
    print("Sentences:")
    for i, s in enumerate(story_default['sentences'], 1):
        print(f"  {i}. {s}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)

# Test new model (gemini-35-flash-lite)
print("\n2. Using new model (gemini-35-flash-lite):")
try:
    story_new = history_story_generator.generate_history_story(
        topic, model_key="gemini-35-flash-lite", length_mode=length
    )
    print(f"Title: {story_new['title']}")
    print("Sentences:")
    for i, s in enumerate(story_new['sentences'], 1):
        print(f"  {i}. {s}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("Comparison complete.")