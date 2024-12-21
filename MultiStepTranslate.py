import os
import requests
import argparse
import json

# Configuration
API_ENDPOINT = "http://localhost:5000/v1/completions"
API_KEY = os.environ.get("MY_API_KEY")  # Set via environment variable
DEFAULT_MODEL = "text-davinci-003"
DEFAULT_TEMPERATURE = 0.7

# Build headers
HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["Authorization"] = f"Bearer {API_KEY}"

def make_api_call(
    prompt: str,
    max_tokens: int = 1000,
    temperature: float = DEFAULT_TEMPERATURE,
    model: str = DEFAULT_MODEL
) -> str:
    """
    Make a call to the API with the given parameters.

    :param prompt: The prompt to send to the API.
    :param max_tokens: Maximum tokens for the response.
    :param temperature: Sampling temperature.
    :param model: The name of the model to use.
    :return: The response text from the API.
    :raises RuntimeError: If the API call fails or returns an empty response.
    """
    payload = {
        "model": model,
        "prompt": prompt.strip(),
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = requests.post(API_ENDPOINT, headers=HEADERS, json=payload)
        response.raise_for_status()
        json_response = response.json()
        text = json_response.get("choices", [{}])[0].get("text", "").strip()
        if not text:
            raise RuntimeError("API returned an empty response.")
        return text
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"API call failed: {e}") from e

def translate_text(source_text: str, source_language: str, target_language: str) -> str:
    """
    Perform a multi-step translation workflow.

    :param source_text: Text to translate.
    :param source_language: Language of the source text.
    :param target_language: Language for the translated text.
    :return: The best translation chosen after comparison.
    :raises ValueError: If the comparison output is unexpected.
    """
    # Step 1: Initial Translation
    initial_prompt = f"""
Translate the following text from {source_language} to {target_language}:
{source_text}
Ensure the translation is accurate, fluent, and captures all nuances of the original text. 
Provide only the translation in {target_language}.
"""
    initial_translation = make_api_call(initial_prompt)

    # Step 2: Refine Translation
    refine_prompt = f"""
Improve the following translation for accuracy and fluency. 
Make sure it captures the original meaning and reads naturally in {target_language}:
{initial_translation}
"""
    refined_translation = make_api_call(refine_prompt)

    # Step 3: Compare Translations
    compare_prompt = f"""
Between the two translations below, indicate which is better based on accuracy, fluency, 
and naturalness in {target_language}. Respond with 'Version 1' or 'Version 2':
Version 1: {initial_translation}
Version 2: {refined_translation}
"""
    best_version = make_api_call(compare_prompt)

    # Decide the best translation
    if "Version 1" in best_version:
        return initial_translation
    elif "Version 2" in best_version:
        return refined_translation
    else:
        raise ValueError(f"Unexpected output from API during comparison: {best_version}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate text from one language to another and save the result to a file."
    )
    parser.add_argument("infile", help="Path to the input file containing the text to translate.")
    parser.add_argument("outfile", help="Path to the output file where the translated text will be saved.")
    parser.add_argument("source_language", help="Language of the input text (e.g., 'English').")
    parser.add_argument("target_language", help="Language for the translated text (e.g., 'Spanish').")
    args = parser.parse_args()

    try:
        with open(args.infile, "r", encoding="utf-8") as infile:
            source_text = infile.read()

        translated_text = translate_text(
            source_text, args.source_language, args.target_language
        )

        with open(args.outfile, "w", encoding="utf-8") as outfile:
            outfile.write(translated_text)

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        exit(1)
    except RuntimeError as e:
        print(f"Error during translation: {e}")
        exit(1)
    except ValueError as e:
        print(f"Translation error: {e}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        exit(1)

if __name__ == "__main__":
    main()
