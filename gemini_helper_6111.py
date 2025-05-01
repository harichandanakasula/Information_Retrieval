# gemini_helper_6111.py

import google.generativeai as genai


GEMINI_API_KEY = "YOUR_GEMINI_KEY"  

def get_gemini_completion(
    prompt,
    model_name="gemini-1.5-pro",
    max_tokens=200,
    temperature=0.2,
    top_p=1,
    top_k=32,
    gemini_key=None
):
    
    key_to_use = gemini_key or GEMINI_API_KEY
    if not key_to_use:
        raise ValueError("Gemini API key is required but not provided.")

    try:
        genai.configure(api_key=key_to_use)
        model = genai.GenerativeModel(model_name)
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k
        )
        response = model.generate_content(prompt, generation_config=generation_config)
        return response.text.strip() if response.text else "No response received"
    except Exception as e:
        return f"ERROR: {e}"

def main():
    prompt_text = """Given a sentence, extract all the Nouns.
sentence: Rob is an engineer at NASA and he lives in California.
extracted:"""

    response_text = get_gemini_completion(
        prompt_text,
        model_name="gemini-1.5-pro",
        max_tokens=100,
        temperature=0.2,
        top_p=1,
        top_k=32,
        gemini_key="AIzaSyBO5WY7_4iXLj4HXyBTTRKM2Kc_ZBdrcvk"  
    )
    print(response_text)

if __name__ == "__main__":
    main()

