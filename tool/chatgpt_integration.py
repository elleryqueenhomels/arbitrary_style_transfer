import requests

# Replace these with your own API keys
chatgpt_api_key = "your_chatgpt_api_key_here"
dalle_api_key = "your_dalle_api_key_here"

# Replace with the actual API endpoint
chatgpt_api_url = "https://api.openai.com/v1/engines/davinci-codex/completions"
dalle_api_url = "https://api.example-dalle.com/v1/images/generate"

def generate_text(prompt):
    headers = {
        "Authorization": f"Bearer {chatgpt_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "max_tokens": 50,
        "temperature": 0.5
    }
    response = requests.post(chatgpt_api_url, headers=headers, json=data)
    return response.json()["choices"][0]["text"].strip()

def generate_image(text):
    headers = {
        "Authorization": f"Bearer {dalle_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": text,
        "num_images": 1
    }
    response = requests.post(dalle_api_url, headers=headers, json=data)
    return response.json()["data"][0]["url"]

def main():
    prompt = input("Enter a text prompt for ChatGPT: ")
    text_output = generate_text(prompt)
    print(f"Generated text: {text_output}")

    image_url = generate_image(text_output)
    print(f"Generated image URL: {image_url}")

if __name__ == "__main__":
    main()
