from groq import Groq
import base64
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client with API key
client = Groq(api_key=os.getenv("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"))
def analyze_ingredient(image_bytes):
    # Convert the image to base64
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    # Send request to Groq API to identify ingredients
    response = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                     "type": "text",
                     "text": "Identify the ingredients in this image. 'Only the ingredients' comma separated and nothing else."
                  },
                    {
                     "type": "image_url",
                     "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                     }
                  }
                ]
            }
        ],
        stream=False,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stop=None,
    )

    return response.choices[0].message.content



def suggest_recipe(ingredients):
    # Send identified ingredients to Llama3.2 to get recipe suggestions
    response = client.chat.completions.create(
        model="llama-3.2-11b-text-preview",
        messages=[
            {"role": "user", "content": f"Suggest a recipe using these ingredients: {ingredients}"}
        ]
    )
    return response.choices[0].message.content
if __name__ == "__main__":
    with open("./Steak.jpg", "rb") as image_file:
        image_bytes = image_file.read()
    # Example image bytes (replace with actual image bytes)
    image_bytes = b'...'  # Replace with actual image bytes

    # Analyze ingredients in the image
    ingredients = analyze_ingredient(image_bytes)
    print("Identified Ingredients:", ingredients)

    # Suggest a recipe based on the identified ingredients
    recipe = suggest_recipe(ingredients)
    print("Suggested Recipe:", recipe)