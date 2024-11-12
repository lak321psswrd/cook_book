from groq import Groq
import base64
import os
from dotenv import load_dotenv
from typing import Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecipeRecommender:
    def __init__(self):
        """Initialize the RecipeRecommender with API configuration."""
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.client = Groq(api_key=self.api_key)
        
        # Model configurations
        self.vision_model = "llama-3.2-11b-vision-preview"
        self.text_model = "llama-3.2-11b-text-preview"

    def _encode_image(self, image_bytes: bytes) -> str:
        """
        Encode image bytes to base64 string.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Base64 encoded string
        """
        try:
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            raise

    def analyze_ingredient(self, image_bytes: bytes) -> Optional[str]:
        """
        Analyze an image to identify ingredients.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            String of comma-separated ingredients
        """
        try:
            base64_image = self._encode_image(image_bytes)
            
            response = self.client.chat.completions.create(
                model=self.vision_model,
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
                temperature=0.7,  # Reduced for more consistent outputs
                max_tokens=1024,
                top_p=1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error analyzing ingredients: {str(e)}")
            return None

    def suggest_recipe(self, ingredients: str) -> Optional[str]:
        """
        Generate recipe suggestions based on identified ingredients.
        
        Args:
            ingredients: Comma-separated string of ingredients
            
        Returns:
            Recipe suggestion as a string
        """
        try:
            prompt = (
                f"Create a detailed recipe using these ingredients: {ingredients}\n"
                "Include:\n"
                "1. Recipe name\n"
                "2. Ingredients with measurements\n"
                "3. Step-by-step instructions\n"
                "4. Cooking time and difficulty level"
            )
            
            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=1024
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error suggesting recipe: {str(e)}")
            return None

    def process_image_and_get_recipe(self, image_path: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Process an image file and return both ingredients and recipe suggestion.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (ingredients, recipe)
        """
        try:
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
                
            ingredients = self.analyze_ingredient(image_bytes)
            if not ingredients:
                return None, None
                
            recipe = self.suggest_recipe(ingredients)
            return ingredients, recipe
            
        except Exception as e:
            logger.error(f"Error processing image and getting recipe: {str(e)}")
            return None, None

def main():
    """Main function to demonstrate the RecipeRecommender usage."""
    try:
        recommender = RecipeRecommender()
        
        # Example usage with an image file
        image_path = "./Steak.jpg"
        ingredients, recipe = recommender.process_image_and_get_recipe(image_path)
        
        if ingredients and recipe:
            print("Identified Ingredients:", ingredients)
            print("\nSuggested Recipe:", recipe)
        else:
            print("Failed to process image or generate recipe.")
            
    except Exception as e:
        logger.error(f"Main function error: {str(e)}")

if __name__ == "__main__":
    main()
