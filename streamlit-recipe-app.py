# app.py
import streamlit as st
from recipe_recommender import RecipeRecommender
import logging
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_session_state():
    """Initialize session state variables."""
    if 'recommender' not in st.session_state:
        try:
            st.session_state.recommender = RecipeRecommender()
        except Exception as e:
            st.error(f"Failed to initialize RecipeRecommender: {str(e)}")
            return False
    return True

def display_image(image_bytes):
    """Display uploaded image in the UI."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption="Uploaded Image", use_column_width=True)
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")

def main():
    # Page configuration
    st.set_page_config(
        page_title="AI-Powered Cooking Assistant",
        page_icon="🍳",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .stTitle {
            font-size: 42px !important;
            padding-bottom: 20px;
        }
        .recipe-section {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title
    st.title("🍳 AI-Powered Cooking Assistant")

    # Initialize RecipeRecommender
    if not initialize_session_state():
        st.stop()

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload ingredient images",
        type=["png", "jpg", "jpeg"],
        help="Drag and drop your ingredient image here. Limit 200MB per file."
    )

    if uploaded_file:
        try:
            # Display the uploaded image
            image_bytes = uploaded_file.read()
            display_image(image_bytes)

            # Create columns for results
            col1, col2 = st.columns(2)

            with col1:
                # Analyze ingredients
                with st.spinner("Analyzing ingredients..."):
                    ingredients = st.session_state.recommender.analyze_ingredient(image_bytes)
                    if ingredients:
                        st.markdown("### 📝 Identified Ingredients")
                        st.markdown(f"_{ingredients}_")
                    else:
                        st.error("Failed to identify ingredients in the image.")
                        st.stop()

            with col2:
                # Generate recipe
                with st.spinner("Generating recipe..."):
                    recipe = st.session_state.recommender.suggest_recipe(ingredients)
                    if recipe:
                        st.markdown("### 👩‍🍳 Suggested Recipe")
                        st.markdown(recipe)
                    else:
                        st.error("Failed to generate recipe suggestions.")
                        st.stop()

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Error processing request: {str(e)}")

    # Add helpful instructions
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. Upload a clear image of your ingredients
        2. Wait for the AI to identify the ingredients
        3. Get a personalized recipe suggestion
        4. The recipe will include ingredients, measurements, and cooking instructions
        """)

if __name__ == "__main__":
    main()
