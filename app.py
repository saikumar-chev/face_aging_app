import streamlit as st
import cv2
import numpy as np
from PIL import Image
from model.utils import apply_aging_effect

# Streamlit page configuration
st.set_page_config(page_title="Face Aging / De-Aging", layout="centered")

# Title and description
st.title("👵👶 Realistic Face Aging & De-Aging")
st.markdown("""
Upload a face image to apply realistic aging or de-aging effects. 
- **Aging**: Adds wrinkles, age spots, dulls skin, and grays hair.
- **De-Aging**: Smooths skin, enhances cheek fullness, brightens colors, and boosts hair vibrancy.
Select the target age to control the effect intensity.
""")

# File uploader
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"], help="Upload a clear face image for best results.")

# Effect selection and age slider
col1, col2 = st.columns(2)
with col1:
    effect = st.radio("Choose an effect:", ["Age (👵)", "De-Age (👶)"], help="Aging adds elderly features; De-Aging restores youthful features.")
with col2:
    age = st.slider("Select Target Age", min_value=1, max_value=100, value=50, help="1 is youngest (infant-like), 100 is oldest (elderly).")

# Process and display image
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    if st.button("Apply Effect", help="Click to process the image with the selected effect and age."):
        with st.spinner("Applying effect..."):
            result = apply_aging_effect(image, effect, age)
            result_image = Image.fromarray(result)
            st.image(result_image, caption=f"Result - {effect} (Target Age: {age})", use_column_width=True)
            # Option to download the result
            result_image.save("result_image.png")
            with open("result_image.png", "rb") as file:
                st.download_button(
                    label="Download Result Image",
                    data=file,
                    file_name=f"{effect.lower().replace(' ', '_')}_age_{age}.png",
                    mime="image/png",
                    help="Download the processed image."
                )