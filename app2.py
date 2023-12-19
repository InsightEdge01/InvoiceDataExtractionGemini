import streamlit as st
import google.generativeai as genai
from pathlib import Path

# Configure Google API Key
genai.configure(api_key="AIzaSyA42QdFXNwD0zlDAPtY0LRAvqLHd0bKi84")

# Set up the model
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name="gemini-pro-vision",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

def input_image_setup(uploaded_files, selected_image_index):
    if not uploaded_files or selected_image_index is None:
        raise FileNotFoundError("Please upload images and select an image to ask questions.")

    selected_file = uploaded_files[selected_image_index]

    temp_path = Path(f"temp_image_{selected_image_index}.jpg")
    temp_path.write_bytes(selected_file.read())

    image_parts = [{"mime_type": "image/jpeg", "data": temp_path.read_bytes()}]

    return image_parts, temp_path

def generate_gemini_response(input_prompt, uploaded_files, selected_image_index, question_prompt):
    image_prompt, temp_path = input_image_setup(uploaded_files, selected_image_index)
    prompt_parts = [image_prompt[0], question_prompt]
    response = model.generate_content(prompt_parts)

    # Remove temporary file after generating the response
    temp_path.unlink()

    return response.text

input_prompt = """
               You are an expert in understanding invoices and rece.
               You will receive input images as invoices &
               you will have to answer questions based on the input image
               """

def main():
    st.title("Gemini Pro Vision Streamlit App")

    # File Upload
    st.sidebar.header("Upload Image or Images")
    uploaded_files = st.sidebar.file_uploader("Choose images...", type=["jpg", "png"], accept_multiple_files=True, key="file_uploader")

    # Create a list of file names for display
    file_names = [file.name for file in uploaded_files] if uploaded_files else []

    # Select Image
    selected_image_index = st.sidebar.selectbox(
        "Select Image for Questions",
        options=file_names,
        key="image_selector"
    )

    # Question prompt
    question_prompt = st.text_input("Question Prompt", "")

    if st.button("Generate Response"):
        try:
            response_text = generate_gemini_response(input_prompt, uploaded_files, file_names.index(selected_image_index), question_prompt)

            # Display conversation history and select the chat
            st.subheader("Generated Response:")
            st.write(response_text)
        except FileNotFoundError as e:
            st.warning(str(e))

if __name__ == "__main__":
    main()
