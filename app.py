import streamlit as st
import cv2
import pytesseract
import pandas as pd
import numpy as np
from io import StringIO

def main():
    st.title("Invoice OCR Web App")
    st.write("Upload an image of the invoice and extract information into a CSV file.")

    # File uploader
    image_file = st.file_uploader("Upload Invoice Image", type=['png', 'jpg', 'jpeg'])

    if image_file is not None:
        # Read the image using OpenCV
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Preprocess the image (if required)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Extract text using Pytesseract
        extracted_text = pytesseract.image_to_string(thresh)

        # Process the extracted text and save it to a CSV file
        df = process_extracted_text(extracted_text)
        save_to_csv(df)

        # Display the extracted text
        st.subheader("Extracted Text:")
        st.write(extracted_text)

        # Display the CSV file download link
        st.subheader("Download CSV:")
        st.markdown(get_csv_download_link(df), unsafe_allow_html=True)

def process_extracted_text(text):
    # Process the extracted text and convert it to a DataFrame
    lines = text.split('\n')
    data = []
    for line in lines:
        data.append(line.split())
    df = pd.DataFrame(data)
    return df

def save_to_csv(df):
    # Save the DataFrame to a CSV file
    csv = df.to_csv(index=False)

def get_csv_download_link(df):
    # Generate a download link for the CSV file
    csv = df.to_csv(index=False)
    b64 = csv.encode().decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="invoice_data.csv">Download CSV File</a>'
    return href

if __name__ == '__main__':
    main()
