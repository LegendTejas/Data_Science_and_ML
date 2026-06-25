# If you only need text extraction, pdfminer.six is a solid choice.
# If you need to extract text, images, edit PDFs, annotate documents, or render pages, PyMuPDF is often the better option because it combines many capabilities in one library.

import io

import fitz
from PIL import Image


# Open the PDF that contains the embedded images.
pdf = fitz.open("sample.pdf")
counter = 1

for i in range(len(pdf)):
    page = pdf[i]
    # Collect every image reference found on the current page.
    images = page.get_images(full=True)
    for image in images:
        # Extract the raw image bytes and metadata for the current image.
        base_img = pdf.extract_image(image[0])
        print(base_img)
        image_data = base_img["image"]
        # Convert the bytes into a Pillow image so it can be saved to disk.
        img = Image.open(io.BytesIO(image_data))
        extension = base_img["ext"]
        # Save each image with a unique filename using the original file type.
        img.save(f"image{counter}.{extension}")
        counter += 1