import PyPDF2

def extract_text_from_pdf(file):

    text = ""

    try:
        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:
            content = page.extract_text()

            if content:
                text += content

    except:
        text = "Invalid PDF file."

    return text