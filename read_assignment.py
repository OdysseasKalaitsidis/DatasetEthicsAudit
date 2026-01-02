from docx import Document
import sys

def read_docx(file_path):
    try:
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    path = "Ηθική της Πληροφορικής- ΕΡΓΑΣΙΑ - ΒΡΑΧΑΤΗΣ - Αντιγραφή.docx"
    content = read_docx(path)
    with open("assignment_text.txt", "w", encoding="utf-8") as f:
        f.write(content)
    print("Done writing to assignment_text.txt")
