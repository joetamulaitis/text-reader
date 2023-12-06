from tts import tts_say
from read_text import select_letters, select_letters_manual, select_letters_pytesseract


image = (input('Name of image file: '))

method = int((input('1. Automatic letter selection\n2. Manual letter selection\n3. Tesseract OCR\n')))

if method == 1:
    image_text = select_letters(image)
    text = " ".join(image_text)
    tts_say(text)
elif method == 2:
    image_text = select_letters_manual(image)
    text = " ".join(image_text)
    tts_say(text)
elif method == 3:
    image_text = select_letters_pytesseract(image)
    text = " ".join(image_text)
    tts_say(text)

