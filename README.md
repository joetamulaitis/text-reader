# OCR Final Project
Our CS 445 project is a OCR engine that can read characters/words off of images and then output it with text to speech. There are 3 methods available: automatic character selection, manual character selection, and Tesseract OCR.

## Authors
Joe Tamulaitis - **jpt5**
Ben Rosen - **rosen14**
Eric Armendariz - **earmen3**

## Prepare to run
To utilize this project, ensure you have a Python 3.7+ environment on your local setup, and use `pip` to install the required libraries as are featured in the `requirements.txt` file. 

Additionally, Tesseract OCR must be installed to run this software. The link for Tesseract is: https://github.com/tesseract-ocr/tesseract. This [helpful page](https://stackoverflow.com/a/52231794/3677087) on StackOverflow goes over quick shortcuts to install it on macOS, Linux, and Windows.
## Usage

1.  Setup a Python version 3.7 or later environment on your machine and run the `main.py` script.
    
2.  You will be prompted to enter the name of a file that is in the directory of the project. Enter the name of this image.
    
3.  Choose one of the following 3 methods for which technology will be used.
    
    -   `1` for automatic letter selection.
    -   `2` for manual letter selection.
    -   `3` for tesseract OCR.
4. The script will now run, which will extract the text off of the image and then read it aloud with the text-to-speech engine.

## References

 - https://stackoverflow.com/a/52231794/3677087
 - https://medium.com/geekculture/building-a-complete-ocr-engine-from-scratch-in-python-be1fd184753b
 - https://www.nist.gov/itl/products-and-services/emnist-dataset
 - https://www.geeksforgeeks.org/text-detection-and-extraction-using-opencv-and-ocr/

