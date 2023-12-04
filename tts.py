import pyttsx3
import autocorrect


def tts(inputString):
    engine = pyttsx3.init()
    engine.setProperty('rate', 300)
    spell = autocorrect.Speller()
    engine.say(spell(inputString))
    engine.runAndWait()
    # print(engine.getProperty('rate'))

def ttsSaveToFile(inputString):
    engine = pyttsx3.init()
    engine.setProperty('rate', 300)
    spell = autocorrect.Speller()
    engine.save_to_file(spell(inputString), '../speech.mp3')
    engine.runAndWait()


tts("hello")
ttsSaveToFile("hello")
