import pyttsx3
import autocorrect


def tts_say(inputString):
    engine = pyttsx3.init()
    engine.setProperty('rate', 300)
    spell = autocorrect.Speller()
    engine.say(spell(inputString))
    engine.runAndWait()
    print(inputString)

def ttsSaveToFile(inputString):
    engine = pyttsx3.init()
    engine.setProperty('rate', 300)
    spell = autocorrect.Speller()
    engine.save_to_file(spell(inputString), '../speech.mp3')
    engine.runAndWait()


