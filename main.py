import io
import os
from sys import argv as args
import random
import string

# Imports the Google Cloud client library
from google.cloud import speech
from google.cloud.speech import enums as speech_enums
from google.cloud.speech import types as speech_types

# Imports the Google Cloud client library
from google.cloud import language
from google.cloud.language import enums as language_enums
from google.cloud.language import types as language_types

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("cloudconfig.json")

# Instantiates a client
client = speech.SpeechClient()

fileName = 'final.flac'

inputFile = os.path.abspath(args[1])
os.system('ffmpeg -i ' + inputFile + ' -ac 1 -sample_fmt s16 -f flac ' + fileName + ' >/dev/null 2>&1')


# Loads the audio into memory
with io.open(fileName, 'rb') as audio_file:
    content = audio_file.read()
    if len(args) > 2:
        pass #Here we will make it work with URLs so we can work with audio files longer than a minute
    audio = speech_types.RecognitionAudio(content=content)#, uri='gs://nolabel-b9748.appspot.com/fyslwymmd.flac')

config = speech_types.RecognitionConfig(
    encoding=speech_enums.RecognitionConfig.AudioEncoding.FLAC,
    sample_rate_hertz=44100,
    language_code='en-US',
    model='default')

# Detects speech in the audio file
# This stuff is for when we want audio files > 1 min
"""
operation = client.long_running_recognize(config, audio)

response = operation.result(timeout=150)
"""
text = ''

response = client.recognize(config, audio)
os.system('rm ' + fileName)
for result in response.results:
    text += result.alternatives[0].transcript
if (len(response.results) == 0):
    print('sux to succ')

# use results to do sentiment analysis

# Instantiates a client
lang_client = language.LanguageServiceClient()

document = language_types.Document(
    content=text,
    type=language_enums.Document.Type.PLAIN_TEXT)

# Detects the sentiment of the text
sentiment = lang_client.analyze_sentiment(document=document).document_sentiment

print('{},{}'.format(sentiment.score, sentiment.magnitude))
print('{}'.format(text))