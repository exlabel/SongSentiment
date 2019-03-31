import io
import os
from sys import argv as args
import random
import string

# Imports the Google Cloud client library
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("cloudconfig.json")

# Instantiates a client
client = speech.SpeechClient()

fileName = ''.join(random.choices(string.ascii_uppercase, k=13)).join('.flac')

os.system('ffmpeg -i ' + args[1] + ' -ac 1 -sample_fmt s16 -f flac ' + fileName)
# The name of the audio file to transcribe
file_name = os.path.join(
    os.path.dirname(__file__),
    fileName)

# Loads the audio into memory
with io.open(file_name, 'rb') as audio_file:
    content = audio_file.read()
    if len(args) > 2:
        pass #Here we will make it work with URLs so we can work with audio files longer than a minute
    audio = types.RecognitionAudio(content=content)#, uri='gs://nolabel-b9748.appspot.com/fyslwymmd.flac')

config = types.RecognitionConfig(
    encoding=enums.RecognitionConfig.AudioEncoding.FLAC,
    sample_rate_hertz=44100,
    language_code='en-US',
    model='default')

# Detects speech in the audio file
# This stuff is for when we want audio files > 1 min
"""
operation = client.long_running_recognize(config, audio)

response = operation.result(timeout=150)
"""

response = client.recognize(config, audio)
for result in response.results:
    print('Transcript: {}'.format(result.alternatives[0].transcript))