import io
import os
from sys import argv as args
import subprocess

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

# print('{},{}'.format(sentiment.score, sentiment.magnitude))
print('{}'.format(text))
args=("ffprobe","-v", "error", "-show_entries", "format=duration","-of", "default=noprint_wrappers=1:nokey=1",fileName)
popen = subprocess.Popen(args, stdout = subprocess.PIPE)
popen.wait()
length = popen.stdout.read()
print(length)
os.system('rm ' + fileName)

# --- nicks code --- #
sentiment = (sentiment.score, sentiment.magnitude)
length = int(length)
FPS = 30
NEG_COLOR = np.array([1, 0.2, 0.9])
POS_COLOR = np.array([0.7, 0.9, 0.2])

def change_noise(noise):
    inc = 0.0001
    rand_idx = int(np.random.normal(0, noise.shape[0], 1))
    noise[0, rand_idx] += inc
    return noise

def tile_image(inputArray, cols, rows):
	tile = Image.fromarray(inputArray, 'RGB')
	width, height = tile.size
	complete = Image.new("RGB", (height*rows,width*cols))
	for row in range(0, rows):
		for col in range(0, cols):
			complete.paste(tile, (row*height, col*width))
	return np.array(complete)

latent_dim = 100
print("loading model")
model = load_model('model.h5')

print("opening video to write")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, FPS, (640,480))
#fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter("output.avi", fourcc, 30.0, (640,480))

pos_weight = max((sentiment[0] - 0.5), 0)*2 
neg_weight = max((1 - sentiment[0] - 0.5), 0)*2 
sent_color = NEG_COLOR * neg_weight + POS_COLOR * pos_weight
noise_vec = np.random.normal(0, 1, (1, latent_dim))

for i in range(FPS * length):
    noise_vec = change_noise(noise_vec)
    gen_img = model.predict(noise_vec)
    gen_img = gen_img[0]
    print(gen_img)
    gen_img = (0.5 * gen_img + 0.5)*255
    it = np.nditer(gen_img, flags=['multi_index'],op_flags=['readwrite'])
    for x in range(gen_img.shape[0]):
        for y in range(gen_img.shape[1]):
            gen_img[x,y]*=sent_color

    tiled = tile_image(gen_img, 12, 16)
    cv2.imshow("image", tiled)
    #cv2.imwrite("image.jpg", tiled)
    #print(tiled.shape)
    out.write(tiled)
    #print("image")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        continue

out.release()

# add audio in the sketchiest possible way
from moviepy.editor import *
clip = VideoFileClip("output.avi")
audioclip = AudioFileClip("final.flac")
with_audio = clip.set_audio(audioclip)
clip.write_videofile("music_vid.mp4")

