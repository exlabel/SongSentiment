import numpy
from PIL import Image

def tileImage(inputArray, cols, rows):
	tile = Image.fromarray(inputArray, 'BGR')
	width, height = tile.size
	complete = Image.new("RGB", (height*rows,width*cols))
	for row in range(0, rows):
		for col in range(0, cols):
			complete.paste(tile, (row*height, col*width))
	return numpy.array(complete)

def tileImagePath(inputPath, cols, rows):
	tile = Image.open(inputPath)#Image.fromarray(inputArray, 'BGR')
	width, height = tile.size
	complete = Image.new("RGB", (height*rows,width*cols))
	for row in range(0, rows):
		for col in range(0, cols):
			complete.paste(tile, (row*height, col*width))
	complete.save('final.png')
	 
# tileImagePath("images/real.png", 4, 10)