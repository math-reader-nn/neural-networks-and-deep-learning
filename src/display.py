"""
Some functions for displaying handwritten digits from the mnist data, as well as
input layer weight vectors for a particular neuron of the first hidden layer.

Usage examples: display(training_data[25][0] displays the 25th letter in 
training_data.

display_weight(net.weights[0],2) displays the weighting that affect the input 
into the 2nd neuron of the first hidden layer.
"""

import pygame as pg
import network2

def display(data):
    scr = pg.display.set_mode((56,56))
    letterarray = (data.reshape(28,28,order = 'F')*255).astype(int)
    letterarray = (letterarray * (0x000001) + letterarray * (0x000100) + letterarray * (0x010000))
    letterarray ^= 2 ** 32 - 1
    temp = pg.Surface((28,28))
    pg.surfarray.blit_array(temp,letterarray)
    temp = pg.transform.scale2x(temp)
    scr.blit(temp,(0,0))
    pg.display.flip()

def display_weight(matrix, i):
    display(network2.sigmoid(matrix[i]) )