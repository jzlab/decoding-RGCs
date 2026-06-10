import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def train():
    """need to produce an organized RetinaAutoencoder class that performs both the encoding and decoding
    so that i can get the gradients to flow more easily (both comptuationally and in an organized way). 
    currently my files and methods are kind of spread out everywhere and they should be consolidated in 
    one place.
    """
    
