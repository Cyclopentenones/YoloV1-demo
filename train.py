from criterion import Loss
from dataset import Dataset 
from PIL import Image 
import torch 
import os 
import numpy as np 

def train(train_path, val_path, 