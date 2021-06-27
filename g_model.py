import urllib
# import urllib2
import requests
import zipfile
from disentanglement_lib.utils import results
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import gin.tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
import tensorflow_hub as hub
from disentanglement_lib.visualize.visualize_model import sigmoid

def glib_model(id):
    return f"https://storage.googleapis.com/disentanglement_lib/unsupervised_study_v1/{id}.zip"


def download_model(id):
    urllib.request.urlretrieve(glib_model(id), "tmp.zip")
    file = zipfile.ZipFile("tmp.zip",'r')
    file.extractall('tmp')
    # !mv tmp/{id}/* tmp
    # !rm -r tmp/{id}



def get_tf_decoder(f):
    def _decoder(latent_vectors):
        return sigmoid(f(
            dict(latent_vectors=latent_vectors),
            signature="decoder",
            as_dict=True)["images"]).transpose(0,3,1,2)
    return _decoder

def get_tf_encoder(f):
    def _encoder(imgs):
        return f(
            dict(images=x.transpose(0,3,1,2)), signature="gaussian_encoder", as_dict=True)
    return _encoder