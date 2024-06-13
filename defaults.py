from yacs.config import CfgNode as CN

_C = CN()

_C.OUTPUT_DIR = "results"

_C.DATASET = CN()

# updating the % of outliners classes. I just want, for the moment, 30 out and 70 in
_C.DATASET.PERCENTAGES = [30] 

# Valori per MNIST
_C.DATASET.MEAN = 0.1307
_C.DATASET.STD = 0.3081

_C.DATASET.PATH = "mnist"
_C.DATASET.TOTAL_CLASS_COUNT = 10
_C.DATASET.FOLDS_COUNT = 5

_C.MODEL = CN()
_C.MODEL.LATENT_SIZE = 32
_C.MODEL.INPUT_IMAGE_SIZE = 32
_C.MODEL.INPUT_IMAGE_CHANNELS = 1
_C.MODEL.Z_DISCRIMINATOR_CROSS_BATCH = False

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 256
_C.TRAIN.EPOCH_COUNT = 80
_C.TRAIN.BASE_LEARNING_RATE = 0.002

_C.TEST = CN()
_C.TEST.BATCH_SIZE = 1024

_C.MAKE_PLOTS = True

_C.OUTPUT_FOLDER = 'results'
_C.RESULTS_NAME = 'results.csv'
_C.LOSSES = 'classic'

def get_cfg_defaults():
    return _C.clone()
