from save_to_csv import save_results
import logging
import sys
import utils.multiprocessing
from defaults import get_cfg_defaults
import os

logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

if len(sys.argv) > 1:
    cfg_file = 'configs/' + sys.argv[1]
else:
    cfg_file = 'configs/' + input("Config file:")

cfg = get_cfg_defaults()
cfg.merge_from_file(cfg_file)
cfg.freeze()

# Impostiamo le classi inliner e outliner
inliner_classes = [0, 1, 2, 3]
outliner_classes = [4, 5, 9]

def train_and_evaluate(_):
    import train_AAE
    import novelty_detector

    train_AAE.train(inliner_classes, cfg=cfg)
    res = novelty_detector.main(inliner_classes, outliner_classes, cfg.DATASET.TOTAL_CLASS_COUNT, cfg.DATASET.PERCENTAGES[0], cfg=cfg)
    return res

gpu_count = utils.multiprocessing.get_gpu_count()
results = utils.multiprocessing.map(train_and_evaluate, gpu_count, [None] * gpu_count)  # Passiamo un placeholder per ogni GPU

save_results(results, os.path.join(cfg.OUTPUT_FOLDER, cfg.RESULTS_NAME))
