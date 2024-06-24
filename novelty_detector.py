import torch.utils.data
from torchvision.utils import save_image
from net import *
from torch.autograd import Variable
from utils.jacobian import compute_jacobian_autograd
import numpy as np
import logging
import os
import scipy.optimize
from dataloading import make_datasets
from evaluation import get_f1, evaluate
from utils.threshold_search import find_maximum
from utils.save_plot import save_plot
import matplotlib.pyplot as plt
import scipy.stats
from scipy.special import loggamma


def r_pdf(x, bins, counts):
    if bins[0] < x < bins[-1]:
        i = np.digitize(x, bins) - 1
        return max(counts[i], 1e-308)
    if x < bins[0]:
        return max(counts[0] * x / bins[0], 1e-308)
    return 1e-308

def extract_statistics(cfg, train_set, E, G):
    zlist = []
    rlist = []

    device = torch.cuda.current_device()
    E.to(device)
    G.to(device)

    for label, x in train_set:
        x = x.view(-1, 1, cfg.MODEL.INPUT_IMAGE_SIZE, cfg.MODEL.INPUT_IMAGE_SIZE).to(device).float()
        z = E(x).view(x.size(0), -1)
        z = z.unsqueeze(0) if z.dim() == 1 else z
        recon_batch = G(z.view(-1, cfg.MODEL.LATENT_SIZE, 1, 1))
        z = z.squeeze()
        recon_batch = recon_batch.squeeze().cpu().detach().numpy()
        x = x.squeeze().cpu().detach().numpy()
        z = z.cpu().detach().numpy()

        distance = np.linalg.norm(x.flatten() - recon_batch.flatten())
        rlist.append(distance)
        zlist.append(z)

    zlist = np.array(zlist)
    counts, bin_edges = np.histogram(rlist, bins=30, density=True)

    def fmin(func, x0, args, disp):
        x0 = [2.0, 0.0, 1.0]
        return scipy.optimize.fmin(func, x0, args, xtol=1e-12, ftol=1e-12, disp=0)

    gennorm_param = np.zeros([3, cfg.MODEL.LATENT_SIZE])
    for i in range(cfg.MODEL.LATENT_SIZE):
        betta, loc, scale = scipy.stats.gennorm.fit(zlist[:, i], optimizer=fmin)
        gennorm_param[0, i] = betta
        gennorm_param[1, i] = loc
        gennorm_param[2, i] = scale

    return counts, bin_edges, gennorm_param

def run_novely_prediction_on_images(images, labels, inliner_classes, cfg, counts, bin_edges, gennorm_param, threshold=None, E=None, G=None):
    results = []
    gt_novel = []

    include_jacobian = True

    N = (cfg.MODEL.INPUT_IMAGE_SIZE * cfg.MODEL.INPUT_IMAGE_SIZE - cfg.MODEL.LATENT_SIZE) * cfg.DATASET.PERCENTAGES[0]
    logC = loggamma(N / 2.0) - (N / 2.0) * np.log(2.0 * np.pi)

    def logPe_func(x):
        return logC - (N - 1) * np.log(x) + np.log(r_pdf(x, bin_edges, counts))

    for i, (label, x) in enumerate(zip(labels, images)):
        print(f"Processing image {i+1}/{len(images)}")
        x = x.view(-1, cfg.MODEL.INPUT_IMAGE_CHANNELS * cfg.MODEL.INPUT_IMAGE_SIZE * cfg.MODEL.INPUT_IMAGE_SIZE)
        x = Variable(x.data, requires_grad=True).to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).float()
        x.retain_grad()
        z = E(x.view(-1, cfg.MODEL.INPUT_IMAGE_CHANNELS, cfg.MODEL.INPUT_IMAGE_SIZE, cfg.MODEL.INPUT_IMAGE_SIZE))
        z = z.squeeze()

        recon_batch = G(z.view(-1, cfg.MODEL.LATENT_SIZE, 1, 1))
        z = z.squeeze()

        if include_jacobian:
            J = compute_jacobian_autograd(x, z)
            J = J.cpu().numpy()

        z = z.cpu().detach().numpy()
        recon_batch = recon_batch.squeeze().cpu().detach().numpy()
        x = x.squeeze().cpu().detach().numpy()

        if include_jacobian:
            u, s, vh = np.linalg.svd(J, full_matrices=False)
            logD = -np.sum(np.log(np.abs(s)))
        else:
            logD = 0

        p = scipy.stats.gennorm.pdf(z, gennorm_param[0, :], gennorm_param[1, :], gennorm_param[2, :])
        logPz = np.sum(np.log(p))

        if not np.isfinite(logPz):
            logPz = -1000

        distance = np.linalg.norm(x.flatten() - recon_batch.flatten())
        logPe = logPe_func(distance)

        P = logD + logPz + logPe

        result = 'in' if threshold is not None and P > threshold else 'out'
        results.append(P if threshold is None else (1 if result == 'in' else 0))
        gt_novel.append(label in inliner_classes)

    return np.asarray(results, dtype=np.float32), np.asarray(gt_novel, dtype=np.float32)

def compute_threshold(images, labels, inliner_classes, cfg, counts, bin_edges, gennorm_param, E, G):
    logger = logging.getLogger("logger")
    y_scores, y_true = run_novely_prediction_on_images(images, labels, inliner_classes, cfg, counts, bin_edges, gennorm_param, threshold=None, E=E, G=G)
    
    y_scores = np.array([score for score in y_scores], dtype=np.float32)
    minP = min(y_scores) - 1
    maxP = max(y_scores) + 1
    y_false = np.logical_not(y_true)

    def evaluate(e):
        y = np.greater(y_scores, e)
        true_positive = np.sum(np.logical_and(y, y_true))
        false_positive = np.sum(np.logical_and(y, y_false))
        false_negative = np.sum(np.logical_and(np.logical_not(y), y_true))
        return get_f1(true_positive, false_positive, false_negative)

    best_th, best_f1 = find_maximum(evaluate, minP, maxP, 1e-4)

    logger.info("Best e: %f best f1: %f" % (best_th, best_f1))
    return best_th

def main_inference(inliner_classes, outliner_classes, cfg):
    logger = logging.getLogger("logger")

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.cuda.current_device()
    print("Running on ", torch.cuda.get_device_name(device))

    train_set, test_set = make_datasets(cfg, inliner_classes)

    print('Train set size: %d' % len(train_set))
    print('Test set size: %d' % len(test_set))

    train_set.shuffle()

    G = Generator(cfg.MODEL.LATENT_SIZE, channels=cfg.MODEL.INPUT_IMAGE_CHANNELS)
    E = Encoder(cfg.MODEL.LATENT_SIZE, channels=cfg.MODEL.INPUT_IMAGE_CHANNELS)

    G.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_FOLDER, "models/Gmodel.pkl")))
    E.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_FOLDER, "models/Emodel.pkl")))

    G.eval()
    E.eval()

    sample = torch.randn(64, cfg.MODEL.LATENT_SIZE).to(device)
    sample = G(sample.view(-1, cfg.MODEL.LATENT_SIZE, 1, 1)).cpu()
    save_image(sample.view(64, cfg.MODEL.INPUT_IMAGE_CHANNELS, cfg.MODEL.INPUT_IMAGE_SIZE, cfg.MODEL.INPUT_IMAGE_SIZE), 'sample.png')

    counts, bin_edges, gennorm_param = extract_statistics(cfg, train_set, E, G)

    inliner_images = [train_set[i][1] for i in range(5)]
    inliner_labels = [train_set[i][0] for i in range(5)]
    outliner_images = [test_set[i][1] for i in range(5)]
    outliner_labels = [test_set[i][0] for i in range(5)]

    threshold = compute_threshold(inliner_images, inliner_labels, inliner_classes, cfg, counts, bin_edges, gennorm_param, E, G)

    results_in, _ = run_novely_prediction_on_images(inliner_images, inliner_labels, inliner_classes, cfg, counts, bin_edges, gennorm_param, threshold, E, G)
    results_out, _ = run_novely_prediction_on_images(outliner_images, outliner_labels, inliner_classes, cfg, counts, bin_edges, gennorm_param, threshold, E, G)

    results = {
        'inliner_results': results_in,
        'outliner_results': results_out
    }

    return results
