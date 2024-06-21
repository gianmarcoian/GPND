import torch.utils.data
from torchvision.utils import save_image
from net import *
from torch.autograd import Variable
from utils.jacobian import compute_jacobian_autograd
import numpy as np
import logging
import os
import scipy.optimize
from dataloading import make_datasets, make_dataloader
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

def extract_statistics(cfg, train_set, inliner_classes, E, G):
    zlist = []
    rlist = []

    data_loader = make_dataloader(train_set, cfg.TEST.BATCH_SIZE, torch.cuda.current_device())

    for label, x in data_loader:
        print("Original x type:", x.dtype, "device:", x.device)  # Debugging statement
        x = x.view(-1, cfg.MODEL.INPUT_IMAGE_SIZE * cfg.MODEL.INPUT_IMAGE_SIZE)
        print("Reshaped x type:", x.dtype, "device:", x.device)  # Debugging statement
        x = x.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")).float()
        print("Converted x type:", x.dtype, "device:", x.device)  # Debugging statement

        z = E(x.view(-1, 1, cfg.MODEL.INPUT_IMAGE_SIZE, cfg.MODEL.INPUT_IMAGE_SIZE))
        recon_batch = G(z)
        z = z.squeeze()

        recon_batch = recon_batch.squeeze().cpu().detach().numpy()
        x = x.squeeze().cpu().detach().numpy()

        z = z.cpu().detach().numpy()

        for i in range(x.shape[0]):
            distance = np.linalg.norm(x[i].flatten() - recon_batch[i].flatten())
            rlist.append(distance)

        zlist.append(z)

    zlist = np.concatenate(zlist)

    counts, bin_edges = np.histogram(rlist, bins=30, density=True)

    if cfg.MAKE_PLOTS:
        plt.plot(bin_edges[1:], counts, linewidth=2)
        save_plot(r"Distance, $\left \|\| I - \hat{I} \right \|\|$",
                  'Probability density',
                  r"PDF of distance for reconstruction error, $p\left(\left \|\| I - \hat{I} \right \|\| \right)$",
                  cfg.OUTPUT_FOLDER + '/mnist_%s_reconstruction_error.pdf' % ("_".join([str(x) for x in inliner_classes])))

    for i in range(cfg.MODEL.LATENT_SIZE):
        plt.hist(zlist[:, i], bins='auto', histtype='step')

    if cfg.MAKE_PLOTS:
        save_plot(r"$z$",
                  'Probability density',
                  r"PDF of embedding $p\left(z \right)$",
                  cfg.OUTPUT_FOLDER + '/mnist_%s_embedding.pdf' % ("_".join([str(x) for x in inliner_classes])))

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

def run_novely_prediction_on_images(images, inliner_classes, cfg, counts, bin_edges, gennorm_param, threshold, E, G):
    device = torch.cuda.current_device()

    def logPe_func(x):
        N = (cfg.MODEL.INPUT_IMAGE_SIZE * cfg.MODEL.INPUT_IMAGE_SIZE - cfg.MODEL.LATENT_SIZE) * 1.0
        logC = loggamma(N / 2.0) - (N / 2.0) * np.log(2.0 * np.pi)
        return logC - (N - 1) * np.log(x) + np.log(r_pdf(x, bin_edges, counts))

    results = []
    gt_novel = []

    for label, image in images:
        x = image.view(-1, cfg.MODEL.INPUT_IMAGE_CHANNELS * cfg.MODEL.INPUT_IMAGE_SIZE * cfg.MODEL.INPUT_IMAGE_SIZE)
        x = Variable(x.data, requires_grad=True).to(device)

        z = E(x.view(-1, cfg.MODEL.INPUT_IMAGE_CHANNELS, cfg.MODEL.INPUT_IMAGE_SIZE, cfg.MODEL.INPUT_IMAGE_SIZE))
        recon_batch = G(z)
        z = z.squeeze()

        z = z.cpu().detach().numpy()
        recon_batch = recon_batch.squeeze().cpu().detach().numpy()
        x = x.squeeze().cpu().detach().numpy()

        distance = np.linalg.norm(x.flatten() - recon_batch.flatten())
        logPe = logPe_func(distance)

        p = scipy.stats.gennorm.pdf(z, gennorm_param[0, :], gennorm_param[1, :], gennorm_param[2, :])
        logPz = np.sum(np.log(p))

        if not np.isfinite(logPz):
            logPz = -1000

        P = logPe + logPz

        result = 'in' if P > threshold else 'out'
        results.append(result)
        gt_novel.append(label in inliner_classes)

    return results, gt_novel

def compute_threshold(valid_set, inliner_classes, percentage, cfg, counts, bin_edges, gennorm_param, E, G):
    print("Computing threshold...")  # Debug statement
    y_scores, y_true = run_novely_prediction_on_images(valid_set, inliner_classes, cfg, counts, bin_edges, gennorm_param, threshold=None, E=E, G=G)

    if y_scores is None or y_true is None:
        raise ValueError("y_scores or y_true are None")

    print("y_scores:", y_scores)  # Debug statement
    print("y_true:", y_true)  # Debug statement

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

    if best_th is None:
        raise ValueError("Threshold calculation failed")

    logger.info("Best e: %f best f1: %f" % (best_th, best_f1))
    return best_th


def main_inference(inliner_classes, outliner_classes, cfg):
    logger = logging.getLogger("logger")

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.cuda.current_device()
    print("Running on ", torch.cuda.get_device_name(device))

    train_set, test_set = make_datasets(cfg, inliner_classes)

    G = Generator(cfg.MODEL.LATENT_SIZE, channels=cfg.MODEL.INPUT_IMAGE_CHANNELS)
    E = Encoder(cfg.MODEL.LATENT_SIZE, channels=cfg.MODEL.INPUT_IMAGE_CHANNELS)

    G.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_FOLDER, "models/Gmodel.pkl")))
    E.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_FOLDER, "models/Emodel.pkl")))

    G.eval()
    E.eval()

    counts, bin_edges, gennorm_param = extract_statistics(cfg, train_set, inliner_classes, E, G)

    # Calcolo della threshold
    valid_set = [(label, image) for label, image in zip([test_set[i][0] for i in range(5)], [test_set[i][1] for i in range(5)])]  # Usare un subset del test_set come validazione per calcolare la soglia
    threshold = compute_threshold(valid_set, inliner_classes, cfg.DATASET.PERCENTAGES[0], cfg, counts, bin_edges, gennorm_param, E, G)

    # Eseguiamo l'inferenza su un set di immagini in-domain e out-of-domain
    inliner_images = [(test_set[i][0], test_set[i][1]) for i in range(5)]  # Prendiamo 5 immagini in-domain a caso
    outliner_images = [(MNISTDataset(cfg.DATASET.PATH_OUT, transform=transforms.ToTensor())[i][0], MNISTDataset(cfg.DATASET.PATH_OUT, transform=transforms.ToTensor())[i][1]) for i in range(5)]  # Prendiamo 5 immagini out-of-domain a caso

    inliner_results, inliner_truth = run_novely_prediction_on_images(inliner_images, inliner_classes, cfg, counts, bin_edges, gennorm_param, threshold, E, G)
    outliner_results, outliner_truth = run_novely_prediction_on_images(outliner_images, inliner_classes, cfg, counts, bin_edges, gennorm_param, threshold, E, G)

    print("Inliner Results: ", inliner_results)
    print("Outliner Results: ", outliner_results)

    return inliner_results, outliner_results
