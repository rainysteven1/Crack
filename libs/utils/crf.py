import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels


def apply_crf(shape: list, mask: np.ndarray):
    annotated_label = mask.astype(np.int32)

    _, labels = np.unique(annotated_label, return_inverse=True)
    n_labels = 2

    d = dcrf.DenseCRF2D(shape[1], shape[0], n_labels)

    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(
        sxy=(3, 3),
        compat=3,
        kernel=dcrf.DIAG_KERNEL,
        normalization=dcrf.NORMALIZE_SYMMETRIC,
    )

    # Run Inference for 10 steps
    Q = d.inference(10)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    return MAP.reshape(shape)
