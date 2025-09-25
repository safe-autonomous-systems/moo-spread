import numpy as np


def random_perturbation(data, perturb_scale=0.05):
    perturb = (2 * np.random.rand(*data.shape) - 1) * perturb_scale
    return data + perturb


def interpolation(data, num_samples):
    interpolated_samples = []
    n = data.shape[0]
    for _ in range(num_samples):
        idx1, idx2 = np.random.choice(n, 2, replace=False)
        alpha = np.random.rand()
        interpolated_sample = alpha * data[idx1] + (1 - alpha) * data[idx2]
        interpolated_samples.append(interpolated_sample)
    return np.array(interpolated_samples)


def gaussian_noise(data, noise_scale=0.05):
    noise = np.random.normal(0, noise_scale, data.shape)
    return data + noise

def data_enhancement(offspringA, augmentation_factor=2, max_generation=32):
    """
    offspringA: np.array of shape [M, d]
    augmentation_factor: desired final size = M * augmentation_factor
    max_generation: upper‐limit on number of NEW (=augmented) samples
    """
    M = offspringA.shape[0]

    # how many augmented samples we'd like
    want = M * (augmentation_factor - 1)
    # but never more than max_generation
    augmented_needed = min(want, max_generation)

    # -- now proceed exactly as before, using augmented_needed --
    # generate candidates
    perturbed    = random_perturbation(offspringA)
    interpolated = interpolation(offspringA, augmented_needed // 3)
    noised       = gaussian_noise(offspringA)

    all_aug = np.vstack([perturbed, interpolated, noised])
    np.random.shuffle(all_aug)

    # slice down to the capped number
    all_aug = all_aug[:augmented_needed, :]

    # stack originals + capped augmented
    return np.vstack([offspringA, all_aug])