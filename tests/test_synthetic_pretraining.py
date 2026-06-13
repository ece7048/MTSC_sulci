"""Synthetic 3D self-training smoke tests.

The production project trains large 3D MONAI/PyTorch models. These tests keep
the same objective shapes on tiny pure-Python models so they can run anywhere:
reconstruction, contrastive learning, diffusion-style denoising, and a small
GAN-style adversarial reconstruction loop.
"""

from __future__ import annotations

import math
import random
import unittest


def sigmoid(value):
    """Numerically stable logistic function for toy adversarial tests."""
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def make_sphere_volume(size=8, center=(4, 4, 4), radius=2.5):
    """Create a flattened 3D binary sphere volume."""
    volume = []
    for z in range(size):
        for y in range(size):
            for x in range(size):
                distance = math.sqrt(
                    (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
                )
                volume.append(1.0 if distance <= radius else 0.0)
    return volume


def add_noise(volume, rng, sigma=0.25):
    """Add clipped uniform noise to a flattened 3D volume."""
    noisy = []
    for value in volume:
        perturbed = value + rng.uniform(-sigma, sigma)
        noisy.append(min(1.0, max(0.0, perturbed)))
    return noisy


def local_mean(volume, index, size):
    """Compute the six-neighbour local mean for one flattened voxel."""
    z = index // (size * size)
    rem = index % (size * size)
    y = rem // size
    x = rem % size
    neighbours = []
    for dz, dy, dx in (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ):
        nz, ny, nx = z + dz, y + dy, x + dx
        if 0 <= nz < size and 0 <= ny < size and 0 <= nx < size:
            neighbours.append(volume[nz * size * size + ny * size + nx])
    return sum(neighbours) / len(neighbours)


def mean_value(volume):
    """Return the mean voxel intensity."""
    return sum(volume) / len(volume)


def center_of_mass_x(volume, size):
    """Return x-axis center-of-mass normalized to [0, 1]."""
    total = sum(volume) + 1e-8
    weighted = 0.0
    for index, value in enumerate(volume):
        x = index % size
        weighted += value * x
    return weighted / total / (size - 1)


def reconstruction_dataset(size=8, samples=12, seed=42, sigma=0.25):
    """Generate noisy/clean synthetic 3D reconstruction pairs."""
    rng = random.Random(seed)
    dataset = []
    centers = [(3, 3, 3), (4, 4, 4), (5, 4, 3), (3, 5, 4), (4, 3, 5)]
    for idx in range(samples):
        clean = make_sphere_volume(size=size, center=centers[idx % len(centers)], radius=2.2)
        noisy = add_noise(clean, rng, sigma=sigma)
        dataset.append((noisy, clean))
    return dataset


class TinyDenoisingPretrainer:
    """Tiny linear denoiser with intensity, local context, and noise-level terms."""

    def __init__(self):
        self.bias = 0.0
        self.intensity_weight = 0.2
        self.context_weight = 0.0
        self.noise_weight = 0.0

    def predict_one(self, noisy_value, context_value, noise_level=0.0):
        """Predict one clean voxel value from noisy intensity and context."""
        raw = (
            self.bias
            + self.intensity_weight * noisy_value
            + self.context_weight * context_value
            + self.noise_weight * noise_level
        )
        return min(1.0, max(0.0, raw))

    def reconstruct(self, noisy, size, noise_level=0.0):
        """Reconstruct a full flattened 3D volume."""
        return [
            self.predict_one(noisy[index], local_mean(noisy, index, size), noise_level)
            for index in range(len(noisy))
        ]

    def loss(self, dataset, size):
        """Return mean squared reconstruction error over a dataset."""
        total = 0.0
        count = 0
        for item in dataset:
            noisy, clean = item[:2]
            noise_level = item[2] if len(item) == 3 else 0.0
            pred = self.reconstruct(noisy, size, noise_level)
            for predicted, target in zip(pred, clean):
                total += (predicted - target) ** 2
                count += 1
        return total / count

    def train_epoch(self, dataset, size, lr=0.08):
        """Run one gradient-descent epoch over all synthetic volumes."""
        grad_bias = 0.0
        grad_intensity = 0.0
        grad_context = 0.0
        grad_noise = 0.0
        count = 0

        for item in dataset:
            noisy, clean = item[:2]
            noise_level = item[2] if len(item) == 3 else 0.0
            for index, target in enumerate(clean):
                intensity = noisy[index]
                context = local_mean(noisy, index, size)
                pred = self.predict_one(intensity, context, noise_level)
                error_grad = 2.0 * (pred - target)
                grad_bias += error_grad
                grad_intensity += error_grad * intensity
                grad_context += error_grad * context
                grad_noise += error_grad * noise_level
                count += 1

        self.bias -= lr * grad_bias / count
        self.intensity_weight -= lr * grad_intensity / count
        self.context_weight -= lr * grad_context / count
        self.noise_weight -= lr * grad_noise / count


class TinyContrastiveEncoder:
    """Tiny scalar encoder trained to group augmentations and separate subjects."""

    def __init__(self):
        self.bias = 0.0
        self.mean_weight = 0.05
        self.center_weight = 0.05

    def features(self, volume, size):
        """Extract toy global features from a 3D volume."""
        return mean_value(volume), center_of_mass_x(volume, size)

    def encode(self, volume, size):
        """Project a volume to a scalar embedding."""
        mean_feature, center_feature = self.features(volume, size)
        return self.bias + self.mean_weight * mean_feature + self.center_weight * center_feature

    def loss(self, pairs, size, margin=0.08):
        """Return positive-pair distance plus margin negative-pair loss."""
        total = 0.0
        for anchor, positive, negative in pairs:
            za = self.encode(anchor, size)
            zp = self.encode(positive, size)
            zn = self.encode(negative, size)
            pos = (za - zp) ** 2
            neg_distance = abs(za - zn)
            neg = max(0.0, margin - neg_distance) ** 2
            total += pos + neg
        return total / len(pairs)

    def train_epoch(self, pairs, size, lr=0.4, margin=0.08):
        """Update the scalar projection with finite-difference gradients."""
        base = (self.bias, self.mean_weight, self.center_weight)
        grads = []
        eps = 1e-4
        for attr_index, attr_name in enumerate(("bias", "mean_weight", "center_weight")):
            values = list(base)
            values[attr_index] += eps
            self.bias, self.mean_weight, self.center_weight = values
            high = self.loss(pairs, size, margin)

            values = list(base)
            values[attr_index] -= eps
            self.bias, self.mean_weight, self.center_weight = values
            low = self.loss(pairs, size, margin)
            grads.append((high - low) / (2 * eps))

        self.bias, self.mean_weight, self.center_weight = base
        self.bias -= lr * grads[0]
        self.mean_weight -= lr * grads[1]
        self.center_weight -= lr * grads[2]


class TinyDiscriminator:
    """Logistic discriminator over mean and local-contrast volume features."""

    def __init__(self):
        self.bias = 0.0
        self.mean_weight = 0.0
        self.contrast_weight = 0.0

    def features(self, volume, size):
        """Extract features that separate clean binary from smooth generated volumes."""
        mean_feature = mean_value(volume)
        contrast = sum(abs(value - local_mean(volume, index, size)) for index, value in enumerate(volume))
        return mean_feature, contrast / len(volume)

    def score(self, volume, size):
        """Return discriminator probability that a volume is real."""
        mean_feature, contrast_feature = self.features(volume, size)
        return sigmoid(self.bias + self.mean_weight * mean_feature + self.contrast_weight * contrast_feature)

    def train_epoch(self, clean_volumes, generated_volumes, size, lr=0.8):
        """Train discriminator with binary cross-entropy gradients."""
        grad_bias = 0.0
        grad_mean = 0.0
        grad_contrast = 0.0
        count = 0
        for label, volumes in ((1.0, clean_volumes), (0.0, generated_volumes)):
            for volume in volumes:
                mean_feature, contrast_feature = self.features(volume, size)
                pred = self.score(volume, size)
                grad = pred - label
                grad_bias += grad
                grad_mean += grad * mean_feature
                grad_contrast += grad * contrast_feature
                count += 1
        self.bias -= lr * grad_bias / count
        self.mean_weight -= lr * grad_mean / count
        self.contrast_weight -= lr * grad_contrast / count

    def accuracy(self, clean_volumes, generated_volumes, size):
        """Return binary discrimination accuracy."""
        correct = 0
        count = 0
        for volume in clean_volumes:
            correct += self.score(volume, size) >= 0.5
            count += 1
        for volume in generated_volumes:
            correct += self.score(volume, size) < 0.5
            count += 1
        return correct / count


def diffusion_dataset(size=8, samples=12, seed=7):
    """Generate reconstruction pairs at multiple noise levels."""
    rng = random.Random(seed)
    dataset = []
    for sigma in (0.1, 0.25, 0.4):
        for noisy, clean in reconstruction_dataset(size=size, samples=samples // 3, seed=seed, sigma=sigma):
            dataset.append((noisy, clean, sigma))
    rng.shuffle(dataset)
    return dataset


def contrastive_pairs(size=8, samples=10, seed=11):
    """Create anchor, positive augmentation, and negative augmentation triples."""
    rng = random.Random(seed)
    centers = [(2, 3, 3), (3, 3, 3), (4, 4, 4), (5, 4, 3), (6, 5, 4)]
    clean = [make_sphere_volume(size=size, center=center, radius=2.0) for center in centers]
    pairs = []
    for idx in range(samples):
        subject = idx % len(clean)
        negative = (subject + 2) % len(clean)
        pairs.append(
            (
                add_noise(clean[subject], rng, sigma=0.2),
                add_noise(clean[subject], rng, sigma=0.2),
                add_noise(clean[negative], rng, sigma=0.2),
            )
        )
    return pairs


class SyntheticPretrainingTest(unittest.TestCase):
    """Validate that toy 3D self-training objectives learn."""

    def test_reconstruction_loss_decreases(self):
        size = 8
        dataset = reconstruction_dataset(size=size, samples=12)
        model = TinyDenoisingPretrainer()

        initial_loss = model.loss(dataset, size)
        for _ in range(40):
            model.train_epoch(dataset, size)
        final_loss = model.loss(dataset, size)

        print(
            "reconstruction:",
            f"initial_loss={initial_loss:.6f}",
            f"final_loss={final_loss:.6f}",
            f"improvement={(initial_loss - final_loss) / initial_loss:.2%}",
        )
        self.assertLess(final_loss, initial_loss * 0.75)

    def test_contrastive_loss_decreases(self):
        size = 8
        pairs = contrastive_pairs(size=size)
        encoder = TinyContrastiveEncoder()

        initial_loss = encoder.loss(pairs, size)
        for _ in range(60):
            encoder.train_epoch(pairs, size)
        final_loss = encoder.loss(pairs, size)

        print(
            "contrastive:",
            f"initial_loss={initial_loss:.6f}",
            f"final_loss={final_loss:.6f}",
            f"improvement={(initial_loss - final_loss) / initial_loss:.2%}",
        )
        self.assertLess(final_loss, initial_loss * 0.75)

    def test_diffusion_denoising_loss_decreases(self):
        size = 8
        dataset = diffusion_dataset(size=size, samples=12)
        model = TinyDenoisingPretrainer()

        initial_loss = model.loss(dataset, size)
        for _ in range(45):
            model.train_epoch(dataset, size, lr=0.07)
        final_loss = model.loss(dataset, size)

        print(
            "diffusion:",
            f"initial_loss={initial_loss:.6f}",
            f"final_loss={final_loss:.6f}",
            f"improvement={(initial_loss - final_loss) / initial_loss:.2%}",
        )
        self.assertLess(final_loss, initial_loss * 0.75)

    def test_gan_generator_improves_with_adversarial_feedback(self):
        size = 8
        dataset = reconstruction_dataset(size=size, samples=10, seed=21)
        generator = TinyDenoisingPretrainer()
        discriminator = TinyDiscriminator()
        clean_volumes = [clean for _noisy, clean in dataset]

        initial_reconstruction = generator.loss(dataset, size)
        initial_generated = [generator.reconstruct(noisy, size) for noisy, _clean in dataset]
        initial_disc_acc = discriminator.accuracy(clean_volumes, initial_generated, size)

        for _ in range(30):
            generator.train_epoch(dataset, size, lr=0.06)
            generated = [generator.reconstruct(noisy, size) for noisy, _clean in dataset]
            discriminator.train_epoch(clean_volumes, generated, size)

        final_reconstruction = generator.loss(dataset, size)
        final_generated = [generator.reconstruct(noisy, size) for noisy, _clean in dataset]
        final_disc_acc = discriminator.accuracy(clean_volumes, final_generated, size)

        print(
            "gan:",
            f"initial_reconstruction={initial_reconstruction:.6f}",
            f"final_reconstruction={final_reconstruction:.6f}",
            f"reconstruction_improvement={(initial_reconstruction - final_reconstruction) / initial_reconstruction:.2%}",
            f"discriminator_accuracy={final_disc_acc:.2%}",
        )
        self.assertLess(final_reconstruction, initial_reconstruction * 0.8)
        self.assertGreaterEqual(final_disc_acc, initial_disc_acc)


if __name__ == "__main__":
    unittest.main()
