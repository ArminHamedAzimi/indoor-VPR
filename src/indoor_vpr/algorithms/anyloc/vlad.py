from __future__ import annotations


class VLAD:
    """Hard-assignment VLAD with intra- and final L2 normalization."""

    def __init__(self, num_clusters: int, device: str) -> None:
        import torch

        self.torch = torch
        self.num_clusters = num_clusters
        self.device = torch.device(device)
        self.centers = None

    def load(self, vocabulary_path) -> None:
        centers = self.torch.load(vocabulary_path, map_location=self.device, weights_only=True)
        if centers.ndim != 2 or centers.shape[0] != self.num_clusters:
            raise ValueError(
                f"Vocabulary must have shape ({self.num_clusters}, descriptor_size); "
                f"received {tuple(centers.shape)}."
            )
        self.centers = centers.float().to(self.device)

    def fit(self, descriptors, iterations: int = 20, seed: int = 42) -> None:
        """Fit a cosine K-means vocabulary from normalized local descriptors."""

        torch = self.torch
        descriptors = torch.nn.functional.normalize(descriptors.float().to(self.device), dim=1)
        if len(descriptors) < self.num_clusters:
            raise ValueError(
                f"AnyLoc needs at least {self.num_clusters} local descriptors to fit VLAD."
            )
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(descriptors), generator=generator).to(self.device)
        centers = descriptors[indices[: self.num_clusters]].clone()
        for _ in range(iterations):
            labels = (descriptors @ centers.T).argmax(dim=1)
            updated = []
            for cluster in range(self.num_clusters):
                members = descriptors[labels == cluster]
                updated.append(centers[cluster] if len(members) == 0 else members.mean(dim=0))
            next_centers = torch.nn.functional.normalize(torch.stack(updated), dim=1)
            if torch.allclose(centers, next_centers, atol=1e-4):
                centers = next_centers
                break
            centers = next_centers
        self.centers = centers

    def generate(self, descriptors):
        if self.centers is None:
            raise RuntimeError("VLAD vocabulary is not fitted or loaded.")
        torch = self.torch
        functional = torch.nn.functional
        descriptors = functional.normalize(descriptors.float().to(self.device), dim=1)
        labels = (descriptors @ self.centers.T).argmax(dim=1)
        residual_blocks = []
        for cluster in range(self.num_clusters):
            residuals = descriptors[labels == cluster] - self.centers[cluster]
            block = torch.zeros_like(self.centers[cluster]) if len(residuals) == 0 else residuals.sum(0)
            residual_blocks.append(functional.normalize(block, dim=0))
        return functional.normalize(torch.cat(residual_blocks), dim=0)
