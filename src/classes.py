import torch

class DatasetSplit:
    """Represents a single dataset split (train/val/test) with both raw and processed data"""

    def __init__(self, name: str, tensor_dataset: torch.utils.data.TensorDataset):
        """
        Args:
            name: Split name ("train", "val", or "test")
            tensor_dataset: Processed TensorDataset containing data and labels
        """
        self.name = name
        self.tensor_dataset = tensor_dataset

    @property
    def data(self) -> torch.Tensor:
        """Get data tensor"""
        return self.tensor_dataset.tensors[0]

    @property
    def labels(self) -> torch.Tensor:
        """Get labels tensor"""
        return self.tensor_dataset.tensors[1]


class PreparedDataset:
    """Container for all three dataset splits (train/val/test)"""

    def __init__(self, train: DatasetSplit, val: DatasetSplit, test: DatasetSplit):
        """
        Args:
            train: Training split
            val: Validation split
            test: Test split
        """
        self.train = train
        self.val = val
        self.test = test

class SubsetDataset(torch.utils.data.Dataset):
    """A subset of a dataset at specified indices.

    Args:
        dataset (torch.utils.data.Dataset): The whole Dataset.
        indices (sequence): Indices in the whole set selected for subset.
    """
    def __init__(self, dataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx: int):
        return self.dataset[self.indices[idx]]

    def __len__(self) -> int:
        return len(self.indices)
