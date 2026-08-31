# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""BreizhCrops dataset."""

import os
from collections.abc import Callable, Sequence
from typing import ClassVar, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

from .errors import DatasetNotFoundError
from .geo import NonGeoDataset
from .utils import Path, Sample, download_url, extract_archive, lazy_import


class BreizhCrops(NonGeoDataset):
    """BreizhCrops dataset.

    The `BreizhCrops <https://github.com/dl4sits/BreizhCrops>`__ dataset is a
    benchmark for crop-type classification from Sentinel-2 time series in
    Brittany, France.

    Dataset features:

    * 608,263 L1C and 608,489 L2A parcel-level time series
    * 13 Sentinel-2 L1C or 10 Sentinel-2 L2A reflectance bands
    * 9 crop-type classes
    * spatially disjoint train, validation, and test splits
    * This implementation only supports the year 2017.

    Dataset format:

    * one CSV index and HDF5 database per region and processing level
    * one variable-length time series per agricultural parcel

    Dataset classes:

    0. Barley
    1. Wheat
    2. Rapeseed
    3. Corn
    4. Sunflower
    5. Orchards
    6. Nuts
    7. Permanent meadows
    8. Temporary meadows

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.5194/isprs-archives-XLIII-B2-2020/1545/2020

    This dataset requires the following additional library to be installed:

       * `h5py <https://pypi.org/project/h5py/>`_ to load the dataset

    .. versionadded:: 0.11
    """

    year = 2017
    base_url = 'https://breizhcrops.s3.eu-central-1.amazonaws.com'
    class_mapping_url = f'{base_url}/classmapping.csv'

    classes = ('barley', 'wheat', 'rapeseed', 'corn', 'sunflower') + (
        'orchards',
        'nuts',
        'permanent meadows',
        'temporary meadows',
    )

    # matches the split in the paper
    split_regions: ClassVar[dict[str, tuple[str, ...]]] = {
        'train': ('frh01', 'frh02'),  # Cotes-d'Armor, Finistere
        'val': ('frh03',),  # Ille-et-Vilaine
        'test': ('frh04',),  # Morbihan
    }

    bands_by_level: ClassVar[dict[str, tuple[str, ...]]] = {
        'L1C': ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7')
        + ('B8', 'B8A', 'B9', 'B10', 'B11', 'B12'),
        'L2A': ('B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'),
    }

    # internal HDF5 column order.
    h5_bands: ClassVar[dict[str, tuple[str, ...]]] = {
        'L1C': (
            ('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7')
            + ('B8', 'B8A', 'B9', 'B10', 'B11', 'B12')
            + ('QA10', 'QA20', 'QA60', 'doa')
        ),
        'L2A': (
            ('doa', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7')
            + ('B8', 'B8A', 'B11', 'B12', 'CLD', 'EDG', 'SAT')
        ),
    }

    file_sizes: ClassVar[dict[str, dict[str, int]]] = {
        'L1C': {
            'frh01': 2_559_635_960,
            'frh02': 2_253_658_856,
            'frh03': 2_493_572_704,
            'frh04': 1_555_075_632,
        },
        'L2A': {
            'frh01': 987_259_904,
            'frh02': 803_457_960,
            'frh03': 890_027_448,
            'frh04': 639_215_848,
        },
    }

    def __init__(
        self,
        root: Path = 'data',
        split: Literal['train', 'val', 'test'] = 'train',
        level: Literal['L1C', 'L2A'] = 'L1C',
        bands: Sequence[str] | None = None,
        transforms: Callable[[Sample], Sample] | None = None,
        download: bool = False,
    ) -> None:
        """Initialize a new BreizhCrops dataset instance.

        Args:
            root: Root directory where the dataset can be found.
            split: Dataset split to load. The train split contains FRH01 and
                FRH02, the validation split contains FRH03, and the test split
                contains FRH04.
            level: Sentinel-2 processing level.
            bands: Bands to load. If ``None``, all reflectance bands for the
                selected processing level are loaded.
            transforms: A function/transform that takes a sample dictionary and
                returns a transformed version.
            download: If True, download the dataset and store it in ``root``.

        Raises:
            ValueError: If ``split``, ``level``, or ``bands`` is invalid.
            DatasetNotFoundError: If the dataset is not found and ``download`` is
                False.
            DependencyNotFoundError: If h5py is not installed.

        """
        lazy_import('h5py')

        if split not in self.split_regions:
            raise ValueError(
                f'split must be one of {tuple(self.split_regions)}; got {split!r}'
            )

        if level not in self.bands_by_level:
            raise ValueError(
                f'level must be one of {tuple(self.bands_by_level)}; got {level!r}'
            )

        if bands is None:
            bands = self.bands_by_level[level]

        bands = tuple(bands)
        if not bands:
            raise ValueError('bands must not be empty')

        invalid_bands = set(bands) - set(self.bands_by_level[level])
        if invalid_bands:
            raise ValueError(f'Invalid bands for {level}: {sorted(invalid_bands)}')

        self.root = root
        self.split = split
        self.level = level
        self.bands = bands
        self.transforms = transforms
        self.download = download
        self.regions = self.split_regions[split]
        self.band_indices = [self.h5_bands[level].index(band) for band in self.bands]
        self.date_index = self.h5_bands[level].index('doa')

        self._verify()
        self.files = self._load_files()

    def __getitem__(self, index: int) -> Sample:
        """Return a parcel time series and its crop-type label.

        Args:
            index: Index of the sample to return.

        Returns:
            A sample containing the image time series, acquisition dates, label,
            and field ID.

        """
        row = self.files.iloc[index]
        h5py = lazy_import('h5py')

        with h5py.File(self._h5_path(row['region']), 'r') as file:
            array = np.asarray(file[row['h5_key']])

        image = torch.from_numpy(array[:, self.band_indices]).float()
        dates = torch.from_numpy(array[:, self.date_index]).long()
        sample = {
            'image': image,
            'dates': dates,
            'label': torch.tensor(row['label'], dtype=torch.long),
            'field_id': torch.tensor(row['field_id'], dtype=torch.long),
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of parcels in the selected split.

        Returns:
            The number of parcels.

        """
        return len(self.files)

    def _load_files(self) -> pd.DataFrame:
        """Load and combine the regional indexes for the selected split.

        Returns:
            Metadata for every parcel in the selected split.

        """
        mapping = pd.read_csv(self._class_mapping_path(), index_col='code')['id']
        indexes = []

        for region in self.regions:
            index = pd.read_csv(self._index_path(region))
            index = index[index['CODE_CULTU'].isin(mapping.index)].copy()
            index['label'] = index['CODE_CULTU'].map(mapping).astype(int)
            index['region'] = region
            indexes.append(
                index.rename(columns={'path': 'h5_key', 'id': 'field_id'})[
                    ['region', 'h5_key', 'field_id', 'label', 'sequencelength']
                ]
            )

        return pd.concat(indexes, ignore_index=True)

    def _verify(self) -> None:
        """Verify that files required by the selected split exist."""
        files_exist = os.path.exists(self._class_mapping_path()) and all(
            os.path.exists(self._index_path(region))
            and os.path.exists(self._h5_path(region))
            for region in self.regions
        )

        if files_exist:
            files_valid = all(self._h5_is_valid(region) for region in self.regions)
            if files_valid:
                return

        if not self.download:
            raise DatasetNotFoundError(self)

        self._download()

    def _download(self) -> None:
        """Download all regions belonging to the selected split."""
        download_url(self.class_mapping_url, self.root)

        directory = self._data_directory()
        for region in self.regions:
            download_url(self._index_url(region), directory)

            archive = f'{region}.h5.tar.gz'
            download_url(self._h5_url(region), directory, filename=archive)
            extract_archive(
                os.path.join(directory, archive), directory, remove_finished=True
            )

            if not self._h5_is_valid(region):
                raise RuntimeError('Download corrupted, file size mismatch.')

    def _h5_is_valid(self, region: str) -> bool:
        """Check a regional HDF5 file against its published file size."""
        path = self._h5_path(region)
        return (
            os.path.exists(path)
            and os.path.getsize(path) == self.file_sizes[self.level][region]
        )

    def _data_directory(self) -> Path:
        """Return the directory containing indexes and HDF5 files."""
        return os.path.join(self.root, str(self.year), self.level)

    def _class_mapping_path(self) -> Path:
        """Return the path to the shared class mapping."""
        return os.path.join(self.root, 'classmapping.csv')

    def _index_path(self, region: str) -> Path:
        """Return the path to a regional index."""
        return os.path.join(self._data_directory(), f'{region}.csv')

    def _h5_path(self, region: str) -> Path:
        """Return the path to a regional HDF5 database."""
        path = os.path.join(self._data_directory(), f'{region}.h5')
        if os.path.exists(path):
            return path

        return os.path.join(
            self._data_directory(),
            'data',
            'BreizhCrops',
            f'{self.level}_img',
            'data',
            str(self.year),
            self.level,
            f'{region}.h5',
        )

    def _index_url(self, region: str) -> str:
        """Return the URL of a regional index."""
        return f'{self.base_url}/{self.year}/{self.level}/{region}.csv'

    def _h5_url(self, region: str) -> str:
        """Return the URL of a regional HDF5 archive."""
        return f'{self.base_url}/{self.year}/{self.level}/{region}.h5.tar.gz'

    def plot(self, sample: Sample, suptitle: str | None = None) -> Figure:
        """Plot a parcel time series.

        Args:
            sample: A sample returned by :meth:`__getitem__`.
            suptitle: Optional string to use as a suptitle.

        Returns:
            A matplotlib Figure with each selected band plotted over time.

        """
        dates = pd.to_datetime(sample['dates'].numpy())
        image = sample['image'].numpy()
        label = sample['label'].item()
        field_id = sample['field_id'].item()

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(dates, image, '*-')
        ax.legend(self.bands, ncol=max(1, len(self.bands) // 3))
        ax.set_xlabel('Date of acquisition')
        ax.set_ylabel('Mean reflectance')
        ax.set_title(f'{self.level}: {self.classes[label]} (field ID: {field_id})')

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
