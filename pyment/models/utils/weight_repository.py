import os

from ...utils import download


# Pretrained weights hosted as GitHub blobs in estenhl/pyment-public.
# The historical Google Drive links for "brain-age" are dead (404);
# "brain-age" therefore resolves to the current reg-2025 checkpoint,
# which has the same SFCN topology and loads by layer order.
_GITHUB_BLOB_BASE = (
    'https://api.github.com/repos/estenhl/pyment-public/git/blobs'
)

_mapping = {
    ('RegressionSFCN', 'brain-age', True): {
        'url': f'{_GITHUB_BLOB_BASE}/a8baaed43082ebac70427a8bf122ffd7c63b51e2',
        'filename': 'regression_sfcn_reg_2025_weights.h5',
        'decode_github': True,
    },
    ('RegressionSFCN', 'reg-2025', True): {
        'url': f'{_GITHUB_BLOB_BASE}/a8baaed43082ebac70427a8bf122ffd7c63b51e2',
        'filename': 'regression_sfcn_reg_2025_weights.h5',
        'decode_github': True,
    },
}


_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)


class WeightRepository:
    root = os.path.join(_REPO_ROOT, 'output', 'pyment', 'models')

    @staticmethod
    def get_path(model: str, weights: str, include_top: bool):
        key = (model, weights, include_top)

        if key not in _mapping:
            raise ValueError(
                f"Weights '{weights}' does not exist for model {model} "
                f'with include_top={include_top}. '
                f'Available: {sorted(_mapping)}'
            )

        entry = _mapping[key]
        filename = entry['filename']
        path = os.path.join(WeightRepository.root, filename)

        if os.path.isfile(path) and not _is_hdf5(path):
            os.remove(path)

        if not os.path.isfile(path):
            if not os.path.isdir(WeightRepository.root):
                os.makedirs(WeightRepository.root)

            download(
                entry['url'],
                path,
                decode_github=entry.get('decode_github', False),
                expect_hdf5=True,
            )

        return path


def _is_hdf5(path: str) -> bool:
    with open(path, 'rb') as f:
        return f.read(8) == b'\x89HDF\r\n\x1a\n'
