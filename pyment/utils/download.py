import base64
import json
import logging
import math
import os

import requests
from tqdm import tqdm


logformat = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO)
logger = logging.getLogger(__name__)

_HDF5_MAGIC = b'\x89HDF\r\n\x1a\n'


def download(url: str, filename: str, chunksize: int = 2**16,
             *, decode_github: bool = False,
             expect_hdf5: bool = False) -> None:
    logger.info(f'Downloading {url} to {filename}')

    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    size = int(resp.headers.get('content-length')) \
           if 'content-length' in resp.headers else None

    if size is None:
        logger.warning(('Unable to get header \'content-length\'. '
                        'Downloading without progress bar'))

    tmp = f'{filename}.partial'
    try:
        with open(tmp, 'wb') as f:
            chunks = resp.iter_content(chunk_size=chunksize)
            total = math.ceil(size / chunksize) if size else None
            for chunk in tqdm(chunks, total=total):
                f.write(chunk)

        if decode_github:
            with open(tmp, 'rb') as f:
                payload = json.load(f)
            with open(tmp, 'wb') as f:
                f.write(base64.b64decode(payload['content']))

        if expect_hdf5:
            with open(tmp, 'rb') as f:
                magic = f.read(8)
            if magic != _HDF5_MAGIC:
                raise OSError(
                    f'Downloaded file is not a valid HDF5 weights file '
                    f'(got magic={magic!r}). URL may be dead or returning '
                    f'HTML: {url}'
                )

        os.replace(tmp, filename)
    except Exception:
        if os.path.isfile(tmp):
            os.remove(tmp)
        raise
