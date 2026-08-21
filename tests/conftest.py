import glob
import os

import numpy as np
import pytest


@pytest.fixture
def model():
    path = 'model/grandmaster_resnet_v3.pt'
    if not os.path.exists(path):
        pytest.skip(f'artifact integration test requires {path}')
    from load_model import load_trained_model
    return load_trained_model(path)


@pytest.fixture
def rows():
    from tests import distill_ingest_test as ingest_test

    paths = glob.glob(
        os.path.join(ingest_test.DATA_DIR, 'train_*.parquet')
    )
    if not paths:
        pytest.skip(
            'artifact integration test requires distillation parquet shards'
        )
    return ingest_test._sample_rows(np.random.default_rng(7))
