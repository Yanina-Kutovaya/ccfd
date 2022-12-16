import logging
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["load_dataset"]

PATH = "https://storage.yandexcloud.net/ccfd-input-data/"

APPLICATION_HISTORY_URL = PATH + "applications_history.parquet.gzip"
BKI_URL = PATH + "bki.parquet.gzip"
CLIENT_PROFILE_URL = PATH + "client_profile.parquet.gzip"
PAYMENTS_URL = PATH + "payments.parquet.gzip"
SAMPLE_SUBMIT_URL = PATH + "sample_submit.parquet.gzip"
TEST_URL = PATH + "test.parquet.gzip"
TRAIN_URL = PATH + "train.parquet.gzip"


def load_data(
    applications_history_path: Optional[str] = None,
    bki_path: Optional[str] = None,
    client_profile_path: Optional[str] = None,
    payments_path: Optional[str] = None,
    sample_submit_path: Optional[str] = None,
    test_path: Optional[str] = None,
    train_path: Optional[str] = None,
) -> pd.DataFrame:

    if applications_history_path is None:
        applications_history_path = APPLICATION_HISTORY_URL
    logging.info(f"Reading applications_history from {applications_history_path}...")
    applications_history = pd.read_parquet(applications_history_path)

    if bki_path is None:
        bki_path = BKI_URL
    logging.info(f"Reading bki dataset from {bki_path}...")
    bki = pd.read_parquet(bki_path)

    if client_profile_path is None:
        client_profile_path = CLIENT_PROFILE_URL
    logging.info(f"Reading client_profile from {client_profile_path}...")
    client_profile = pd.read_parquet(client_profile_path)

    if payments_path is None:
        payments_path = PAYMENTS_URL
    logging.info(f"Reading payments dataset from {payments_path}...")
    payments = pd.read_parquet(payments_path)

    if sample_submit_path is None:
        sample_submit_path = SAMPLE_SUBMIT_URL
    logging.info(f"Reading sample_submit from {sample_submit_path}...")
    sample_submit = pd.read_parquet(sample_submit_path)

    if test_path is None:
        test_path = TEST_URL
    logging.info(f"Reading test dataset from {test_path}...")
    test = pd.read_parquet(test_path)

    if train_path is None:
        train_path = TRAIN_URL
    logging.info(f"Reading train dataset from {train_path}...")
    train = pd.read_parquet(train_path)

    return (
        applications_history,
        bki,
        client_profile,
        payments,
        sample_submit,
        test,
        train,
    )
