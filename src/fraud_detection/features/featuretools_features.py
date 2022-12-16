import logging
import pandas as pd
import featuretools as ft
from woodwork.logical_types import Categorical, AgeFractional, Ordinal, BooleanNullable
from src.fraud_detection.data.make_dataset import load_data

logger = logging.getLogger(__name__)

FEATURE_MATRIX_PATH = 'data/02_intermediate/feature_matrix.parquet.gzip'

def get_feature_matrix():
    es = ft.EntitySet(id="app")    
    es = add_dataframes(es)
    es = add_relationships(es)
    feature_matrix, feature_defs = generate_features(es)
    feature_matrix = select_features(feature_matrix, feature_defs)
    save_features(feature_matrix)

    return feature_matrix


def add_dataframes(es):
    (
        applications_history,
        bki,
        client_profile,
        payments,
        sample_submit,
        test,
        train,
    ) = load_data()

    logging.info(f"Adding dataframes to entity set...")

    app = pd.concat([train, test], ignore_index=True, sort=False)
    income_order = ["XNA", "low_action", "low_normal", "middle", "high"]
    es = es.add_dataframe(
        dataframe_name="applications",
        dataframe=app,
        index="application_number",
    )
    es = es.add_dataframe(
        dataframe_name="applications_history",
        dataframe=applications_history,
        index="prev_application_number",
        logical_types={
            "name_yield_group": Ordinal(order=income_order),
            "nflag_insured_on_approval": BooleanNullable,
        },
    )
    es = es.add_dataframe(
        dataframe_name="bki",
        dataframe=bki,
        index="index",
    )
    education_order = [
        "Lower secondary",
        "Secondary / secondary special",
        "Incomplete higher",
        "Higher education",
        "Academic degree",
    ]
    es = es.add_dataframe(
        dataframe_name="client_profile",
        dataframe=client_profile,
        index="index",
        logical_types={
            "education_level": Ordinal(order=education_order),
            "age": AgeFractional,
        },
    )
    es = es.add_dataframe(dataframe_name="payments", dataframe=payments, index="index")
    return es


def add_relationships(es):

    logging.info(f"Adding relationships to entity set...")

    es = es.add_relationship(
        "applications",
        "application_number",
        "applications_history",
        "application_number",
    )
    es = es.add_relationship(
        "applications", "application_number", "bki", "application_number"
    )
    es = es.add_relationship(
        "applications", "application_number", "client_profile", "application_number"
    )
    es = es.add_relationship(
        "applications", "application_number", "payments", "application_number"
    )
    es = es.add_relationship(
        "applications_history",
        "prev_application_number",
        "payments",
        "prev_application_number",
    )
    return es


def generate_features(es):

    logging.info(f"Generating features...")

    feature_matrix, feature_defs = ft.dfs(
        entityset=es,
        target_dataframe_name="applications",
        max_features=1000,
        chunk_size=4000,
        verbose=True,
        max_depth=3,
        n_jobs=-1,
    )
    return feature_matrix, feature_defs


def select_features(feature_matrix, feature_defs):

    logging.info(f"Selecting features...")

    feature_matrix, features = ft.selection.remove_single_value_features(
        feature_matrix, features=feature_defs
    )
    feature_matrix = ft.selection.remove_highly_null_features(feature_matrix)

    return feature_matrix


def save_features(feature_matrix, feature_matrix_path=None):

    logging.info(f"Saving feature_matrix...")

    if feature_matrix_path is None:
        feature_matrix_path = FEATURE_MATRIX_PATH

    feature_matrix.to_parquet(feature_matrix_path, compression='gzip')
