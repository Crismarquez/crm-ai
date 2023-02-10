from typing import List, Dict, Tuple
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config.config import DATA_DIR, TABLES_CONFIG
from vision_analytic.utils import create_bbdd, load_pickle


class CRMProcesor:
    """
    Manage conexion to data bases, query for face math and update register

    Methods
    -------
    query_embedding(embeddings: np.ndarray, threshold_score: float=0.7)
        search by similarity cosine and get the user id

    query_infoclient(self, id_user: str)
        consult info available of the consumer

    update_register(self, info_register: Dict)
        insert a new register in data bases

    """

    def __init__(self) -> None:
        self.users_embeddings_dir = DATA_DIR / "user_embeddings.pkl"
        self.info_users_dir = DATA_DIR / "info_users.pkl"
        self.load_bbdd()

    def load_bbdd(self) -> None:

        if not self.users_embeddings_dir.exists():
            create_bbdd(["user_embeddings"])

        if not self.info_users_dir.exists():
            create_bbdd(["info_users"])

        self.users_embeddings = load_pickle(self.users_embeddings_dir)
        self.info_users = load_pickle(self.info_users_dir)
        self.matrix_crm = [
            embbeding for embbeding in self.users_embeddings["embedding"].values
        ]

    def create_register(self, info_register: Dict) -> Dict:
        """Create initial register for a new user

        Args:
            info_register (Dict): {"column": [values]}

        Returns:
            Dict: status for update
        """

        cols_user_embeddings = TABLES_CONFIG["user_embeddings"]
        cols_info_users = TABLES_CONFIG["info_users"]

        new_register_embedding = {
            key: info_register[key] for key in cols_user_embeddings
        }

        new_register_info = {key: info_register[key] for key in cols_info_users}

        self.users_embeddings = pd.concat(
            [self.users_embeddings, pd.DataFrame.from_dict(new_register_embedding)]
        )

        self.info_users = pd.concat(
            [self.info_users, pd.DataFrame.from_dict(new_register_info)]
        )

        # save new bbdd
        self.users_embeddings.to_pickle(DATA_DIR / "user_embeddings.pkl")
        self.info_users.to_pickle(DATA_DIR / "info_users.pkl")

        # update matrix embedding
        self.matrix_crm = [
            embbeding for embbeding in self.users_embeddings["embedding"].values
        ]

        status_create = {
            "status": True,
            "cols_create": {
                "user_embedding_table": cols_user_embeddings,
                "user_info_table": cols_info_users,
            },
        }
        return status_create

    def query_embedding(
        self, embeddings: np.ndarray, threshold_score: float = 0.7
    ) -> List[Dict]:
        """Search in bbdd by embedding taking a threshold_score to decide if query is acceptable

        Args:
            embeddings (np.ndarray): 2D matrix where each vector represent the facial embedding
            for a given person.
            threshold_score (float, optional): Criterial for cosine similarity. Defaults to 0.7.

        Returns:
            List[Dict]: Each embedding generate a dict with info about the query
                {
                    "status": boolean. result for threshold_score acceptance
                    "id_user": number id if the user is fount
                    "score": score the cosine similarity
                }
        """
        query_resuls = []
        id_users, scores = self._predict(embeddings)

        for id_user, score in zip(id_users, scores):
            if score > threshold_score:
                status = True
            else:
                status = False
            query_resuls.append({"status": status, "id_user": id_user, "score": score})

        return query_resuls

    def query_infoclient(self, id_user: str) -> Dict:
        """
        Query for aditional information about the user

        Args:
            id_user (str): number identification for user

        Returns:
            Dict: {
                "register": {
                    "dimension": values
                    }
            }
        """
        query_result = self.info_users[self.info_users["id_user"] == id_user]
        query_result = query_result.to_dict(orient="index")

        return query_result

    def calculate_cosine_similarity(self, embeddings_input: np.ndarray) -> np.ndarray:
        """calculate cosine similarity between embeddings and bbdd-embeddings

        Args:
            embeddings_input (np.ndarray): matrix with embeddings to search

        Returns:
            np.ndarray: each embeddings_input with each bbdd-embeddings matrix
        """
        return cosine_similarity(embeddings_input, self.matrix_crm)

    def _predict(self, embeddings_input: np.ndarray) -> Tuple:

        # calculate distance
        matrix_similarity = self.calculate_cosine_similarity(embeddings_input)

        scores = matrix_similarity.max(axis=1)
        arg_max = matrix_similarity.argmax(axis=1)

        # get id_user
        id_users = []
        for max_value in arg_max:
            id_users.append(self.users_embeddings["id_user"].values[max_value])

        return (id_users, scores)
