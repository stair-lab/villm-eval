from typing import Dict, List
import numpy as np
from .base import BaseMetric
from ranx import Qrels, Run, evaluate as ranx_evaluate
from datasets import load_dataset


class InformationRetrievalMetric(BaseMetric):
    def __init__(self) -> None:
        pass

    def _get_qrel(self, references: List[Dict]) -> Qrels:
        relevant_dict = {}
        for reference in references:
            query_id = str(reference["id"])
            if query_id not in relevant_dict:
                relevant_dict[query_id] = {}
            for doc_id in reference["references"]:
                relevant_dict[query_id][str(doc_id)] = 1

        qrels = Qrels(relevant_dict)
        return qrels

    def _get_prob_from_log_prob(self,
                                score: float,
                                is_positive_predict: bool,
                                ) -> float:
        prob = np.exp(score)
        prob = 1 - prob if not is_positive_predict else prob
        return prob

    def _get_run(self,
                 predictions: List[Dict],
                 k: int,
                 args) -> Run:
        run_dict = {}
        for prediction in predictions:
            query_id = str(prediction["query_id"])
            if query_id not in run_dict:
                run_dict[query_id] = {}

            predict = self._get_answer(prediction["prediction"], args)
            is_positive_predict = predict == "yes"
            log_prob = (
                prediction["calib_probs"][0][0][0]
                if is_positive_predict
                else prediction["calib_probs"][1][0][0]
            )

            prob = self._get_prob_from_log_prob(log_prob, is_positive_predict)
            if len(run_dict[query_id]) < k:
                run_dict[query_id][str(prediction["passage_id"])] = prob

        run = Run(run_dict)
        return run

    def evaluate(self, data: Dict, args, **kwargs) -> (Dict, Dict):
        result = {}
        if "mmarco" in args.filepath:
            refenreces = load_dataset("json",
                                      data_files="./mmarco.json",
                                      split="train")
        else:
            refenreces = load_dataset(
                "json", data_files="./mrobust.json", split="train"
            )

        predictions = data["prediction"]

        qrels = self._get_qrel(refenreces)

        for mode in ["regular", "boosted"]:
            if mode == "regular":
                k = 30
            else:
                k = 9999
            run = self._get_run(predictions, k, args)
            result[f"{mode}_recall@10"] = ranx_evaluate(
                qrels, run, "recall@10", make_comparable=True
            )
            result[f"{mode}_precision@10"] = ranx_evaluate(
                qrels, run, "precision@10", make_comparable=True
            )
            result[f"{mode}_hit_rate@10"] = ranx_evaluate(
                qrels, run, "hit_rate@10", make_comparable=True
            )
            result[f"{mode}_mrr@10"] = ranx_evaluate(
                qrels, run, "mrr@10", make_comparable=True
            )
            result[f"{mode}_ndcg@10"] = ranx_evaluate(
                qrels, run, "ndcg@10", make_comparable=True
            )
            print(result)
        return data, result