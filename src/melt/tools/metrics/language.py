from typing import Dict, List
import numpy as np
from .base import BaseMetric
from .basic_metrics import exact_match
from .utils import normalize_text
import evaluate
import math
import Levenshtein


class LanguageMetric(BaseMetric):
    """Evaluate language generation tasks."""

    def __init__(self, data, args) -> None:
        self.cer_metrics = evaluate.load("cer")
        self.wer_metrics = evaluate.load("wer")
        super().__init__(data, args)

    def get_num_bytes(self, tokens: List[str]) -> int:
        """Calculates the total number of bytes of a list of tokens
        when encoded in UTF-8.

        Args:
            tokens (List[str]): A list of string tokens for which the byte
            length is to be calculated.
        """
        num_bytes = 0
        for token in tokens:
            num_bytes += len(bytes(token, encoding="utf-8"))
        return num_bytes

    def evaluate(self, data: Dict, args) -> (Dict, Dict):
        """Evaluates the predictions against references and
        computes various metrics.

        Args:
            data (Dict): A dictionary that must contain keys
            "predictions", "references", and "generation_probs".
            It is used to store the predictions, the references for comparison,
            and the log probabilities for each prediction.

        Returns:
            Returns a tuple containing:
            - data: The original data dictionary, updated
            with raw metric scores
            for each prediction-reference pair.
            - result: A dictionary with the average scores of the metrics
            across all prediction-reference pairs.
        """
        predictions = data["predictions"]
        predictions = [self._get_answer(pred, args) for pred in predictions]
        references = data["references"]
        references = [normalize_text(ref) for ref in references]

        em_scores = [
            exact_match(pred, ref)
            for ref, pred in zip(references, predictions)
        ]
        cer_score = self.cer_metrics.compute(
            predictions=predictions, references=references
        )
        wer_score = self.wer_metrics.compute(
            predictions=predictions, references=references
        )

        ced_scores = [
            Levenshtein.distance(pred, ref)
            for pred, ref in zip(predictions, references)
        ]
        wed_scores = [
            Levenshtein.distance(
                np.array(pred.split(" ")), np.array(ref.split(" "))
            )
            for pred, ref in zip(predictions, references)
        ]

        perplexity_scores = []
        bits_per_byte = []
        logprob_per_byte = []
        for prediction, generation_prob in zip(
            data["predictions"], data["generation_probs"]
        ):
            logprob, num_perplexity_tokens, num_bytes = (
                np.array(generation_prob).sum(),
                len(generation_prob),
                self.get_num_bytes(prediction.split(" ")),
            )

            perplexity_scores.append(
                math.e ** (-logprob / num_perplexity_tokens)
            )
            bits_per_byte.append(-logprob / num_bytes / math.log(2))
            logprob_per_byte.append(logprob / num_bytes)

        data.update(
            {
                "average_exact_match": em_scores,
                "ced": ced_scores,
                "wed": wed_scores,
                "perplexity": perplexity_scores,
                "bits_per_byte": bits_per_byte,
                "logprob_per_byte": logprob_per_byte,
            }
        )
        result = {
            "average_exact_match": np.array(em_scores).mean(),
            "cer": cer_score,
            "wer": wer_score,
            "ced": np.array(ced_scores).mean(),
            "wed": np.array(wed_scores).mean(),
            "perplexity": np.array(perplexity_scores).mean(),
            "bits_per_byte": np.array(bits_per_byte).mean(),
            "logprob_per_byte": np.array(logprob_per_byte).mean(),
        }

        return data, result
