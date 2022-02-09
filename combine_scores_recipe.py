from __future__ import annotations
from typing import Sequence

import numpy as np
from pineappleflow.core.artifacts.fold import FoldHolder
from pineappleflow.core.artifacts.matrix import MatrixHolder
from pineappleflow.core.artifacts.result import ModelResult
from pineappleflow.core.components.model_recipe import ModelRecipe
from typing import Tuple, Any, List

from pineapple.contrib.specs.model_recipe_specs import CombineScoresRecipeSpec
from pineapple.contrib.utils.stats.statistical_tests import specificity_threshold_interpolated
from pineappleflow.core.misc import are_exclusive_groups, get_object_from_path

import pandas as pd


class CombineScoresRecipe(ModelRecipe):
    __spec_cls__ = CombineScoresRecipeSpec

    @classmethod
    def from_spec(cls, spec: CombineScoresRecipeSpec) -> CombineScoresRecipe:
        return cls(
            chain_order=spec["chain_order"],
            spec_thresholds=spec["spec_thresholds"],
            combination_rule=spec["combination_rule"],
            combination_rule_kwargs=spec["combination_rule_kwargs"],
        )

    def __init__(
        self,
        chain_order: Sequence[str],
        spec_thresholds: Sequence[float],
        combination_rule: str,
        combination_rule_kwargs: dict,
    ) -> None:
        """
        Args:
            chain_order: [str], a list of transformer chain output names to process in sequence, should match chain names
                in the spec for the CombineChainsTransformer
            spec_thresholds: [float], a list of specificity thresholds to use for calling samples using each chain score
            combination_rule: str, a named method to use for combining sample calls into one final binary call.
            combination_rule_kwargs: kwargs required for the specific combination_rule used below.
        """
        self.chain_order = chain_order
        self.spec_thresholds = spec_thresholds
        self.combination_rule = combination_rule
        self.combination_rule_kwargs = combination_rule_kwargs
        self.calculated_cutoffs: List = []
        self.min_max_training_scores: List = []
        self.TOO_calls: Sequence[str] = []
        self.target_thresh: float

        if len(self.chain_order) != len(self.spec_thresholds):
            raise ValueError(
                f"Threshold list ({len(self.spec_thresholds)}) must equal number of transformer chains \
                             ({len(self.chain_order)})!"
            )

    def fit(self, fold_holder: FoldHolder) -> CombineScoresRecipe:
        """This model assumes that the CombinedTransformerChains transformer was used to generate a concatenated feature
        matrix containing different scores to use in combination for final binary classification.
        This model recipe is similar to SklearnFrozenStepwise but instead uses multiple specificity thresholds specified
        by the user to combine different transformer chains. The same feature can be used more than once in different
        transformer chains and combined for prediction as opposed to Stepwise which requires different feature names
        for each result. It is assumed that input labels are binary.
        """
        self.features = fold_holder.features
        X = np.hstack([fold_holder[feature].train.x for feature in fold_holder.features])
        col_names = [fold_holder[feature].train.column_metadata for feature in fold_holder.features]
        col_names = [n["transformed_by_chain"] for sublist in col_names for n in sublist]
        y = fold_holder.train.y
        y_cancer_type = np.array([m.cancer_type for m in fold_holder[fold_holder.features[0]].train.sample_metadata])
        train_cancer_types = [m.cancer_type for m in fold_holder[fold_holder.features[0]].train.sample_metadata]

        for chain_name in self.chain_order:
            if chain_name not in col_names:
                raise ValueError(f"Chain name {chain_name} not in the train column metadata.")

        if set(y) != set([0, 1]):
            raise ValueError(f"Binary labels must be 0 and 1. Got {set(y)}")

        # calculate spec cutoff for each score column
        self.calculated_cutoffs = []
        self.min_max_training_scores = []

        def get_threshold(chain_name: str, spec_threshold: float) -> Tuple[Any, List[Any]]:
            y = fold_holder.train.y
            y_scores = np.squeeze(X[:, col_names.index(chain_name)])  # works only on FIRST index of chain name
            spec_cutoff = specificity_threshold_interpolated(y_true=y, y_score=y_scores, specificity=spec_threshold)
            return spec_cutoff, y_scores

        for chain_name, threshold in zip(self.chain_order, self.spec_thresholds):
            spec_cutoff, y_scores = get_threshold(chain_name, threshold)
            self.calculated_cutoffs.append(spec_cutoff)
            self.min_max_training_scores.append((min(y_scores), max(y_scores)))

        if "min_train_TOO_quantile" in self.combination_rule_kwargs:
            mask = [ct in self.combination_rule_kwargs["cancer_types_for_quantile"] for ct in train_cancer_types]
            crc_scores = X[mask, 1]
            ncc_scores = np.max(X[mask, 2:], axis=1)
            self.min_train_TOO_quantile_value = np.quantile(
                crc_scores / ncc_scores, self.combination_rule_kwargs["min_train_TOO_quantile"]
            )

        if self.combination_rule == "linear_combo":
            probs_cancer = X[:, 1]  # model 1 crc prob from log reg on outlier scores.
            probs_crc = 1 - np.apply_along_axis(max, 1, X[:, 3:])  # 1 - model 2 max ncc prob.
            alpha = self.combination_rule_kwargs["tuning_param"]
            training_scores = np.array([(alpha * a) + ((1 - alpha) * b) for a, b in zip(probs_cancer, probs_crc)])
            self.target_thresh = np.percentile(training_scores[y == 0], self.combination_rule_kwargs["fold_spec"])

        if self.combination_rule == "filter_by_max_TOO_metaclassifier":
            self.classifier = get_object_from_path(self.combination_rule_kwargs["model"])(**self.combination_rule_kwargs["kwargs"])

            # for binomial regression, supply two lists of cancer types 0/1
            if len(self.combination_rule_kwargs["cancer_type_classes"]) == 2:
                unlist = [ct for sublist in self.combination_rule_kwargs["cancer_type_classes"] for ct in sublist]
                mask = [ct in unlist for ct in y_cancer_type]
                y_bin = ["Colorectal Cancer" if ct in self.combination_rule_kwargs["cancer_type_classes"][0] else "NCC" for ct in y_cancer_type[mask] ]
                self.classifier.fit(X[mask], y_bin)
            # for multinomial regression, supply one list of cancer types 1,2,3,4,5 etc
            else:
                mask = [ct in self.combination_rule_kwargs["cancer_type_classes"] for ct in y_cancer_type]
                self.classifier.fit(X[mask], y_cancer_type[mask])

        return self

    def predict(self, matrix_holder: MatrixHolder) -> ModelResult:
        """
        Make binary predictions based on thresholds calculated in fit. Combine calls according to
        self.combination_rule
        """
        # concatenate data across features
        X = np.hstack([matrix_holder[feature].x for feature in matrix_holder.features])
        score_col_names = [matrix_holder[feature].column_metadata for feature in matrix_holder.features]
        score_col_names = [n["transformed_by_chain"] for sublist in score_col_names for n in sublist]

        for chain_name in self.chain_order:
            if chain_name not in score_col_names:
                raise ValueError(f"Chain name {chain_name} not in the predict column metadata.")

        calls = []
        for chain_name, threshold in list(zip(self.chain_order, self.calculated_cutoffs)):
            scores = X[:, score_col_names.index(chain_name)]
            calls.append([1 if score > threshold else 0 for score in scores])

        df_calls = pd.DataFrame(calls).T
        df_calls.columns = self.chain_order

        final_calls = []
        final_scores = []
        self.TOO_calls = []
        if self.combination_rule == "filter_by_max_TOO":
            """
            Use model 1 to detect healthy samples (vs. CRC). Use subsequent model scores to determine TOO by max score.
            If TOO score for a different cancer is greater than CRC score, call as healthy. NOTE: Binary metric computer
            will not function as predicted using this rule.
            """
            for i, row in df_calls.iterrows():
                call_0 = row[self.chain_order[0]]
                # first col X[i,0] is the score from the outlier model
                crc_score = X[i, 1] # prob crc from the bi/multinomial
                non_crc_scores = X[i, 2:]  # probs ncc from the bi/multinomial
                if call_0 == 0:
                    final_scores.append(crc_score)
                    final_calls.append(0)
                    self.TOO_calls.append("Healthy")
                elif call_0 == 1:
                    # convert other cancers to negative call
                    if crc_score < max(non_crc_scores):
                        final_calls.append(0)
                        final_scores.append(max(non_crc_scores))
                    # label CRCs as positive
                    else:
                        final_calls.append(1)
                        final_scores.append(crc_score)
                    self.TOO_calls.append(np.argmax(X[i, 1:]))

        elif self.combination_rule == "filter_by_max_TOO_metaclassifier":
            '''
            Impose a model on the combined chain outputs. Use chain1 calls combined with 
            model output to make final calls
            '''

            y_class_scores = self.classifier.predict_proba(X)
            classes = list(self.classifier.classes_)

            for i, row in df_calls.iterrows():
                call_0 = row[self.chain_order[0]]
                # first col X[i,0] is the score from the CRC outlier model
                TOO_scores = y_class_scores[i, :]  # TOO probs from the bi/multinomial
                crc_score = TOO_scores[classes.index("Colorectal Cancer")]
                if call_0 == 0:
                    final_scores.append(crc_score)
                    final_calls.append(0)
                    self.TOO_calls.append("Healthy")
                elif call_0 == 1:
                    # convert other cancers to negative call
                    if crc_score < max(TOO_scores):
                        final_calls.append(0)
                        final_scores.append(max(TOO_scores))
                    # label CRCs as positive
                    else:
                        final_calls.append(1)
                        final_scores.append(crc_score)
                    self.TOO_calls.append(classes[np.argmax(TOO_scores)])

        elif self.combination_rule == "filter_by_TOO_ratio":
            """transformers should output one score for the first chain and N scores for the second chain. the first
            chain is used to call healthy/crc and the second chain is used for tissue of origin score. to call non-crc
            the TOO_ratio must be greater than the self.min_train_TOO_quantile_value.
            the assigned sample fold score is 0 or the TOO_ratio
            """
            # assign combined scores
            for i, row in df_calls.iterrows():
                call_0 = row[self.chain_order[0]]
                crc_too_score = X[i, 1]
                max_TOO_score = max(X[i, 2:])
                # TODO fix div by zero here
                TOO_ratio = crc_too_score / max_TOO_score
                if call_0 == 0:
                    final_scores.append(0)  # probability of CRC
                    self.TOO_calls.append("Healthy")
                elif call_0 == 1:
                    final_scores.append(TOO_ratio)  # higher scores >> call 1
                    self.TOO_calls.append(score_col_names[1:][np.argmax(X[i, 1:])])

            # assign fold calls
            final_calls = [1 if ratio >= self.min_train_TOO_quantile_value else 0 for ratio in final_scores]

        elif self.combination_rule == "linear_combo":
            """
            Final sample scores are taken as alpha pct. of score from model 1 and a 1-alpha pct. of model2.
            """
            test_probs_cancer = X[:, 1]  # model 1 crc prob from log reg on outlier scores.
            # probs_crc = X[:, 2] # model 2, prob. of crc vs. ncc.
            test_probs_crc = 1 - np.apply_along_axis(max, 1, X[:, 3:])  # 1 - model 2 max ncc prob.

            self.TOO_calls = np.apply_along_axis(lambda row: score_col_names[2:][np.argmax(row)], 1, X[:, 2:])
            alpha = self.combination_rule_kwargs["tuning_param"]
            final_scores = np.array(
                [(alpha * a) + ((1 - alpha) * b) for a, b in zip(test_probs_cancer, test_probs_crc)]
            )
            final_calls = [1 if s > self.target_thresh else 0 for s in final_scores]

        else:
            raise ValueError(f"Score combination rule is not specified. Got: {self.combination_rule}")

        final_scores = [np.array([1 - s, s]) for s in final_scores]

        return ModelResult(predictions=np.array(final_calls), scores=np.vstack(final_scores))
