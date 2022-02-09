from __future__ import annotations

from pineappleflow.core.artifacts.matrix import Matrix
from pineappleflow.core.components.transformer import Transformer
from pineapple.contrib.specs.transformer_specs import LogRegByMetadataSpec
from pineapple.contrib.utils.metadata_extract_by_string_key import get_metadata_val_from_key_for_single_sample
from pineapple.core.misc import get_object_from_path
from pineappleflow.core.logging.logger import pprint_logger
from pineappleflow.core.experiment_context import get_context

import numpy as np


class LogRegByMetadata(Transformer):
    __spec_cls__ = LogRegByMetadataSpec

    @classmethod
    def from_spec(cls, spec: LogRegByMetadataSpec) -> LogRegByMetadata:
        return cls(**spec["kwargs"])

    def __init__(
        self,
        metadata_for_classes: str,
        class0: list,
        class1: list,
        class2: list,
        class3: list,
        class4: list,
        class5: list,
        class6: list,
        class7: list,
        class8: list,
        class9: list,
        class10: list,
        kwargs: dict,
        predictionfunc: str,
        class_for_output_score: str,
        reduce_methyl_feature: bool,
        reduce_methyl_feature_index: int,
    ) -> None:
        """
        Args:
            metadata_for_classes: sample metadata filed to be used for determining class lables (i.e. 'cancer_type')
            class0: labels to be assumed class negs (e.g. cancers other than CRC and AA)
            class1: labels to be assumed class positives (e.g. CRC and AA)
            classN: labels to be used for other classes in the multi-label case
            kwargs: parameters of the sklearn logistic regression classifier
            class_for_output_score: class name (e.g. "class1") to be used for returning a single probability score
            reduce_methyl_feature_: bool. whether to reduce the number of dimensions of the input data. used for methyl
            reduce_methyl_feature_index: CpG index to use for calculating HMF rates from the CHMFC feature. defaults to
                7, the expected input data is already thresholded by the PoissonOutlier transformer.
            predictionfunc: model attribute to use for generating samples scores from the model. defaults to
                sklearn predict_proba() which yields the probability of each class.
        """
        super().__init__()

        self.class0 = class0
        self.class1 = class1
        self.class2 = class2
        self.class3 = class3
        self.class4 = class4
        self.class5 = class5
        self.class6 = class6
        self.class7 = class7
        self.class8 = class8
        self.class9 = class9
        self.class10 = class10
        kwargs["random_state"] = get_context().random_state
        self.classifier = get_object_from_path("sklearn.linear_model.LogisticRegression")(**kwargs)
        self.predictionfunc = predictionfunc
        # -1 indicates return all class scores, else only for classN
        self.class_for_output_score = class_for_output_score
        self.metadata_for_classes = metadata_for_classes
        self.reduce_methyl_feature = reduce_methyl_feature
        self.reduce_methyl_feature_index = reduce_methyl_feature_index

    def fit(self, matrix: Matrix) -> LogRegByMetadata:
        """
        This transformer fits a logistic regression to sample classes defined by metadata fields. Multiple sample types
        can be combined into one class. Multi-class logistic regression is also supported. In the multi-
        class case, a single probability score or multiple scores can be returned at the end.
        """

        classes = [
            self.class0,
            self.class1,
            self.class2,
            self.class3,
            self.class4,
            self.class5,
            self.class6,
            self.class7,
            self.class8,
            self.class9,
            self.class10,
        ]
        classes_str = [
            "class0",
            "class1",
            "class2",
            "class3",
            "class4",
            "class5",
            "class6",
            "class7",
            "class8",
            "class9",
            "class10",
        ]

        # sample_labels = np.array(
        #     [x.raw_sample_metadata["lims_top_diagnosis"][self.metadata_for_classes] for x in matrix.sample_metadata]
        # )
        sample_labels = np.array(
            [get_metadata_val_from_key_for_single_sample(m, self.metadata_for_classes) for m in matrix.sample_metadata]
        )
        n_none = np.sum([s in [None, np.nan] for s in sample_labels])
        pprint_logger(
            name="log_reg_by_metadata",
            msg_list=[f"Missing metadata {self.metadata_for_classes} in {n_none} samples"],
            level="warn",
        )

        for sample_types in classes:
            for sample_type in sample_types:
                assert sample_type in sample_labels, f"No samples found for class type: {sample_type}"

        all_labels = [label for sublist in classes for label in sublist]
        unknown_labels = []
        for sample_label in sample_labels:
            if sample_label not in all_labels:
                unknown_labels.append(sample_label)
        pprint_logger(
            name="LogRegByMetadata",
            msg_list=[
                f"WARNING: Found {len(unknown_labels)} samples that do not fall into any logistic regression class."
                f"Labeled: {set(unknown_labels)}"
            ],
            level="warn",
        )

        indx_dict = {}
        for classN_str, classN in zip(classes_str, classes):
            indx_dict[classN_str] = [label in classN for label in sample_labels]

        X = matrix.x
        if self.reduce_methyl_feature is True:
            # reduce the 4 dimensional HMFC feature to HMF rates (2 dim)
            if len(X.shape) == 4:
                X = X[..., self.reduce_methyl_feature_index, 0] / (X[..., self.reduce_methyl_feature_index, 1] + 1)
            # reduce the PoissonOutlier fitted 3 dimensional HMFC feature to HMF rates
            elif len(X.shape) == 3:
                X = X[..., 0] / (X[..., 1] + 1)

        assert len(X.shape) == 2, f"Input array has wrong number of dims for fitting, got: {X.shape}"

        X_all = np.empty((0, X.shape[1]))

        y = []
        for classN_str in classes_str:
            mat = X[indx_dict[classN_str]]
            y.extend([classN_str] * len(mat))
            X_all = np.concatenate((X_all, mat), axis=0)

        self.classifier.fit(X_all, np.array(y))

        return self

    def transform(self, matrix: Matrix) -> Matrix:
        """
        Use the binomial or multinomial classifier from fit to obtain scores for test samples.
        Args:
            matrix: data to transform
        Returns:
            x (numpy array): (n_samples, N) one or more model scores (or probabilities) per sample
        """

        X = matrix.x

        if self.reduce_methyl_feature is True:
            # reduce the 4 dimensional HMFC feature to HMF rates (2 dim)
            if len(X.shape) == 4:
                X = X[..., self.reduce_methyl_feature_index, 0] / (X[..., self.reduce_methyl_feature_index, 1] + 1)
            # reduce the PoissonOutlier fitted 3 dimensional HMFC feature to HMF rates
            elif len(X.shape) == 3:
                X = X[..., 0] / (X[..., 1] + 1)

        assert len(X.shape) == 2, f"Input array has wrong number of dims for prediction, got: {X.shape}"

        predictions = getattr(self.classifier, self.predictionfunc)(X)

        if self.class_for_output_score == -1:
            # TODO create a MultiClassPassThroughModel Recipe to use with the output here
            y_hat = predictions
            col_meta = [{"name": classN_str} for classN_str in self.classifier.classes_]
            return matrix.replace_x_and_axis_metadata(
                x=np.reshape(y_hat, newshape=(-1, len(col_meta))),
                axis_metadata=(
                    matrix.row_metadata,
                    np.array(col_meta),
                ),
            )
        else:
            indx = list(self.classifier.classes_).index(self.class_for_output_score)
            y_hat = predictions[:, indx]
            col_meta = [{"name": self.class_for_output_score}]
            return matrix.replace_x_and_axis_metadata(
                x=np.reshape(y_hat, newshape=(-1, 1)),
                axis_metadata=(
                    matrix.row_metadata,
                    np.array(col_meta),
                ),
            )
