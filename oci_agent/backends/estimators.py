# This module contains generally useful estimators.

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted


class XGBEarlyStoppingRegressor(BaseEstimator, RegressorMixin):
    """
    Class for gradient boosting regression with early stopping.
    """
    def __init__(
        self,
        base_estimator,
        validation_fraction=0.2,
        early_stopping_rounds=50,
        eval_metric=None,
        random_state=123,
        shuffle=True,
        fit_params=None,
    ):
        self.base_estimator = base_estimator
        self.validation_fraction = validation_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.shuffle = shuffle
        self.fit_params = fit_params

    def fit(self, X, y, sample_weight=None):
        X_train, X_val, y_train, y_val, sw_train, sw_val = self._split_data(
            X, y, sample_weight
        )

        self.estimator_ = clone(self.base_estimator)

        params = {}
        if self.eval_metric is not None:
            params["eval_metric"] = self.eval_metric
        if self.early_stopping_rounds is not None:
            params["early_stopping_rounds"] = self.early_stopping_rounds

        if params:
            self.estimator_.set_params(**params)

        fit_kwargs = {} if self.fit_params is None else dict(self.fit_params)
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = False

        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sw_train
            fit_kwargs["sample_weight_eval_set"] = [sw_val]

        self.estimator_.fit(X_train, y_train, **fit_kwargs)

        self.n_features_in_ = getattr(self.estimator_, "n_features_in_", None)
        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_
        if hasattr(self.estimator_, "feature_importances_"):
            self.feature_importances_ = self.estimator_.feature_importances_

        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    def _split_data(self, X, y, sample_weight):
        if sample_weight is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                shuffle=self.shuffle,
            )
            return X_train, X_val, y_train, y_val, None, None

        X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
            X,
            y,
            sample_weight,
            test_size=self.validation_fraction,
            random_state=self.random_state,
            shuffle=self.shuffle,
        )

        return X_train, X_val, y_train, y_val, sw_train, sw_val

    @property
    def best_iteration_(self):
        check_is_fitted(self, "estimator_")
        return getattr(self.estimator_, "best_iteration", None)

    @property
    def best_score_(self):
        check_is_fitted(self, "estimator_")
        return getattr(self.estimator_, "best_score", None)


class XGBEarlyStoppingClassifier(BaseEstimator, ClassifierMixin):
    """
    Class for gradient boosting classification with early stopping.
    """
    def __init__(
        self,
        base_estimator,
        validation_fraction=0.2,
        early_stopping_rounds=50,
        eval_metric=None,
        random_state=123,
        shuffle=True,
        stratify=True,
        fit_params=None,
    ):
        self.base_estimator = base_estimator
        self.validation_fraction = validation_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratify = stratify
        self.fit_params = fit_params

    def fit(self, X, y, sample_weight=None):
        stratify_arg = y if self.stratify else None

        X_train, X_val, y_train, y_val, sw_train, sw_val = self._split_data(
            X, y, sample_weight, stratify_arg
        )

        self.estimator_ = clone(self.base_estimator)

        params = {}
        if self.eval_metric is not None:
            params["eval_metric"] = self.eval_metric
        if self.early_stopping_rounds is not None:
            params["early_stopping_rounds"] = self.early_stopping_rounds

        if params:
            self.estimator_.set_params(**params)

        fit_kwargs = {} if self.fit_params is None else dict(self.fit_params)
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["verbose"] = False

        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sw_train
            fit_kwargs["sample_weight_eval_set"] = [sw_val]

        self.estimator_.fit(X_train, y_train, **fit_kwargs)

        self.classes_ = self.estimator_.classes_
        self.n_features_in_ = getattr(self.estimator_, "n_features_in_", None)
        if hasattr(self.estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.estimator_.feature_names_in_
        if hasattr(self.estimator_, "feature_importances_"):
            self.feature_importances_ = self.estimator_.feature_importances_

        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)

    def _split_data(self, X, y, sample_weight, stratify_arg):
        if sample_weight is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                shuffle=self.shuffle,
                stratify=stratify_arg,
            )
            return X_train, X_val, y_train, y_val, None, None

        X_train, X_val, y_train, y_val, sw_train, sw_val = train_test_split(
            X,
            y,
            sample_weight,
            test_size=self.validation_fraction,
            random_state=self.random_state,
            shuffle=self.shuffle,
            stratify=stratify_arg,
        )

        return X_train, X_val, y_train, y_val, sw_train, sw_val

    @property
    def best_iteration_(self):
        check_is_fitted(self, "estimator_")
        return getattr(self.estimator_, "best_iteration", None)

    @property
    def best_score_(self):
        check_is_fitted(self, "estimator_")
        return getattr(self.estimator_, "best_score", None)