"""KNN Imputer for Missing Data"""
# Author: Ashim Bhattarai
# License: GNU General Public License v3 (GPLv3)

import warnings

import numpy as np
from scipy import stats
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.neighbors.base import _check_weights
from sklearn.neighbors.base import _get_weights

from .pairwise_external import pairwise_distances
from .pairwise_external import _get_mask
from .pairwise_external import _MASKED_METRICS

__all__ = [
    'KNNImputer',
]


class KNNImputer(BaseEstimator, TransformerMixin):
    """Imputation for completing missing values using k-Nearest Neighbors.

    Each sample's missing values are imputed using values from ``n_neighbors``
    nearest neighbors found in the training set. Each missing feature is then
    imputed as the average, either weighted or unweighted, of these neighbors.
    Note that if a sample has more than one feature missing, then the
    neighbors for that sample can be different depending on the particular
    feature being imputed. Finally, where the number of donor neighbors is
    less than ``n_neighbors``, the training set average for that feature is
    used during imputation.

    Parameters
    ----------
    missing_values : integer or "NaN", optional (default = "NaN")
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For missing values encoded as
        ``np.nan``, use the string value "NaN".

    n_neighbors : int, optional (default = 5)
        Number of neighboring samples to use for imputation.

    weights : str or callable, optional (default = "uniform")
        Weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

    metric : str or callable, optional (default = "masked_euclidean")
        Distance metric for searching neighbors. Possible values:
        - 'masked_euclidean'
        - [callable] : a user-defined function which conforms to the
        definition of _pairwise_callable(X, Y, metric, **kwds). In other
        words, the function accepts two arrays, X and Y, and a
        ``missing_values`` keyword in **kwds and returns a scalar distance
        value.

    row_max_missing : float, optional (default = 0.5)
        The maximum fraction of columns (i.e. features) that can be missing
        before the sample is excluded from nearest neighbor imputation. It
        means that such rows will not be considered a potential donor in
        ``fit()``, and in ``transform()`` their missing feature values will be
        imputed to be the column mean for the entire dataset.

    col_max_missing : float, optional (default = 0.8)
        The maximum fraction of rows (or samples) that can be missing
        for any feature beyond which an error is raised.

    copy : boolean, optional (default = True)
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible. Note that, if metric is
        "masked_euclidean" and copy=False then missing_values in the
        input matrix X will be overwritten with zeros.

    Attributes
    ----------
    statistics_ : 1-D array of length {n_features}
        The 1-D array contains the mean of each feature calculated using
        observed (i.e. non-missing) values. This is used for imputing
        missing values in samples that are either excluded from nearest
        neighbors search because they have too many ( > row_max_missing)
        missing features or because all of the sample's k-nearest neighbors
        (i.e., the potential donors) also have the relevant feature value
        missing.

    References
    ----------
    * Olga Troyanskaya, Michael Cantor, Gavin Sherlock, Pat Brown, Trevor
      Hastie, Robert Tibshirani, David Botstein and Russ B. Altman, Missing
      value estimation methods for DNA microarrays, BIOINFORMATICS Vol. 17
      no. 6, 2001 Pages 520-525.

    Examples
    --------
    >>> from missingpy import KNNImputer
    >>> nan = float("NaN")
    >>> X = [[1, 2, nan], [3, 4, 3], [nan, 6, 5], [8, 8, 7]]
    >>> imputer = KNNImputer(n_neighbors=2, weights="uniform")
    >>> imputer.fit_transform(X)
    array([[1. , 2. , 4. ],
           [3. , 4. , 3. ],
           [5.5, 6. , 5. ],
           [8. , 8. , 7. ]])
    """

    def __init__(self, missing_values="NaN", n_neighbors=5,
                 weights="uniform", metric="gower",n_jobs=1,
                 row_max_missing=0.5, col_max_missing=0.8, copy=True):

        self.missing_values = missing_values
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.row_max_missing = row_max_missing
        self.col_max_missing = col_max_missing
        self.copy = copy
        self.n_jobs = n_jobs

    def _impute(self, dist, X_imp, fitted_X, mask, mask_fx):
        """Helper function to find and impute missing values"""

        # For each column, find and impute
        X = X_imp.copy()
        n_rows_X, n_cols_X = X.shape
        for c in range(n_cols_X):
            if not np.any(mask.iloc[:, c], axis=0):
                continue

            # Row index for receivers and potential donors (pdonors)
            receivers_row_idx = np.where(mask.iloc[:, c])[0]
            pdonors_row_idx = np.where(~mask_fx.iloc[:, c])[0]

            # Get distance from potential donors
            dist_pdonors = dist[receivers_row_idx][:, pdonors_row_idx]
            dist_pdonors = dist_pdonors.reshape(-1,
                                                len(pdonors_row_idx))

            # Argpartition to separate actual donors from the rest
            pdonors_idx = np.argpartition(
                dist_pdonors, self.n_neighbors - 1, axis=1)

            # Get final donors row index from pdonors
            donors_idx = pdonors_idx[:, :self.n_neighbors]

            # Get weights or None
            dist_pdonors_rows = np.arange(len(donors_idx))[:, None]
            donor_row_idx_ravel = donors_idx.ravel()

            # Retrieve donor values and calculate kNN score
            fitted_X_temp = fitted_X.iloc[pdonors_row_idx].copy()
            donors = fitted_X_temp.iloc[donor_row_idx_ravel, c].values.reshape((-1, self.n_neighbors))

            # Final imputation
            if type(fitted_X_temp.iloc[0,c]) == str:
                s = list(set(donors.flatten()))
                dic1 = dict(zip(s, np.arange(len(s))))
                dic2 = dict(zip(np.arange(len(s)), s))
                imputed = np.vectorize(dic2.get)(stats.mode(np.vectorize(dic1.get)(donors), axis=1)[0].ravel())
            elif len(set(fitted_X_temp.iloc[:,c]))==2:
                imputed = stats.mode(donors, axis=1)[0].ravel()
            else:
                imputed = np.mean(donors, axis=1)
            X.iloc[receivers_row_idx, c] = imputed
        return X

    def fit(self, X, y=None):
        """Fit the imputer on X.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        self : object
            Returns self.
        """

        # Check if % missing in any column > col_max_missing
        mask = (X.isnull())*1
        if np.any(mask.sum(axis=0) > (X.shape[0] * self.col_max_missing)):
            raise ValueError("Some column(s) have more than {}% missing values"
                             .format(self.col_max_missing * 100))
        

        self.fitted_X_ = X

        return self

    def transform(self, X):
        """Impute all missing values in X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            The input data to complete.

        Returns
        -------
        X : {array-like}, shape = [n_samples, n_features]
            The imputed dataset.
        """

        # Get fitted data and ensure correct dimension
        n_rows_fit_X, n_cols_fit_X = self.fitted_X_.shape
        n_rows_X, n_cols_X = X.shape

        if n_cols_X != n_cols_fit_X:
            raise ValueError("Incompatible dimension between the fitted "
                             "dataset and the one to be transformed.")
        mask = (X.isnull())*1
        row_total_missing = mask.sum(axis=1)
        row_has_missing = row_total_missing.astype(np.bool)
        if not np.any(row_total_missing):
            return X

        if np.any(row_has_missing):

            # Mask for fitted_X
            mask_fx = self.fitted_X_.isnull()

            # Pairwise distances between receivers and fitted samples
            dist = np.empty((len(X), len(self.fitted_X_)))
            l = []
            for col in self.fitted_X_.columns:
                if type(self.fitted_X_.loc[~mask_fx[col], col].iloc[0]) == str:
                    l.append(0)
                else:
                    l.append(self.fitted_X_.loc[~mask_fx[col], col].max()-self.fitted_X_.loc[~mask_fx[col], col].min())
            dist[row_has_missing] = pairwise_distances(
                X[row_has_missing], self.fitted_X_, metric=self.metric, n_jobs=self.n_jobs,
                missing_values=self.missing_values, l=np.array(l))

            # Find and impute missing
            X = self._impute(dist, X, self.fitted_X_, mask, mask_fx)
            X_imp = pd.DataFrame()
            for col in X.columns:
                if type(X.loc[:, col].iloc[0]) == str:
                    X_imp = pd.concat([X_imp,pd.get_dummies(X[[col]])],1)
                else:
                    X_imp = pd.concat([X_imp,X[[col]]],1)

        return X_imp

    def fit_transform(self, X, y=None, **fit_params):
        """Fit KNNImputer and impute all missing values in X.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        X : {array-like}, shape (n_samples, n_features)
            Returns imputed dataset.
        """
        return self.fit(X).transform(X)
