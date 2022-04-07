# This is a modified version of the Inter Subject Correlation standalone implementation
# The major change is the addition of functions for calculating inter-subject
# edge/node time-series (IS-NTS/ETS):
#   * IS-NTS - [isc_ets]
#   * IS-ETS - [isfc_ets]
# Author: Gidon Levakov
# Ben Gurion University, 2022
# Both modified and and unmodified part of the code are distributed with the
# original license that appears bellow:

#  Copyright 2017 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Intersubject correlation (ISC) analysis

Functions for computing intersubject correlation (ISC) and related
analyses (e.g., intersubject funtional correlations; ISFC), as well
as statistical tests designed specifically for ISC analyses. This
"standalone" module is specifically intended to replicate the BrainIAK
(https://brainiak.org) ISC functionality without requiring an
installation of BrainIAK.

The implementation is based on the work in [Hasson2004]_, [Kauppi2014]_,
[Simony2016]_, and [Chen2016]_.

.. [Chen2016] "Untangling the relatedness among correlations, part I:
   nonparametric approaches to inter-subject correlation analysis at the
   group level.", G. Chen, Y. W. Shin, P. A. Taylor, D. R. Glen, R. C.
   Reynolds, R. B. Israel, R. W. Cox, 2016, NeuroImage, 142, 248-259.
   https://doi.org/10.1016/j.neuroimage.2016.05.023

.. [Hasson2004] "Intersubject synchronization of cortical activity
   during natural vision.", U. Hasson, Y. Nir, I. Levy, G. Fuhrmann,
   R. Malach, 2004, Science, 303, 1634-1640.
   https://doi.org/10.1126/science.1089506

.. [Kauppi2014] "A versatile software package for inter-subject
   correlation based analyses of fMRI.", J. P. Kauppi, J. Pajula,
   J. Tohka, 2014, Frontiers in Neuroinformatics, 8, 2.
   https://doi.org/10.3389/fninf.2014.00002

.. [Simony2016] "Dynamic reconfiguration of the default mode network
   during narrative comprehension.", E. Simony, C. J. Honey, J. Chen, O.
   Lositsky, Y. Yeshurun, A. Wiesel, U. Hasson, 2016, Nature Communications,
   7, 12141. https://doi.org/10.1038/ncomms12141
"""

# Authors: Sam Nastase, Christopher Baldassano, Qihong Lu,
#          Mai Nguyen, and Mor Regev
# Princeton University, 2018

import numpy as np
import logging
from scipy.spatial.distance import squareform
from scipy.fftpack import fft, ifft
import itertools as it
import nibabel as nib
from nibabel.spatialimages import SpatialImage
from pathlib import Path
from typing import Callable, Iterable, Sequence, Type, TypeVar, Union
import itertools

logger = logging.getLogger(__name__)

__all__ = [
    "array_correlation",
    "array_edge_time_series",
    "array_edge_isc",
    "bootstrap_isc",
    "compute_summary_statistic",
    "isfc",
    "isfc_ets",
    "isc",
    "isc_ets",
    "permutation_isc",
    "phaseshift_isc",
    "squareform_isfc",
    "timeshift_isc",
]


MAX_RANDOM_SEED = 2**32 - 1

def isc_ets(data, pairwise=False, tolerate_nans=True):
    """Intersubject node time-series (IS-NTS)

    For each voxel or ROI, compute the edge time-series between each
    subject's response time series and other subjects' corresponding
    ROI/voxel time series.
    If pairwise is False (default), use the leave-one-out approach, where
    correlation is computed between each subject and the average of the other
    subjects. If pairwise is True, compute correlations between all pairs of
    subjects. Input data should be a n_TRs by n_voxels by
    n_subjects array (e.g., brainiak.image.MaskedMultiSubjectData) or a list
    where each item is a n_TRs by n_voxels ndarray for a given subject.
    Multiple input ndarrays must be the same shape. If a 2D array is supplied,
    the last dimension is assumed to correspond to subjects. If only two
    subjects are supplied, simply compute Pearson correlation (precludes
    averaging in leave-one-out approach, and does not apply summary statistic).
    When using leave-one-out approach, NaNs are ignored when computing mean
    time series of N-1 subjects (default: tolerate_nans=True). Alternatively,
    you may supply a float between 0 and 1 indicating a threshold proportion
    of N subjects with non-NaN values required when computing the average time
    series for a given voxel. For example, if tolerate_nans=.8, ISCs will be
    computed for any voxel where >= 80% of subjects have non-NaN values,
    while voxels with < 80% non-NaN values will be assigned NaNs. If set to
    False, NaNs are not tolerated and voxels with one or more NaNs among the
    N-1 subjects will be assigned NaN. Setting tolerate_nans to True or False
    will not affect the pairwise approach; however, if a threshold float is
    provided, voxels that do not reach this threshold will be excluded. Note
    that accommodating NaNs may be notably slower than setting tolerate_nans to
    False. Output shape is: n_subjects, n_tr, n_nodes

    The implementation is based on the work in [Levakov2022]_.

    Parameters
    ----------
    data : list or ndarray (n_TRs x n_voxels x n_subjects)
        fMRI data for which to compute ISC

    pairwise : bool, default: False
        Whether to use pairwise (True) or leave-one-out (False) approach

    tolerate_nans : bool or float, default: True
        Accommodate NaNs (when averaging in leave-one-out approach)

    Returns
    -------
    iscs : subjects' IS-NTS ndarray
        shape: n_subjects, n_tr, n_nodes
    """

    # Check response time series input format
    data, n_TRs, n_voxels, n_subjects = _check_timeseries_input(data)

    # No summary statistic if only two subjects
    if n_subjects == 2:
        logger.info("Only two subjects! Simply computing Pearson correlation.")

    # Check tolerate_nans input and use either mean/nanmean and exclude voxels
    if tolerate_nans:
        mean = np.nanmean
    else:
        mean = np.mean
    data, mask = _threshold_nans(data, tolerate_nans)

    # Compute correlation for only two participants
    if n_subjects == 2:
        # Compute correlation for each corresponding voxel
        iscs_stack = array_edge_isc(data[..., 0],
                                       data[..., 1])[np.newaxis, :]

    # Compute pairwise ISCs using voxel loop and corrcoef for speed
    elif pairwise:
        iscs_stack = pairwise_edge_isc(data)
        # n_tr * n_subjects * n_subjects * n_voxels

    # Compute leave-one-out ISCs
    elif not pairwise:

        # Loop through left-out subjects
        iscs_stack = []
        for s in np.arange(n_subjects):

            # Correlation between left-out subject and mean of others
            iscs_stack.append(array_edge_isc(
                data[..., s],
                mean(np.delete(data, s, axis=2), axis=2)))

        iscs_stack = np.array(iscs_stack)

    # Get ISCs back into correct shape after masking out NaNs
    iscs = np.full(iscs_stack.shape[:-1] + (n_voxels,), np.nan)
    iscs[..., np.where(mask)[0]] = iscs_stack

    # Throw away first dimension if singleton
    if iscs.shape[0] == 1:
        iscs = iscs[0,...]
    return iscs

def isc(data, pairwise=False, summary_statistic=None, tolerate_nans=True):
    """Intersubject correlation
    For each voxel or ROI, compute the Pearson correlation between each
    subject's response time series and other subjects' response time series.
    If pairwise is False (default), use the leave-one-out approach, where
    correlation is computed between each subject and the average of the other
    subjects. If pairwise is True, compute correlations between all pairs of
    subjects. If summary_statistic is None, return N ISC values for N subjects
    (leave-one-out) or N(N-1)/2 ISC values for each pair of N subjects,
    corresponding to the upper triangle of the pairwise correlation matrix
    (see scipy.spatial.distance.squareform). Alternatively, use either
    'mean' or 'median' to compute summary statistic of ISCs (Fisher Z will
    be applied if using mean). Input data should be a n_TRs by n_voxels by
    n_subjects array (e.g., brainiak.image.MaskedMultiSubjectData) or a list
    where each item is a n_TRs by n_voxels ndarray for a given subject.
    Multiple input ndarrays must be the same shape. If a 2D array is supplied,
    the last dimension is assumed to correspond to subjects. If only two
    subjects are supplied, simply compute Pearson correlation (precludes
    averaging in leave-one-out approach, and does not apply summary statistic).
    When using leave-one-out approach, NaNs are ignored when computing mean
    time series of N-1 subjects (default: tolerate_nans=True). Alternatively,
    you may supply a float between 0 and 1 indicating a threshold proportion
    of N subjects with non-NaN values required when computing the average time
    series for a given voxel. For example, if tolerate_nans=.8, ISCs will be
    computed for any voxel where >= 80% of subjects have non-NaN values,
    while voxels with < 80% non-NaN values will be assigned NaNs. If set to
    False, NaNs are not tolerated and voxels with one or more NaNs among the
    N-1 subjects will be assigned NaN. Setting tolerate_nans to True or False
    will not affect the pairwise approach; however, if a threshold float is
    provided, voxels that do not reach this threshold will be excluded. Note
    that accommodating NaNs may be notably slower than setting tolerate_nans to
    False. Output is an ndarray where the first dimension is the number of
    subjects or pairs and the second dimension is the number of voxels (or
    ROIs). If only two subjects are supplied or a summary statistic is invoked,
    the output is a ndarray n_voxels long.

    The implementation is based on the work in [Hasson2004]_.

    Parameters
    ----------
    data : list or ndarray (n_TRs x n_voxels x n_subjects)
        fMRI data for which to compute ISC

    pairwise : bool, default: False
        Whether to use pairwise (True) or leave-one-out (False) approach

    summary_statistic : None or str, default: None
        Return all ISCs or collapse using 'mean' or 'median'

    tolerate_nans : bool or float, default: True
        Accommodate NaNs (when averaging in leave-one-out approach)

    Returns
    -------
    iscs : subjects or pairs by voxels ndarray
        ISC for each subject or pair (or summary statistic) per voxel

    """

    # Check response time series input format
    data, n_TRs, n_voxels, n_subjects = _check_timeseries_input(data)

    # No summary statistic if only two subjects
    if n_subjects == 2:
        logger.info("Only two subjects! Simply computing Pearson correlation.")
        summary_statistic = None

    # Check tolerate_nans input and use either mean/nanmean and exclude voxels
    if tolerate_nans:
        mean = np.nanmean
    else:
        mean = np.mean
    data, mask = _threshold_nans(data, tolerate_nans)

    # Compute correlation for only two participants
    if n_subjects == 2:

        # Compute correlation for each corresponding voxel
        iscs_stack = array_correlation(data[..., 0],
                                       data[..., 1])[np.newaxis, :]

    # Compute pairwise ISCs using voxel loop and corrcoef for speed
    elif pairwise:

        # Swap axes for np.corrcoef
        data = np.swapaxes(data, 2, 0)

        # Loop through voxels
        voxel_iscs = []
        for v in np.arange(data.shape[1]):
            voxel_data = data[:, v, :]

            # Correlation matrix for all pairs of subjects (triangle)
            iscs = squareform(np.corrcoef(voxel_data), checks=False)
            voxel_iscs.append(iscs)

        iscs_stack = np.column_stack(voxel_iscs)

    # Compute leave-one-out ISCs
    elif not pairwise:

        # Loop through left-out subjects
        iscs_stack = []
        for s in np.arange(n_subjects):

            # Correlation between left-out subject and mean of others
            iscs_stack.append(array_correlation(
                data[..., s],
                mean(np.delete(data, s, axis=2), axis=2)))

        iscs_stack = np.array(iscs_stack)

    # Get ISCs back into correct shape after masking out NaNs
    iscs = np.full((iscs_stack.shape[0], n_voxels), np.nan)
    iscs[:, np.where(mask)[0]] = iscs_stack

    # Summarize results (if requested)
    if summary_statistic:
        iscs = compute_summary_statistic(iscs,
                                         summary_statistic=summary_statistic,
                                         axis=0)[np.newaxis, :]

    # Throw away first dimension if singleton
    if iscs.shape[0] == 1:
        iscs = iscs[0]

    return iscs


def isfc(data, targets=None, pairwise=False, summary_statistic=None,
         vectorize_isfcs=True, tolerate_nans=True, use_all_targets = False,
         weights = None):

    """Intersubject functional correlation (ISFC)

    For each input voxel or ROI, compute the Pearson correlation between each
    subject's response time series and all input voxels or ROIs in other
    subjects. If a targets array is provided, instead compute ISFCs between
    each input voxel time series and each voxel time series in targets across
    subjects (resulting in asymmetric ISFC values). The targets array must have
    the same number TRs and subjects as the input data. If pairwise is False
    (default), use the leave-one-out approach, where correlation is computed
    between each subject and the average of the other subjects. If pairwise is
    True, compute correlations between all pairs of subjects. If a targets
    array is provided, only the leave-one-out approach is supported. If
    summary_statistic is None, return N ISFC values for N subjects (leave-one-
    out) or N(N-1)/2 ISFC values for each pair of N subjects, corresponding to
    the triangle of the correlation matrix (scipy.spatial.distance.squareform).
    Alternatively, use either 'mean' or 'median' to compute summary statistic
    of ISFCs (Fisher Z is applied if using mean). Input should be n_TRs by
    n_voxels by n_subjects array (e.g., brainiak.image.MaskedMultiSubjectData)
    or a list where each item is a n_TRs by n_voxels ndarray per subject.
    Multiple input ndarrays must be the same shape. If a 2D array is supplied,
    the last dimension is assumed to correspond to subjects. If only two
    subjects are supplied, simply compute ISFC between these two subjects
    (precludes averaging in leave-one-out approach, and does not apply summary
    statistic). Returns vectorized upper triangle of ISFC matrices for each
    subject or pair when vectorized_isfcs=True, or full (redundant) 2D ISFC
    matrices when vectorized_isfcs=False. When using leave-one-out approach,
    NaNs are ignored when computing mean time series of N-1 subjects (default:
    tolerate_nans=True). Alternatively, you may supply a float between 0 and
    1 indicating a threshold proportion of N subjects with non-NaN values
    required when computing the average time series for a given voxel. For
    example, if tolerate_nans=.8, ISCs will be computed for any voxel where
    >= 80% of subjects have non-NaN values, while voxels with < 80% non-NaN
    values will be assigned NaNs. If set to False, NaNs are not tolerated
    and voxels with one or more NaNs among the N-1 subjects will be assigned
    NaN. Setting tolerate_nans to True or False will not affect the pairwise
    approach; however, if a threshold float is provided, voxels that do not
    reach this threshold will be excluded. Note that accommodating NaNs may
    be notably slower than setting tolerate_nans to False. Output is either
    a tuple comprising condensed off-diagonal ISFC values and the diagonal
    ISC values if vectorize_isfcs=True, or a single ndarray with shape
    n_subjects (or n_pairs) by n_voxels by n_voxels 3D array if
    vectorize_isfcs=False (see brainiak.isc.squareform_isfc). If targets array
    is provided (yielding asymmetric ISFCs), output ISFCs are not vectorized,
    resulting in an n_subjects by n_voxels by n_targets ISFC array. If
    summary_statistic is supplied, output is collapsed along first dimension.

    The implementation is based on the work in [Simony2016]_.

    Parameters
    ----------
    data : list or ndarray (n_TRs x n_voxels x n_subjects)
        fMRI data for which to compute ISFC

    targets : list or ndarray (n_TRs x n_voxels x n_subjects), optional
        fMRI data to use as targets for ISFC

    pairwise : bool, default: False
        Whether to use pairwise (True) or leave-one-out (False) approach

    summary_statistic : None or str, default: None
        Return all ISFCs or collapse using 'mean' or 'median'

    vectorize_isfcs : bool, default: True
        Return tuple of condensed ISFCs and ISCs (True) or square (redundant)
        ISFCs (False)

    tolerate_nans : bool or float, default: True
        Accommodate NaNs (when averaging in leave-one-out approach)

    Returns
    -------
    isfcs : ndarray or tuple of ndarrays
        ISFCs for each subject or pair (or summary statistic) per voxel pair

    """

    # Check response time series input format
    data, n_TRs, n_voxels, n_subjects = _check_timeseries_input(data)

    # Check for optional targets input array
    targets, t_n_TRs, t_n_voxels, t_n_subejcts, symmetric = (
        _check_targets_input(targets, data, use_all_targets = use_all_targets))
    if not symmetric:
        pairwise = False

    # Check tolerate_nans input and use either mean/nanmean and exclude voxels
    if tolerate_nans:
        mean = np.nanmean
    else:
        mean = np.mean
    data, mask = _threshold_nans(data, tolerate_nans)
    targets, targets_mask = _threshold_nans(targets, tolerate_nans)

    if weights is None:
        corr_func = np.corrcoef
        args = {}
        slices_1 = slice(None,t_n_voxels)
        slices_2 = slice(t_n_voxels,None)
    else:
        corr_func = array_isfc_weighted
        args = {'weights': weights}
        slices_1 = slice(None)
        slices_2 = slice(None)

    # Handle just two subjects properly (for symmetric approach)
    if symmetric and n_subjects == 2:
        isfcs = corr_func(
                    np.ascontiguousarray(data[..., 0].T),
                    np.ascontiguousarray(data[..., 1].T), **args)[slices_1,
                                                          slices_2]
        isfcs = (isfcs + isfcs.T) / 2
        isfcs = isfcs[..., np.newaxis]
        summary_statistic = None
        logger.info("Only two subjects! Computing ISFC between them.")

    # Compute all pairwise ISFCs (only for symmetric approach)
    elif pairwise:
        isfcs = []
        for pair in it.combinations(np.arange(n_subjects), 2):
            isfc_pair = corr_func(np.ascontiguousarray(
                                        data[..., pair[0]].T),
                                    np.ascontiguousarray(
                                        targets[...,
                                                pair[1]].T), **args)[slices_1,
                                                             slices_2]
            if symmetric:
                isfc_pair = (isfc_pair + isfc_pair.T) / 2
            isfcs.append(isfc_pair)
        isfcs = np.dstack(isfcs)

    # Compute ISFCs using leave-one-out approach
    else:

        # Roll subject axis for loop
        data = np.rollaxis(data, 2, 0)
        targets = np.rollaxis(targets, 2, 0)

        # Compute leave-one-out ISFCs
        if use_all_targets:
            isfcs = [corr_func(np.ascontiguousarray(subject.T),
                                 np.ascontiguousarray(mean(targets,
                                        axis=0).T), **args)[slices_1,
                                                    slices_2]
                     for s, subject in enumerate(data)]
        else:
            isfcs = [corr_func(np.ascontiguousarray(subject.T),
                                 np.ascontiguousarray(mean(
                                        np.delete(targets, s, axis=0),
                                        axis=0).T), **args)[slices_1,
                                                    slices_2]
                     for s, subject in enumerate(data)]

        # Transpose and average ISFC matrices for both directions
        isfcs = np.dstack([(isfc_matrix + isfc_matrix.T) / 2 if
                           symmetric else isfc_matrix for
                           isfc_matrix in isfcs])

    # Get ISCs back into correct shape after masking out NaNs
    isfcs_all = np.full((n_voxels, t_n_voxels, isfcs.shape[2]), np.nan)
    isfcs_all[np.ix_(np.where(mask)[0], np.where(targets_mask)[0])] = isfcs
    isfcs = np.moveaxis(isfcs_all, 2, 0)

    # Summarize results (if requested)
    if summary_statistic:
        isfcs = compute_summary_statistic(isfcs,
                                          summary_statistic=summary_statistic,
                                          axis=0)

    # Throw away first dimension if singleton
    if isfcs.shape[0] == 1:
        isfcs = isfcs[0]

    # Optionally squareform to vectorize ISFC matrices (only if symmetric)
    if vectorize_isfcs and symmetric:
        isfcs, iscs = squareform_isfc(isfcs)
        return isfcs, iscs
    else:
        return isfcs

def isfc_ets(data, targets=None, pairwise=False, tolerate_nans=True, use_all_targets = False):

    """Intersubject edge time-series (IS-ETS)

    For each input voxel or ROI, compute the edge time-series between each
    subject's response time series and all input voxels or ROIs in other
    subjects. If a targets array is provided, instead compute IS-ETSs between
    each input voxel time series and each voxel time series in targets across
    subjects (resulting in asymmetric IS-ETS values). The targets array must have
    the same number TRs and subjects as the input data. Currently only implemented
    for pairwise=False, meaning a leave-one-out approach, where correlation is
    computed between each subject and the average of the other subjects.
    Input should be n_TRs by n_voxels by n_subjects array (e.g.,
    brainiak.image.MaskedMultiSubjectData) or a list where each item is a n_TRs
    by n_voxels ndarray per subject.
    Multiple input ndarrays must be the same shape. If a 2D array is supplied,
    the last dimension is assumed to correspond to subjects. If only two
    subjects are supplied, simply compute IS-ETS between these two subjects
    (precludes averaging in leave-one-out approach, and does not apply summary
    statistic). The function returns the full (redundant) 3D IS-ETS arrays:
    n_subjects * n_trs * n_nodes * n_nodes

    The implementation is based on the work in [Levakov2022]_.

    Parameters
    ----------
    data : list or ndarray (n_TRs x n_voxels x n_subjects)
        fMRI data for which to compute ISFC

    targets : list or ndarray (n_TRs x n_voxels x n_subjects), optional
        fMRI data to use as targets for ISFC

    pairwise : bool, default: False
        Whether to use pairwise (True - not implemented) or leave-one-out
        (False) approach

    tolerate_nans : bool or float, default: True
        Accommodate NaNs (when averaging in leave-one-out approach)

    Returns
    -------
    isfcs : ndarray or tuple of ndarrays
        IS-ETSs for each subject - [n_subjects * n_trs * n_nodes * n_nodes]

    """

    # Check response time series input format
    data, n_TRs, n_voxels, n_subjects = _check_timeseries_input(data)

    # Check for optional targets input array
    targets, t_n_TRs, t_n_voxels, t_n_subejcts, symmetric = (
        _check_targets_input(targets, data, use_all_targets = use_all_targets))
    if not symmetric:
        pairwise = False

    # Check tolerate_nans input and use either mean/nanmean and exclude voxels
    if tolerate_nans:
        mean = np.nanmean
    else:
        mean = np.mean
    data, mask = _threshold_nans(data, tolerate_nans)
    targets, targets_mask = _threshold_nans(targets, tolerate_nans)

    # Handle just two subjects properly (for symmetric approach)
    if symmetric and n_subjects == 2:
        isfcs = array_edge_time_series(
                    np.ascontiguousarray(data[..., 0]),
                    np.ascontiguousarray(data[..., 1]))
        isfcs = (isfcs + np.transpose(isfcs,(0,2,1))) / 2
        isfcs = isfcs[..., np.newaxis]
        summary_statistic = None
        logger.info("Only two subjects! Computing ISFC between them.")

    # Compute all pairwise ISFCs (only for symmetric approach)
    elif pairwise:
        raise NotImplementedError('Pairwise edges-time series is not implemented')

    # Compute ISFCs using leave-one-out approach
    else:

        # Roll subject axis for loop
        data = np.rollaxis(data, 2, 0)
        targets = np.rollaxis(targets, 2, 0)

        # Compute leave-one-out ISFCs
        if use_all_targets:
            isfcs = [array_edge_time_series(np.ascontiguousarray(subject),
                                 np.ascontiguousarray(mean(targets,
                                        axis=0)))
                     for s, subject in enumerate(data)]
        else:
            isfcs = [array_edge_time_series(np.ascontiguousarray(subject),
                                 np.ascontiguousarray(mean(
                                        np.delete(targets, s, axis=0),
                                        axis=0)))
                     for s, subject in enumerate(data)]

        # Transpose and average ISFC matrices for both directions
        try:
            isfcs = np.stack([(isfc_matrix + np.transpose(isfc_matrix,(0,2,1))) / 2 if
                               symmetric else isfc_matrix for
                               isfc_matrix in isfcs], axis=-1)
        except Exception as e:
            print(e, ', converting to float32.')
            isfcs = [isfc.astype(np.float32) for isfc in isfcs]
            isfcs = np.stack([(isfc_matrix + np.transpose(isfc_matrix,(0,2,1))) / 2 if
                               symmetric else isfc_matrix for
                               isfc_matrix in isfcs], axis=-1)

    # Get ISCs back into correct shape after masking out NaNs
    isfcs_all = np.full((n_TRs, n_voxels, t_n_voxels, isfcs.shape[-1]), np.nan)
    isfcs_all[np.ix_(np.arange(n_TRs),np.where(mask)[0], np.where(targets_mask)[0])] = isfcs
    isfcs = np.moveaxis(isfcs_all, -1, 0)

    # Throw away first dimension if singleton
    if isfcs.shape[0] == 1:
        isfcs = isfcs[0]

    return isfcs

def _check_isc_input(iscs, pairwise=False):

    """Checks ISC inputs for statistical tests

    Input ISCs should be n_subjects (leave-one-out approach) or
    n_pairs (pairwise approach) by n_voxels or n_ROIs array or a 1D
    array (or list) of ISC values for a single voxel or ROI. This
    function is only intended to be used internally by other
    functions in this module (e.g., bootstrap_isc, permutation_isc).

    Parameters
    ----------
    iscs : ndarray or list
        ISC values

    Returns
    -------
    iscs : ndarray
        Array of ISC values

    n_subjects : int
        Number of subjects

    n_voxels : int
        Number of voxels (or ROIs)
    """

    # Standardize structure of input data
    if type(iscs) == list:
        iscs = np.array(iscs)[:, np.newaxis]

    elif isinstance(iscs, np.ndarray):
        if iscs.ndim == 1:
            iscs = iscs[:, np.newaxis]

    # Check if incoming pairwise matrix is vectorized triangle
    if pairwise:
        try:
            test_square = squareform(iscs[:, 0])
            n_subjects = test_square.shape[0]
        except ValueError:
            raise ValueError("For pairwise input, ISCs must be the "
                             "vectorized triangle of a square matrix.")
    elif not pairwise:
        n_subjects = iscs.shape[0]

    # Infer subjects, voxels and print for user to check
    n_voxels = iscs.shape[1]
    logger.info("Assuming {0} subjects with and {1} "
                "voxel(s) or ROI(s) in bootstrap ISC test.".format(n_subjects,
                                                                   n_voxels))

    return iscs, n_subjects, n_voxels


def _check_targets_input(targets, data, use_all_targets = False):

    """Checks ISFC targets input array

    For ISFC analysis, targets input array should either be a list
    of n_TRs by n_targets arrays (where each array corresponds to
    a subject), or an n_TRs by n_targets by n_subjects ndarray. This
    function also checks the shape of the targets array against the
    input data array.

    Parameters
    ----------
    data : list or ndarray (n_TRs x n_voxels x n_subjects)
        fMRI data for which to compute ISFC

    targets : list or ndarray (n_TRs x n_voxels x n_subjects)
        fMRI data to use as targets for ISFC

    Returns
    -------
    targets : ndarray (n_TRs x n_voxels x n_subjects)
        ISFC targets with standadized structure

    n_TRs : int
        Number of time points (TRs) for targets array

    n_voxels : int
        Number of voxels (or ROIs) for targets array

    n_subjects : int
        Number of subjects for targets array

    symmetric : bool
        Indicator for symmetric vs. asymmetric
    """

    if isinstance(targets, np.ndarray) or isinstance(targets, list):
        targets, n_TRs, n_voxels, n_subjects = (
            _check_timeseries_input(targets))
        if data.shape[0] != n_TRs:
            raise ValueError("Targets array must have same number of "
                             "TRs as input data")
        if (data.shape[2] != n_subjects) and not use_all_targets:
            raise ValueError("Targets array must have same number of "
                             "subjects as input data")
        symmetric = False
    else:
        targets = data
        n_TRs, n_voxels, n_subjects = data.shape
        symmetric = True

    return targets, n_TRs, n_voxels, n_subjects, symmetric


def compute_summary_statistic(iscs, summary_statistic='mean', axis=None):

    """Computes summary statistics for ISCs

    Computes either the 'mean' or 'median' across a set of ISCs. In the
    case of the mean, ISC values are first Fisher Z transformed (arctanh),
    averaged, then inverse Fisher Z transformed (tanh).

    The implementation is based on the work in [SilverDunlap1987]_.

    .. [SilverDunlap1987] "Averaging corrlelation coefficients: should
       Fisher's z transformation be used?", N. C. Silver, W. P. Dunlap, 1987,
       Journal of Applied Psychology, 72, 146-148.
       https://doi.org/10.1037/0021-9010.72.1.146

    Parameters
    ----------
    iscs : list or ndarray
        ISC values

    summary_statistic : str, default: 'mean'
        Summary statistic, 'mean' or 'median'

    axis : None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.

    Returns
    -------
    statistic : float or ndarray
        Summary statistic of ISC values

    """

    if summary_statistic not in ('mean', 'median'):
        raise ValueError("Summary statistic must be 'mean' or 'median'")

    # Compute summary statistic
    if summary_statistic == 'mean':
        statistic = np.tanh(np.nanmean(np.arctanh(iscs), axis=axis))
    elif summary_statistic == 'median':
        statistic = np.nanmedian(iscs, axis=axis)

    return statistic


def squareform_isfc(isfcs, iscs=None):

    """Converts square ISFCs to condensed ISFCs (and ISCs), and vice-versa

    If input is a 2- or 3-dimensional array of square ISFC matrices, converts
    this to the condensed off-diagonal ISFC values (i.e., the vectorized
    triangle) and the diagonal ISC values. In this case, input must be a
    single array of shape either n_voxels x n_voxels or n_subjects (or
    n_pairs) x n_voxels x n_voxels. The condensed ISFC values are vectorized
    according to scipy.spatial.distance.squareform, yielding n_voxels *
    (n_voxels - 1) / 2 values comprising every voxel pair. Alternatively, if
    input is an array of condensed off-diagonal ISFC values and an array of
    diagonal ISC values, the square (redundant) ISFC values are returned.
    This function mimics scipy.spatial.distance.squareform, but is intended
    to retain the diagonal ISC values.

    Parameters
    ----------
    isfcs : ndarray
        Either condensed or redundant ISFC values

    iscs: ndarray, optional
        Diagonal ISC values, required when input is condensed

    Returns
    -------
    isfcs : ndarray or tuple of ndarrays
        If condensed ISFCs are passed, a single redundant ISFC array is
        returned; if redundant ISFCs are passed, both a condensed off-
        diagonal ISFC array and the diagonal ISC values are returned
    """

    # Check if incoming ISFCs are square (redundant)
    if not type(iscs) == np.ndarray and isfcs.shape[-2] == isfcs.shape[-1]:
        if isfcs.ndim == 2:
            isfcs = isfcs[np.newaxis, ...]
        if isfcs.ndim == 3:
            iscs = np.diagonal(isfcs, axis1=1, axis2=2)
            isfcs = np.vstack([squareform(isfc, checks=False)[np.newaxis, :]
                               for isfc in isfcs])
        else:
            raise ValueError("Square (redundant) ISFCs must be square "
                             "with multiple subjects or pairs of subjects "
                             "indexed by the first dimension")
        if isfcs.shape[0] == iscs.shape[0] == 1:
            isfcs, iscs = isfcs[0], iscs[0]
        return isfcs, iscs

    # Otherwise, convert from condensed to redundant
    else:
        if isfcs.ndim == iscs.ndim == 1:
            isfcs, iscs = isfcs[np.newaxis, :], iscs[np.newaxis, :]
        isfcs_stack = []
        for isfc, isc in zip(isfcs, iscs):
            isfc_sq = squareform(isfc, checks=False)
            np.fill_diagonal(isfc_sq, isc)
            isfcs_stack.append(isfc_sq[np.newaxis, ...])
        isfcs = np.vstack(isfcs_stack)
        if isfcs.shape[0] == 1:
            isfcs = isfcs[0]
        return isfcs


def _threshold_nans(data, tolerate_nans):

    """Thresholds data based on proportion of subjects with NaNs

    Takes in data and a threshold value (float between 0.0 and 1.0) determining
    the permissible proportion of subjects with non-NaN values. For example, if
    threshold=.8, any voxel where >= 80% of subjects have non-NaN values will
    be left unchanged, while any voxel with < 80% non-NaN values will be
    assigned all NaN values and included in the nan_mask output. Note that the
    output data has not been masked and will be same shape as the input data,
    but may have a different number of NaNs based on the threshold.

    Parameters
    ----------
    data : ndarray (n_TRs x n_voxels x n_subjects)
        fMRI time series data

    tolerate_nans : bool or float (0.0 <= threshold <= 1.0)
        Proportion of subjects with non-NaN values required to keep voxel

    Returns
    -------
    data : ndarray (n_TRs x n_voxels x n_subjects)
        fMRI time series data with adjusted NaNs

    nan_mask : ndarray (n_voxels,)
        Boolean mask array of voxels with too many NaNs based on threshold

    """

    nans = np.all(np.any(np.isnan(data), axis=0), axis=1)

    # Check tolerate_nans input and use either mean/nanmean and exclude voxels
    if tolerate_nans is True:
        logger.info("ISC computation will tolerate all NaNs when averaging")

    elif type(tolerate_nans) is float:
        if not 0.0 <= tolerate_nans <= 1.0:
            raise ValueError("If threshold to tolerate NaNs is a float, "
                             "it must be between 0.0 and 1.0; got {0}".format(
                                tolerate_nans))
        nans += ~(np.sum(~np.any(np.isnan(data), axis=0), axis=1) >=
                  data.shape[-1] * tolerate_nans)
        logger.info("ISC computation will tolerate voxels with at least "
                    "{0} non-NaN values: {1} voxels do not meet "
                    "threshold".format(tolerate_nans,
                                       np.sum(nans)))

    else:
        logger.info("ISC computation will not tolerate NaNs when averaging")

    mask = ~nans
    data = data[:, mask, :]

    return data, mask


def bootstrap_isc(iscs, pairwise=False, summary_statistic='median',
                  n_bootstraps=1000, ci_percentile=95, random_state=None):

    """One-sample group-level bootstrap hypothesis test for ISCs

    For ISCs from one more voxels or ROIs, resample subjects with replacement
    to construct a bootstrap distribution. Input is a list or ndarray of
    ISCs for a single voxel/ROI, or an ISCs-by-voxels ndarray. ISC values
    should be either N ISC values for N subjects in the leave-one-out appraoch
    (pairwise=False), N(N-1)/2 ISC values for N subjects in the pairwise
    approach (pairwise=True). In the pairwise approach, ISC values should
    correspond to the vectorized upper triangle of a square corrlation matrix
    (see scipy.stats.distance.squareform). Shifts bootstrap distribution by
    actual summary statistic (effectively to zero) for two-tailed null
    hypothesis test (Hall & Wilson, 1991). Uses subject-wise (not pair-wise)
    resampling in the pairwise approach. Returns the observed ISC, the
    confidence interval, and a p-value for the bootstrap hypothesis test, as
    well as the bootstrap distribution of summary statistics. According to
    Chen et al., 2016, this is the preferred nonparametric approach for
    controlling false positive rates (FPR) for one-sample tests in the pairwise
    approach.

    The implementation is based on the work in [Chen2016]_ and
    [HallWilson1991]_.

    .. [HallWilson1991] "Two guidelines for bootstrap hypothesis testing.",
       P. Hall, S. R., Wilson, 1991, Biometrics, 757-762.
       https://doi.org/10.2307/2532163

    Parameters
    ----------
    iscs : list or ndarray, ISCs by voxels array
        ISC values for one or more voxels

    pairwise : bool, default: False
        Indicator of pairwise or leave-one-out, should match ISCs structure

    summary_statistic : str, default: 'median'
        Summary statistic, either 'median' (default) or 'mean'

    n_bootstraps : int, default: 1000
        Number of bootstrap samples (subject-level with replacement)

    ci_percentile : int, default: 95
         Percentile for computing confidence intervals

    random_state = int or None, default: None
        Initial random seed

    Returns
    -------
    observed : float, median (or mean) ISC value
        Summary statistic for actual ISCs

    ci : tuple, bootstrap confidence intervals
        Confidence intervals generated from bootstrap distribution

    p : float, p-value
        p-value based on bootstrap hypothesis test

    distribution : ndarray, bootstraps by voxels (optional)
        Bootstrap distribution if return_bootstrap=True

    """

    # Standardize structure of input data
    iscs, n_subjects, n_voxels = _check_isc_input(iscs, pairwise=pairwise)

    # Check for valid summary statistic
    if summary_statistic not in ('mean', 'median'):
        raise ValueError("Summary statistic must be 'mean' or 'median'")

    # Compute summary statistic for observed ISCs
    observed = compute_summary_statistic(iscs,
                                         summary_statistic=summary_statistic,
                                         axis=0)

    # Set up an empty list to build our bootstrap distribution
    distribution = []

    # Loop through n bootstrap iterations and populate distribution
    for i in np.arange(n_bootstraps):

        # Random seed to be deterministically re-randomized at each iteration
        if isinstance(random_state, np.random.RandomState):
            prng = random_state
        else:
            prng = np.random.RandomState(random_state)

        # Randomly sample subject IDs with replacement
        subject_sample = sorted(prng.choice(np.arange(n_subjects),
                                            size=n_subjects))

        # Squareform and shuffle rows/columns of pairwise ISC matrix to
        # to retain correlation structure among ISCs, then get triangle
        if pairwise:

            # Loop through voxels
            isc_sample = []
            for voxel_iscs in iscs.T:

                # Square the triangle and fill diagonal
                voxel_iscs = squareform(voxel_iscs)
                np.fill_diagonal(voxel_iscs, 1)

                # Check that pairwise ISC matrix is square and symmetric
                assert voxel_iscs.shape[0] == voxel_iscs.shape[1]
                assert np.allclose(voxel_iscs, voxel_iscs.T)

                # Shuffle square correlation matrix and get triangle
                voxel_sample = voxel_iscs[subject_sample, :][:, subject_sample]
                voxel_sample = squareform(voxel_sample, checks=False)

                # Censor off-diagonal 1s for same-subject pairs
                voxel_sample[voxel_sample == 1.] = np.NaN

                isc_sample.append(voxel_sample)

            isc_sample = np.column_stack(isc_sample)

        # Get simple bootstrap sample if not pairwise
        elif not pairwise:
            isc_sample = iscs[subject_sample, :]

        # Compute summary statistic for bootstrap ISCs per voxel
        # (alternatively could construct distribution for all voxels
        # then compute statistics, but larger memory footprint)
        distribution.append(compute_summary_statistic(
                                isc_sample,
                                summary_statistic=summary_statistic,
                                axis=0))

        # Update random state for next iteration
        random_state = np.random.RandomState(prng.randint(0, MAX_RANDOM_SEED))

    # Convert distribution to numpy array
    distribution = np.array(distribution)

    # Compute CIs of median from bootstrap distribution (default: 95%)
    ci = (np.percentile(distribution, (100 - ci_percentile)/2, axis=0),
          np.percentile(distribution, ci_percentile + (100 - ci_percentile)/2,
                        axis=0))

    # Shift bootstrap distribution to 0 for hypothesis test
    shifted = distribution - observed

    # Get p-value for actual median from shifted distribution
    p = p_from_null(observed, shifted,
                    side='two-sided', exact=False,
                    axis=0)

    return observed, ci, p, distribution


def _check_group_assignment(group_assignment, n_subjects):
    if type(group_assignment) == list:
        pass
    elif type(group_assignment) == np.ndarray:
        group_assignment = group_assignment.tolist()
    else:
        logger.info("No group assignment provided, "
                    "performing one-sample test.")

    if group_assignment and len(group_assignment) != n_subjects:
        raise ValueError("Group assignments ({0}) "
                         "do not match number of subjects ({1})!".format(
                                len(group_assignment), n_subjects))
    return group_assignment


def _get_group_parameters(group_assignment, n_subjects, pairwise=False):

    # Set up dictionary to contain group info
    group_parameters = {'group_assignment': group_assignment,
                        'n_subjects': n_subjects,
                        'group_labels': None, 'groups': None,
                        'sorter': None, 'unsorter': None,
                        'group_matrix': None, 'group_selector': None}

    # Set up group selectors for two-group scenario
    if group_assignment and len(np.unique(group_assignment)) == 2:
        group_parameters['n_groups'] = 2

        # Get group labels and counts
        group_labels = np.unique(group_assignment)
        groups = {group_labels[0]: group_assignment.count(group_labels[0]),
                  group_labels[1]: group_assignment.count(group_labels[1])}

        # For two-sample pairwise approach set up selector from matrix
        if pairwise:
            # Sort the group_assignment variable if it came in shuffled
            # so it's easier to build group assignment matrix
            sorter = np.array(group_assignment).argsort()
            unsorter = np.array(group_assignment).argsort().argsort()

            # Populate a matrix with group assignments
            upper_left = np.full((groups[group_labels[0]],
                                  groups[group_labels[0]]),
                                 group_labels[0])
            upper_right = np.full((groups[group_labels[0]],
                                   groups[group_labels[1]]),
                                  np.nan)
            lower_left = np.full((groups[group_labels[1]],
                                  groups[group_labels[0]]),
                                 np.nan)
            lower_right = np.full((groups[group_labels[1]],
                                   groups[group_labels[1]]),
                                  group_labels[1])
            group_matrix = np.vstack((np.hstack((upper_left, upper_right)),
                                      np.hstack((lower_left, lower_right))))
            np.fill_diagonal(group_matrix, np.nan)
            group_parameters['group_matrix'] = group_matrix

            # Unsort matrix and squareform to create selector
            group_parameters['group_selector'] = squareform(
                                        group_matrix[unsorter, :][:, unsorter],
                                        checks=False)
            group_parameters['sorter'] = sorter
            group_parameters['unsorter'] = unsorter

        # If leave-one-out approach, just user group assignment as selector
        else:
            group_parameters['group_selector'] = group_assignment

        # Save these parameters for later
        group_parameters['groups'] = groups
        group_parameters['group_labels'] = group_labels

    # Manage one-sample and incorrect group assignments
    elif not group_assignment or len(np.unique(group_assignment)) == 1:
        group_parameters['n_groups'] = 1

        # If pairwise initialize matrix of ones for sign-flipping
        if pairwise:
            group_parameters['group_matrix'] = np.ones((
                                            group_parameters['n_subjects'],
                                            group_parameters['n_subjects']))

    elif len(np.unique(group_assignment)) > 2:
        raise ValueError("This test is not valid for more than "
                         "2 groups! (got {0})".format(
                                len(np.unique(group_assignment))))
    else:
        raise ValueError("Invalid group assignments!")

    return group_parameters


def _permute_one_sample_iscs(iscs, group_parameters, i, pairwise=False,
                             summary_statistic='median', group_matrix=None,
                             exact_permutations=None, prng=None):

    """Applies one-sample permutations to ISC data

    Input ISCs should be n_subjects (leave-one-out approach) or
    n_pairs (pairwise approach) by n_voxels or n_ROIs array.
    This function is only intended to be used internally by the
    permutation_isc function in this module.

    Parameters
    ----------
    iscs : ndarray or list
        ISC values

    group_parameters : dict
        Dictionary of group parameters

    i : int
        Permutation iteration

    pairwise : bool, default: False
        Indicator of pairwise or leave-one-out, should match ISCs variable

    summary_statistic : str, default: 'median'
        Summary statistic, either 'median' (default) or 'mean'

    exact_permutations : list
        List of permutations

    prng = None or np.random.RandomState, default: None
        Initial random seed

    Returns
    -------
    isc_sample : ndarray
        Array of permuted ISC values

    """

    # Randomized sign-flips
    if exact_permutations:
        sign_flipper = np.array(exact_permutations[i])
    else:
        sign_flipper = prng.choice([-1, 1],
                                   size=group_parameters['n_subjects'],
                                   replace=True)

    # If pairwise, apply sign-flips by rows and columns
    if pairwise:
        matrix_flipped = (group_parameters['group_matrix'] * sign_flipper
                                                           * sign_flipper[
                                                                :, np.newaxis])
        sign_flipper = squareform(matrix_flipped, checks=False)

    # Apply flips along ISC axis (same across voxels)
    isc_flipped = iscs * sign_flipper[:, np.newaxis]

    # Get summary statistics on sign-flipped ISCs
    isc_sample = compute_summary_statistic(
                    isc_flipped,
                    summary_statistic=summary_statistic,
                    axis=0)

    return isc_sample


def _permute_two_sample_iscs(iscs, group_parameters, i, pairwise=False,
                             summary_statistic='median',
                             exact_permutations=None, prng=None):

    """Applies two-sample permutations to ISC data

    Input ISCs should be n_subjects (leave-one-out approach) or
    n_pairs (pairwise approach) by n_voxels or n_ROIs array.
    This function is only intended to be used internally by the
    permutation_isc function in this module.

    Parameters
    ----------
    iscs : ndarray or list
        ISC values

    group_parameters : dict
        Dictionary of group parameters

    i : int
        Permutation iteration

    pairwise : bool, default: False
        Indicator of pairwise or leave-one-out, should match ISCs variable

    summary_statistic : str, default: 'median'
        Summary statistic, either 'median' (default) or 'mean'

    exact_permutations : list
        List of permutations

    prng = None or np.random.RandomState, default: None
        Initial random seed
        Indicator of pairwise or leave-one-out, should match ISCs variable

    Returns
    -------
    isc_sample : ndarray
        Array of permuted ISC values

    """

    # Shuffle the group assignments
    if exact_permutations:
        group_shuffler = np.array(exact_permutations[i])
    elif not exact_permutations and pairwise:
        group_shuffler = prng.permutation(np.arange(
            len(np.array(group_parameters['group_assignment'])[
                            group_parameters['sorter']])))
    elif not exact_permutations and not pairwise:
        group_shuffler = prng.permutation(np.arange(
            len(group_parameters['group_assignment'])))

    # If pairwise approach, convert group assignments to matrix
    if pairwise:

        # Apply shuffler to group matrix rows/columns
        group_shuffled = group_parameters['group_matrix'][
                            group_shuffler, :][:, group_shuffler]

        # Unsort shuffled matrix and squareform to create selector
        group_selector = squareform(group_shuffled[
                                    group_parameters['unsorter'], :]
                                    [:, group_parameters['unsorter']],
                                    checks=False)

    # Shuffle group assignments in leave-one-out two sample test
    elif not pairwise:

        # Apply shuffler to group matrix rows/columns
        group_selector = np.array(
                    group_parameters['group_assignment'])[group_shuffler]

    # Get difference of within-group summary statistics
    # with group permutation
    isc_sample = (compute_summary_statistic(
                    iscs[group_selector == group_parameters[
                                            'group_labels'][0], :],
                    summary_statistic=summary_statistic,
                    axis=0) -
                  compute_summary_statistic(
                    iscs[group_selector == group_parameters[
                                            'group_labels'][1], :],
                    summary_statistic=summary_statistic,
                    axis=0))

    return isc_sample


def permutation_isc(iscs, group_assignment=None, pairwise=False,  # noqa: C901
                    summary_statistic='median', n_permutations=1000,
                    random_state=None):

    """Group-level permutation test for ISCs

    For ISCs from one or more voxels or ROIs, permute group assignments to
    construct a permutation distribution. Input is a list or ndarray of
    ISCs  for a single voxel/ROI, or an ISCs-by-voxels ndarray. If two groups,
    ISC values should stacked along first dimension (vertically), and a
    group_assignment list (or 1d array) of same length as the number of
    subjects should be provided to indicate groups. If no group_assignment
    is provided, one-sample test is performed using a sign-flipping procedure.
    Performs exact test if number of possible permutations (2**N for one-sample
    sign-flipping, N! for two-sample shuffling) is less than or equal to number
    of requested permutation; otherwise, performs approximate permutation test
    using Monte Carlo resampling. ISC values should either be N ISC values for
    N subjects in the leave-one-out approach (pairwise=False) or N(N-1)/2 ISC
    values for N subjects in the pairwise approach (pairwise=True). In the
    pairwise approach, ISC values should correspond to the vectorized upper
    triangle of a square corrlation matrix (scipy.stats.distance.squareform).
    Note that in the pairwise approach, group_assignment order should match the
    row/column order of the subject-by-subject square ISC matrix even though
    the input ISCs should be supplied as the vectorized upper triangle of the
    square ISC matrix. Returns the observed ISC and permutation-based p-value
    (two-tailed test), as well as the permutation distribution of summary
    statistic. According to Chen et al., 2016, this is the preferred
    nonparametric approach for controlling false positive rates (FPR) for
    two-sample tests. This approach may yield inflated FPRs for one-sample
    tests.

    The implementation is based on the work in [Chen2016]_.

    Parameters
    ----------
    iscs : list or ndarray, correlation matrix of ISCs
        ISC values for one or more voxels

    group_assignment : list or ndarray, group labels
        Group labels matching order of ISC input

    pairwise : bool, default: False
        Indicator of pairwise or leave-one-out, should match ISCs variable

    summary_statistic : str, default: 'median'
        Summary statistic, either 'median' (default) or 'mean'

    n_permutations : int, default: 1000
        Number of permutation iteration (randomizing group assignment)

    random_state = int, None, or np.random.RandomState, default: None
        Initial random seed

    Returns
    -------
    observed : float, ISC summary statistic or difference
        Actual ISC or group difference (excluding between-group ISCs)

    p : float, p-value
        p-value based on permutation test

    distribution : ndarray, permutations by voxels (optional)
        Permutation distribution if return_bootstrap=True
    """

    # Standardize structure of input data
    iscs, n_subjects, n_voxels = _check_isc_input(iscs, pairwise=pairwise)

    # Check for valid summary statistic
    if summary_statistic not in ('mean', 'median'):
        raise ValueError("Summary statistic must be 'mean' or 'median'")

    # Check match between group labels and ISCs
    group_assignment = _check_group_assignment(group_assignment,
                                               n_subjects)

    # Get group parameters
    group_parameters = _get_group_parameters(group_assignment, n_subjects,
                                             pairwise=pairwise)

    # Set up permutation type (exact or Monte Carlo)
    if group_parameters['n_groups'] == 1:
        if n_permutations < 2**n_subjects:
            logger.info("One-sample approximate permutation test using "
                        "sign-flipping procedure with Monte Carlo resampling.")
            exact_permutations = None
        else:
            logger.info("One-sample exact permutation test using "
                        "sign-flipping procedure with 2**{0} "
                        "({1}) iterations.".format(n_subjects,
                                                   2**n_subjects))
            exact_permutations = list(it.product([-1, 1], repeat=n_subjects))
            n_permutations = 2**n_subjects

    # Check for exact test for two groups
    else:
        if n_permutations < np.math.factorial(n_subjects):
            logger.info("Two-sample approximate permutation test using "
                        "group randomization with Monte Carlo resampling.")
            exact_permutations = None
        else:
            logger.info("Two-sample exact permutation test using group "
                        "randomization with {0}! "
                        "({1}) iterations.".format(
                                n_subjects,
                                np.math.factorial(n_subjects)))
            exact_permutations = list(it.permutations(
                np.arange(len(group_assignment))))
            n_permutations = np.math.factorial(n_subjects)

    # If one group, just get observed summary statistic
    if group_parameters['n_groups'] == 1:
        observed = compute_summary_statistic(
                        iscs,
                        summary_statistic=summary_statistic,
                        axis=0)[np.newaxis, :]

    # If two groups, get the observed difference
    else:
        observed = (compute_summary_statistic(
                        iscs[group_parameters['group_selector'] ==
                             group_parameters['group_labels'][0], :],
                        summary_statistic=summary_statistic,
                        axis=0) -
                    compute_summary_statistic(
                        iscs[group_parameters['group_selector'] ==
                             group_parameters['group_labels'][1], :],
                        summary_statistic=summary_statistic,
                        axis=0))
        observed = np.array(observed)

    # Set up an empty list to build our permutation distribution
    distribution = []

    # Loop through n permutation iterations and populate distribution
    for i in np.arange(n_permutations):

        # Random seed to be deterministically re-randomized at each iteration
        if exact_permutations:
            prng = None
        elif isinstance(random_state, np.random.RandomState):
            prng = random_state
        else:
            prng = np.random.RandomState(random_state)

        # If one group, apply sign-flipping procedure
        if group_parameters['n_groups'] == 1:
            isc_sample = _permute_one_sample_iscs(
                            iscs, group_parameters, i,
                            pairwise=pairwise,
                            summary_statistic=summary_statistic,
                            exact_permutations=exact_permutations,
                            prng=prng)

        # If two groups, set up group matrix get the observed difference
        else:
            isc_sample = _permute_two_sample_iscs(
                            iscs, group_parameters, i,
                            pairwise=pairwise,
                            summary_statistic=summary_statistic,
                            exact_permutations=exact_permutations,
                            prng=prng)

        # Tack our permuted ISCs onto the permutation distribution
        distribution.append(isc_sample)

        # Update random state for next iteration
        if not exact_permutations:
            random_state = np.random.RandomState(prng.randint(
                                                    0, MAX_RANDOM_SEED))

    # Convert distribution to numpy array
    distribution = np.array(distribution)

    # Get p-value for actual median from shifted distribution
    if exact_permutations:
        p = p_from_null(observed, distribution,
                        side='two-sided', exact=True,
                        axis=0)
    else:
        p = p_from_null(observed, distribution,
                        side='two-sided', exact=False,
                        axis=0)

    return observed, p, distribution


def timeshift_isc(data, pairwise=False, summary_statistic='median',
                  n_shifts=1000, tolerate_nans=True, random_state=None):

    """Circular time-shift randomization for one-sample ISC test

    For each voxel or ROI, compute the actual ISC and p-values
    from a null distribution of ISCs where response time series
    are first circularly shifted by random intervals. If pairwise,
    apply time-shift randomization to each subjects and compute pairwise
    ISCs. If leave-one-out approach is used (pairwise=False), apply
    the random time-shift to only the left-out subject in each iteration
    of the leave-one-out procedure. Input data should be a list where
    each item is a time-points by voxels ndarray for a given subject.
    Multiple input ndarrays must be the same shape. If a single ndarray is
    supplied, the last dimension is assumed to correspond to subjects.
    When using leave-one-out approach, NaNs are ignored when computing mean
    time series of N-1 subjects (default: tolerate_nans=True). Alternatively,
    you may supply a float between 0 and 1 indicating a threshold proportion
    of N subjects with non-NaN values required when computing the average time
    series for a given voxel. For example, if tolerate_nans=.8, ISCs will be
    computed for any voxel where >= 80% of subjects have non-NaN values,
    while voxels with < 80% non-NaN values will be assigned NaNs. If set to
    False, NaNs are not tolerated and voxels with one or more NaNs among the
    N-1 subjects will be assigned NaN. Setting tolerate_nans to True or False
    will not affect the pairwise approach; however, if a threshold float is
    provided, voxels that do not reach this threshold will be excluded. Note
    that accommodating NaNs may be notably slower than setting tolerate_nans to
    False. Returns the observed ISC and p-values (two-tailed test), as well as
    the null distribution of ISCs computed on randomly time-shifted data.

    The implementation is based on the work in [Kauppi2010]_ and
    [Kauppi2014]_.

    .. [Kauppi2010] "Inter-subject correlation of brain hemodynamic
       responses during watching a movie: localization in space and
       frequency.", J. P. Kauppi, I. P. Jääskeläinen, M. Sams, J. Tohka,
       2010, Frontiers in Neuroinformatics, 4, 5.
       https://doi.org/10.3389/fninf.2010.00005

    Parameters
    ----------
    data : list or ndarray (n_TRs x n_voxels x n_subjects)
        fMRI data for which to compute ISFC

    pairwise : bool, default: False
        Whether to use pairwise (True) or leave-one-out (False) approach

    summary_statistic : str, default: 'median'
        Summary statistic, either 'median' (default) or 'mean'

    n_shifts : int, default: 1000
        Number of randomly shifted samples

    tolerate_nans : bool or float, default: True
        Accommodate NaNs (when averaging in leave-one-out approach)

    random_state = int, None, or np.random.RandomState, default: None
        Initial random seed

    Returns
    -------
    observed : float, observed ISC (without time-shifting)
        Actual ISCs

    p : float, p-value
        p-value based on time-shifting randomization test

    distribution : ndarray, time-shifts by voxels (optional)
        Time-shifted null distribution if return_bootstrap=True
    """

    # Check response time series input format
    data, n_TRs, n_voxels, n_subjects = _check_timeseries_input(data)

    # Get actual observed ISC
    observed = isc(data, pairwise=pairwise,
                   summary_statistic=summary_statistic,
                   tolerate_nans=tolerate_nans)

    # Roll axis to get subjects in first dimension for loop
    if pairwise:
        data = np.rollaxis(data, 2, 0)

    # Iterate through randomized shifts to create null distribution
    distribution = []
    for i in np.arange(n_shifts):

        # Random seed to be deterministically re-randomized at each iteration
        if isinstance(random_state, np.random.RandomState):
            prng = random_state
        else:
            prng = np.random.RandomState(random_state)

        # Get a random set of shifts based on number of TRs,
        shifts = prng.choice(np.arange(n_TRs), size=n_subjects,
                             replace=True)

        # In pairwise approach, apply all shifts then compute pairwise ISCs
        if pairwise:

            # Apply circular shift to each subject's time series
            shifted_data = []
            for subject, shift in zip(data, shifts):
                shifted_data.append(np.concatenate(
                                        (subject[-shift:, :],
                                         subject[:-shift, :])))
            shifted_data = np.dstack(shifted_data)

            # Compute null ISC on shifted data for pairwise approach
            shifted_isc = isc(shifted_data, pairwise=pairwise,
                              summary_statistic=summary_statistic,
                              tolerate_nans=tolerate_nans)

        # In leave-one-out, apply shift only to each left-out participant
        elif not pairwise:

            shifted_isc = []
            for s, shift in enumerate(shifts):
                shifted_subject = np.concatenate((data[-shift:, :, s],
                                                  data[:-shift, :, s]))
                nonshifted_mean = np.mean(np.delete(data, s, 2), axis=2)
                loo_isc = isc(np.dstack((shifted_subject, nonshifted_mean)),
                              pairwise=False,
                              summary_statistic=None,
                              tolerate_nans=tolerate_nans)
                shifted_isc.append(loo_isc)

            # Get summary statistics across left-out subjects
            shifted_isc = compute_summary_statistic(
                                np.dstack(shifted_isc),
                                summary_statistic=summary_statistic,
                                axis=2)

        distribution.append(shifted_isc)

        # Update random state for next iteration
        random_state = np.random.RandomState(prng.randint(0, MAX_RANDOM_SEED))

    # Convert distribution to numpy array
    distribution = np.vstack(distribution)

    # Get p-value for actual median from shifted distribution
    p = p_from_null(observed, distribution,
                    side='two-sided', exact=False,
                    axis=0)

    return observed, p, distribution


def phaseshift_isc(data, pairwise=False, summary_statistic='median',
                   n_shifts=1000, tolerate_nans=True, random_state=None):

    """Phase randomization for one-sample ISC test

    For each voxel or ROI, compute the actual ISC and p-values
    from a null distribution of ISCs where response time series
    are phase randomized prior to computing ISC. If pairwise,
    apply phase randomization to each subject and compute pairwise
    ISCs. If leave-one-out approach is used (pairwise=False), only
    apply phase randomization to the left-out subject in each iteration
    of the leave-one-out procedure. Input data should be a list where
    each item is a time-points by voxels ndarray for a given subject.
    Multiple input ndarrays must be the same shape. If a single ndarray is
    supplied, the last dimension is assumed to correspond to subjects.
    When using leave-one-out approach, NaNs are ignored when computing mean
    time series of N-1 subjects (default: tolerate_nans=True). Alternatively,
    you may supply a float between 0 and 1 indicating a threshold proportion
    of N subjects with non-NaN values required when computing the average time
    series for a given voxel. For example, if tolerate_nans=.8, ISCs will be
    computed for any voxel where >= 80% of subjects have non-NaN values,
    while voxels with < 80% non-NaN values will be assigned NaNs. If set to
    False, NaNs are not tolerated and voxels with one or more NaNs among the
    N-1 subjects will be assigned NaN. Setting tolerate_nans to True or False
    will not affect the pairwise approach; however, if a threshold float is
    provided, voxels that do not reach this threshold will be excluded. Note
    that accommodating NaNs may be notably slower than setting tolerate_nans to
    False. Returns the observed ISC and p-values (two-tailed test), as well as
    the null distribution of ISCs computed on phase-randomized data.

    The implementation is based on the work in [Lerner2011]_ and
    [Simony2016]_.

    .. [Lerner2011] "Topographic mapping of a hierarchy of temporal
       receptive windows using a narrated story.", Y. Lerner, C. J. Honey,
       L. J. Silbert, U. Hasson, 2011, Journal of Neuroscience, 31, 2906-2915.
       https://doi.org/10.1523/jneurosci.3684-10.2011

    Parameters
    ----------
    data : list or ndarray (n_TRs x n_voxels x n_subjects)
        fMRI data for which to compute ISFC

    pairwise : bool, default: False
        Whether to use pairwise (True) or leave-one-out (False) approach

    summary_statistic : str, default: 'median'
        Summary statistic, either 'median' (default) or 'mean'

    n_shifts : int, default: 1000
        Number of randomly shifted samples

    tolerate_nans : bool or float, default: True
        Accommodate NaNs (when averaging in leave-one-out approach)

    random_state = int, None, or np.random.RandomState, default: None
        Initial random seed

    Returns
    -------
    observed : float, observed ISC (without time-shifting)
        Actual ISCs

    p : float, p-value
        p-value based on time-shifting randomization test

    distribution : ndarray, time-shifts by voxels (optional)
        Time-shifted null distribution if return_bootstrap=True
    """

    # Check response time series input format
    data, n_TRs, n_voxels, n_subjects = _check_timeseries_input(data)

    # Get actual observed ISC
    observed = isc(data, pairwise=pairwise,
                   summary_statistic=summary_statistic,
                   tolerate_nans=tolerate_nans)

    # Iterate through randomized shifts to create null distribution
    distribution = []
    for i in np.arange(n_shifts):

        # Random seed to be deterministically re-randomized at each iteration
        if isinstance(random_state, np.random.RandomState):
            prng = random_state
        else:
            prng = np.random.RandomState(random_state)

        # Get shifted version of data
        shifted_data = phase_randomize(data, random_state=prng)

        # In pairwise approach, apply all shifts then compute pairwise ISCs
        if pairwise:

            # Compute null ISC on shifted data for pairwise approach
            shifted_isc = isc(shifted_data, pairwise=True,
                              summary_statistic=summary_statistic,
                              tolerate_nans=tolerate_nans)

        # In leave-one-out, apply shift only to each left-out participant
        elif not pairwise:

            # Roll subject axis of phase-randomized data
            shifted_data = np.rollaxis(shifted_data, 2, 0)

            shifted_isc = []
            for s, shifted_subject in enumerate(shifted_data):

                # ISC of shifted left-out subject vs mean of N-1 subjects
                nonshifted_mean = np.mean(np.delete(data, s, axis=2),
                                          axis=2)
                loo_isc = isc(np.dstack((shifted_subject, nonshifted_mean)),
                              pairwise=False, summary_statistic=None,
                              tolerate_nans=tolerate_nans)
                shifted_isc.append(loo_isc)

            # Get summary statistics across left-out subjects
            shifted_isc = compute_summary_statistic(
                            np.dstack(shifted_isc),
                            summary_statistic=summary_statistic, axis=2)
        distribution.append(shifted_isc)

        # Update random state for next iteration
        random_state = np.random.RandomState(prng.randint(0, MAX_RANDOM_SEED))

    # Convert distribution to numpy array
    distribution = np.vstack(distribution)

    # Get p-value for actual median from shifted distribution
    p = p_from_null(observed, distribution,
                    side='two-sided', exact=False,
                    axis=0)

    return observed, p, distribution


def phase_randomize(data, voxelwise=False, random_state=None):
    """Randomize phase of time series across subjects

    For each subject, apply Fourier transform to voxel time series
    and then randomly shift the phase of each frequency before inverting
    back into the time domain. This yields time series with the same power
    spectrum (and thus the same autocorrelation) as the original time series
    but will remove any meaningful temporal relationships among time series
    across subjects. By default (voxelwise=False), the same phase shift is
    applied across all voxels; however if voxelwise=True, different random
    phase shifts are applied to each voxel. The typical input is a time by
    voxels by subjects ndarray. The first dimension is assumed to be the
    time dimension and will be phase randomized. If a 2-dimensional ndarray
    is provided, the last dimension is assumed to be subjects, and different
    phase randomizations will be applied to each subject.

    The implementation is based on the work in [Lerner2011]_ and
    [Simony2016]_.

    Parameters
    ----------
    data : ndarray (n_TRs x n_voxels x n_subjects)
        Data to be phase randomized (per subject)

    voxelwise : bool, default: False
        Apply same (False) or different (True) randomizations across voxels

    random_state : RandomState or an int seed (0 by default)
        A random number generator instance to define the state of the
        random permutations generator.

    Returns
    ----------
    shifted_data : ndarray (n_TRs x n_voxels x n_subjects)
        Phase-randomized time series
    """

    # Check if input is 2-dimensional
    data_ndim = data.ndim

    # Get basic shape of data
    data, n_TRs, n_voxels, n_subjects = _check_timeseries_input(data)

    # Random seed to be deterministically re-randomized at each iteration
    if isinstance(random_state, np.random.RandomState):
        prng = random_state
    else:
        prng = np.random.RandomState(random_state)

    # Get randomized phase shifts
    if n_TRs % 2 == 0:
        # Why are we indexing from 1 not zero here? n_TRs / -1 long?
        pos_freq = np.arange(1, data.shape[0] // 2)
        neg_freq = np.arange(data.shape[0] - 1, data.shape[0] // 2, -1)
    else:
        pos_freq = np.arange(1, (data.shape[0] - 1) // 2 + 1)
        neg_freq = np.arange(data.shape[0] - 1,
                             (data.shape[0] - 1) // 2, -1)

    if not voxelwise:
        phase_shifts = (prng.rand(len(pos_freq), 1, n_subjects)
                        * 2 * np.math.pi)
    else:
        phase_shifts = (prng.rand(len(pos_freq), n_voxels, n_subjects)
                        * 2 * np.math.pi)

    # Fast Fourier transform along time dimension of data
    fft_data = fft(data, axis=0)

    # Shift pos and neg frequencies symmetrically, to keep signal real
    fft_data[pos_freq, :, :] *= np.exp(1j * phase_shifts)
    fft_data[neg_freq, :, :] *= np.exp(-1j * phase_shifts)

    # Inverse FFT to put data back in time domain
    shifted_data = np.real(ifft(fft_data, axis=0))

    # Go back to 2-dimensions if input was 2-dimensional
    if data_ndim == 2:
        shifted_data = shifted_data[:, 0, :]

    return shifted_data


def p_from_null(observed, distribution,
                side='two-sided', exact=False,
                axis=None):
    """Compute p-value from null distribution

    Returns the p-value for an observed test statistic given a null
    distribution. Performs either a 'two-sided' (i.e., two-tailed)
    test (default) or a one-sided (i.e., one-tailed) test for either the
    'left' or 'right' side. For an exact test (exact=True), does not adjust
    for the observed test statistic; otherwise, adjusts for observed
    test statistic (prevents p-values of zero). If a multidimensional
    distribution is provided, use axis argument to specify which axis indexes
    resampling iterations.

    The implementation is based on the work in [PhipsonSmyth2010]_.

    .. [PhipsonSmyth2010] "Permutation p-values should never be zero:
       calculating exact p-values when permutations are randomly drawn.",
       B. Phipson, G. K., Smyth, 2010, Statistical Applications in Genetics
       and Molecular Biology, 9, 1544-6115.
       https://doi.org/10.2202/1544-6115.1585

    Parameters
    ----------
    observed : float
        Observed test statistic

    distribution : ndarray
        Null distribution of test statistic

    side : str, default:'two-sided'
        Perform one-sided ('left' or 'right') or 'two-sided' test

    axis: None or int, default:None
        Axis indicating resampling iterations in input distribution

    Returns
    -------
    p : float
        p-value for observed test statistic based on null distribution
    """

    if side not in ('two-sided', 'left', 'right'):
        raise ValueError("The value for 'side' must be either "
                         "'two-sided', 'left', or 'right', got {0}".
                         format(side))

    n_samples = len(distribution)
    logger.info("Assuming {0} resampling iterations".format(n_samples))

    if side == 'two-sided':
        # Numerator for two-sided test
        numerator = np.sum(np.abs(distribution) >= np.abs(observed), axis=axis)
    elif side == 'left':
        # Numerator for one-sided test in left tail
        numerator = np.sum(distribution <= observed, axis=axis)
    elif side == 'right':
        # Numerator for one-sided test in right tail
        numerator = np.sum(distribution >= observed, axis=axis)

    # If exact test all possible permutations and do not adjust
    if exact:
        p = numerator / n_samples

    # If not exact test, adjust number of samples to account for
    # observed statistic; prevents p-value from being zero
    else:
        p = (numerator + 1) / (n_samples + 1)

    return p


def _check_timeseries_input(data):

    """Checks response time series input data (e.g., for ISC analysis)

    Input data should be a n_TRs by n_voxels by n_subjects ndarray
    (e.g., brainiak.image.MaskedMultiSubjectData) or a list where each
    item is a n_TRs by n_voxels ndarray for a given subject. Multiple
    input ndarrays must be the same shape. If a 2D array is supplied,
    the last dimension is assumed to correspond to subjects. This
    function is generally intended to be used internally by other
    functions module (e.g., isc, isfc in brainiak.isc).

    Parameters
    ----------
    data : ndarray or list
        Time series data

    Returns
    -------
    data : ndarray
        Input time series data with standardized structure

    n_TRs : int
        Number of time points (TRs)

    n_voxels : int
        Number of voxels (or ROIs)

    n_subjects : int
        Number of subjects

    """

    # Convert list input to 3d and check shapes
    if type(data) == list:
        data_shape = data[0].shape
        for i, d in enumerate(data):
            if d.shape != data_shape:
                raise ValueError("All ndarrays in input list "
                                 "must be the same shape!")
            if d.ndim == 1:
                data[i] = d[:, np.newaxis]
        data = np.dstack(data)

    # Convert input ndarray to 3d and check shape
    elif isinstance(data, np.ndarray):
        if data.ndim == 2:
            data = data[:, np.newaxis, :]
        elif data.ndim == 3:
            pass
        else:
            raise ValueError("Input ndarray should have 2 "
                             "or 3 dimensions (got {0})!".format(data.ndim))

    # Infer subjects, TRs, voxels and log for user to check
    n_TRs, n_voxels, n_subjects = data.shape
    logger.info("Assuming {0} subjects with {1} time points "
                "and {2} voxel(s) or ROI(s) for ISC analysis.".format(
                    n_subjects, n_TRs, n_voxels))

    return data, n_TRs, n_voxels, n_subjects


def array_correlation(x, y, axis=0):

    """Column- or row-wise Pearson correlation between two arrays
    Computes sample Pearson correlation between two 1D or 2D arrays (e.g.,
    two n_TRs by n_voxels arrays). For 2D arrays, computes correlation
    between each corresponding column (axis=0) or row (axis=1) where axis
    indexes observations. If axis=0 (default), each column is considered to
    be a variable and each row is an observation; if axis=1, each row is a
    variable and each column is an observation (equivalent to transposing
    the input arrays). Input arrays must be the same shape with corresponding
    variables and observations. This is intended to be an efficient method
    for computing correlations between two corresponding arrays with many
    variables (e.g., many voxels).

    Parameters
    ----------
    x : 1D or 2D ndarray
        Array of observations for one or more variables

    y : 1D or 2D ndarray
        Array of observations for one or more variables (same shape as x)

    axis : int (0 or 1), default: 0
        Correlation between columns (axis=0) or rows (axis=1)

    Returns
    -------
    r : float or 1D ndarray
        Pearson correlation values for input variables

    """

    # Accommodate array-like inputs
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)

    # Check that inputs are same shape
    if x.shape != y.shape:
        raise ValueError("Input arrays must be the same shape")

    # Transpose if axis=1 requested (to avoid broadcasting
    # issues introduced by switching axis in mean and sum)
    if axis == 1:
        x, y = x.T, y.T

    # Center (de-mean) input variables
    x_demean = x - np.mean(x, axis=0)
    y_demean = y - np.mean(y, axis=0)

    # Compute summed product of centered variables
    numerator = np.sum(x_demean * y_demean, axis=0)

    # Compute sum squared error
    denominator = np.sqrt(np.sum(x_demean ** 2, axis=0) *
                          np.sum(y_demean ** 2, axis=0))

    return numerator / denominator


def pairwise_edge_isc(x, axis=0):

    """Column- or row-wise edge time series (ETS) between all posible pairs of subjects.
    Computes ETS between all pairs of subjects in a 2D or 3D arrays (e.g.,
    n_TRs by n_voxels by subjects array).

    Parameters
    ----------
    x : 2D or 2D ndarray
        [n_TRs by n_voxels by subjects] or [n_TRs by subjects]

    Returns
    -------
    r : 4D ndarray
        edge time series values for input variables [n_TRs by subjects by subjects by n_voxels]

    """

    # Accommodate array-like inputs
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)

    if len(x.shape) == 2: # add voxel dimension if missing
        x = x[:,np.newaxis,:]

    x = np.transpose(x,axes=(2,0,1))

    # z scored input variables

    x_zscored = x - np.mean(x, axis=1)[:,np.newaxis,:]
    x_zscored = x_zscored / np.std(x, axis=1)[:,np.newaxis,:]

    # Compute summed product of centered variables
    return x_zscored[np.newaxis,:,:,:] * x_zscored[:,np.newaxis,:,:]

def array_edge_isc(x, y, axis=0):

    """Column- or row-wise edge time series (ETS) between two arrays.
    Computes ETS between two 1D or 2D arrays (e.g.,
    two n_TRs by n_voxels arrays). For 2D arrays, computes correlation
    between each corresponding column (axis=0) or row (axis=1) where axis
    indexes observations. If axis=0 (default), each column is considered to
    be a variable and each row is an observation; if axis=1, each row is a
    variable and each column is an observation (equivalent to transposing
    the input arrays). Input arrays must be the same shape with corresponding
    variables and observations. This is intended to be an efficient method
    for computing ETS between two corresponding arrays with many
    variables (e.g., many voxels).

    Parameters
    ----------
    x : 1D or 2D ndarray
        Array of observations for one or more variables

    y : 1D or 2D ndarray
        Array of observations for one or more variables (same shape as x)

    axis : int (0 or 1), default: 0
        Correlation between columns (axis=0) or rows (axis=1)

    Returns
    -------
    r : float or 1D ndarray
        Pearson correlation values for input variables

    """

    # Accommodate array-like inputs
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)

    # Check that inputs are same shape
    if x.shape != y.shape:
        raise ValueError("Input arrays must be the same shape")

    # Transpose if axis=1 requested (to avoid broadcasting
    # issues introduced by switching axis in mean and sum)
    if axis == 1:
        x, y = x.T, y.T

    if len(x.shape) == 1:
        x = x[:,np.newaxis]
        y = y[:,np.newaxis]

    # z scored input variables
    x_zscored = x - np.mean(x, axis=0)[np.newaxis,:]
    x_zscored = x_zscored / np.std(x, axis=0)[np.newaxis,:]
    y_zscored = y - np.mean(y, axis=0)[np.newaxis,:]
    y_zscored = y_zscored / np.std(y, axis=0)[np.newaxis,:]

    # Compute summed product of centered variables
    return x_zscored * y_zscored

def array_edge_time_series(x, y, axis=0):

    """Column- or row-wise edge time series (ETS) between two arrays.
    Computes ETS between two 1D or 2D arrays (e.g.,
    two n_TRs by n_voxels arrays). For 2D arrays, computes correlation
    between each corresponding column (axis=0) or row (axis=1) where axis
    indexes observations. If axis=0 (default), each column is considered to
    be a variable and each row is an observation; if axis=1, each row is a
    variable and each column is an observation (equivalent to transposing
    the input arrays). Input arrays must be the same shape with corresponding
    variables and observations. This is intended to be an efficient method
    for computing ETS between two corresponding arrays with many
    variables (e.g., many voxels).

    Parameters
    ----------
    x : 1D or 2D ndarray
        Array of observations for one or more variables

    y : 1D or 2D ndarray
        Array of observations for one or more variables (same shape as x)

    axis : int (0 or 1), default: 0
        Correlation between columns (axis=0) or rows (axis=1)

    Returns
    -------
    r : float or 1D ndarray
        Pearson correlation values for input variables

    """

    # Accommodate array-like inputs
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)

    # Check that inputs are same shape
    if x.shape != y.shape:
        raise ValueError("Input arrays must be the same shape")
    x, y = x.astype(np.float32), y.astype(np.float32)
    # Transpose if axis=1 requested (to avoid broadcasting
    # issues introduced by switching axis in mean and sum)
    if axis == 1:
        x, y = x.T, y.T

    if len(x.shape) == 1:
        x = x[:,np.newaxis]
        y = y[:,np.newaxis]

    # z scored input variables
    x_zscored = x - np.mean(x, axis=0)[np.newaxis,:]
    x_zscored = x_zscored / np.std(x, axis=0)[np.newaxis,:]
    y_zscored = y - np.mean(y, axis=0)[np.newaxis,:]
    y_zscored = y_zscored / np.std(y, axis=0)[np.newaxis,:]

    return x_zscored[:,:,np.newaxis] * y_zscored[:,np.newaxis,:]


def array_isfc_weighted(x, y, weights, axis=0):

    """Column- or row-wise edge time series (ETS) between two arrays.
    Computes ETS between two 1D or 2D arrays (e.g.,
    two n_TRs by n_voxels arrays). For 2D arrays, computes correlation
    between each corresponding column (axis=0) or row (axis=1) where axis
    indexes observations. If axis=0 (default), each column is considered to
    be a variable and each row is an observation; if axis=1, each row is a
    variable and each column is an observation (equivalent to transposing
    the input arrays). Input arrays must be the same shape with corresponding
    variables and observations. This is intended to be an efficient method
    for computing ETS between two corresponding arrays with many
    variables (e.g., many voxels).

    Parameters
    ----------
    x : 1D or 2D ndarray
        Array of observations for one or more variables

    y : 1D or 2D ndarray
        Array of observations for one or more variables (same shape as x)

    axis : int (0 or 1), default: 0
        Correlation between columns (axis=0) or rows (axis=1)

    Returns
    -------
    r : float or 1D ndarray
        Pearson correlation values for input variables

    """

    # Accommodate array-like inputs
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)

    # Check that inputs are same shape
    if x.shape != y.shape:
        raise ValueError("Input arrays must be the same shape")

    # Transpose if axis=1 requested (to avoid broadcasting
    # issues introduced by switching axis in mean and sum)
    if axis == 1:
        x, y = x.T, y.T

    if len(x.shape) == 1:
        x = x[:,np.newaxis]
        y = y[:,np.newaxis]

    # https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Weighted_correlation_coefficient
    sum_w = weights.sum()
    x_w_mean = (weights[np.newaxis,:]* x).sum(axis=1) / sum_w
    y_w_mean = (weights[np.newaxis,:]* y).sum(axis=1) / sum_w
    x_demeaned = x - x_w_mean[:,np.newaxis]
    y_demeaned = y - y_w_mean[:,np.newaxis]
    # numerator = (x_demeaned * y_demeaned * weights[np.newaxis,:]).sum(axis=1) / sum_w # for ISC
    numerator = (x_demeaned[np.newaxis, :, :] * y_demeaned[:,np.newaxis,:] *
                 weights[np.newaxis,np.newaxis,:]).sum(axis=2) / sum_w
    x_w_cov = (x_demeaned * x_demeaned *
               weights[np.newaxis,:]).sum(axis=1) / sum_w
    y_w_cov = (y_demeaned * y_demeaned *
               weights[np.newaxis,:]).sum(axis=1) / sum_w
    denominator = np.sqrt(x_w_cov[np.newaxis,:] * y_w_cov[:,np.newaxis])

    # x_w_cov = (x_demeaned[np.newaxis, :, :] * x_demeaned[:,np.newaxis,:] *
    #            weights[np.newaxis,np.newaxis,:]).sum(axis=2) / sum_w
    # y_w_cov = (y_demeaned[np.newaxis, :, :] * y_demeaned[:,np.newaxis,:] *
    #            weights[np.newaxis,np.newaxis,:]).sum(axis=2) / sum_w
    # denominator = np.sqrt(x_w_cov * y_w_cov)

    return numerator / denominator

def load_images(image_paths: Iterable[Union[str, Path]]
                ) -> Iterable[SpatialImage]:
    """Load images from paths.

    For efficiency, returns an iterator, not a sequence, so the results cannot
    be accessed by indexing.

    For every new iteration through the images, load_images must be called
    again.

    Parameters
    ----------
    image_paths:
        Paths to images.

    Yields
    ------
    SpatialImage
        Image.
    """
    for image_path in image_paths:
        if isinstance(image_path, Path):
            string_path = str(image_path)
        else:
            string_path = image_path
        logger.debug(
            'Starting to read file %s', string_path
        )
        yield nib.load(string_path)


def load_boolean_mask(path: Union[str, Path],
                      predicate: Callable[[np.ndarray], np.ndarray] = None
                      ) -> np.ndarray:
    """Load boolean nibabel.SpatialImage mask.

    Parameters
    ----------
    path
        Mask path.
    predicate
        Callable used to create boolean values, e.g. a threshold function
        ``lambda x: x > 50``.

    Returns
    -------
    np.ndarray
        Boolean array corresponding to mask.
    """
    if not isinstance(path, str):
        path = str(path)
    data = nib.load(path).get_data()
    if predicate is not None:
        mask = predicate(data)
    else:
        mask = data.astype(np.bool)
    return mask


T = TypeVar("T", bound="MaskedMultiSubjectData")


class MaskedMultiSubjectData(np.ndarray):
    """Array with shape n_TRs, n_voxels, n_subjects."""
    @classmethod
    def from_masked_images(cls: Type[T], masked_images: Iterable[np.ndarray],
                           n_subjects: int) -> T:
        """Create a new instance of MaskedMultiSubjecData from masked images.

        Parameters
        ----------
        masked_images : iterator
            Images from multiple subjects to stack along 3rd dimension
        n_subjects : int
            Number of subjects; must match the number of images

        Returns
        -------
        T
            A new instance of MaskedMultiSubjectData

        Raises
        ------
        ValueError
            Images have different shapes.

            The number of images differs from n_subjects.
        """
        images_iterator = iter(masked_images)
        first_image = next(images_iterator)
        first_image_shape = first_image.T.shape
        result = np.empty((first_image_shape[0], first_image_shape[1],
                           n_subjects))
        for n_images, image in enumerate(itertools.chain([first_image],
                                                         images_iterator)):
            image = image.T
            if image.shape != first_image_shape:
                raise ValueError("Image {} has different shape from first "
                                 "image: {} != {}".format(n_images,
                                                          image.shape,
                                                          first_image_shape))
            result[:, :, n_images] = image
        n_images += 1
        if n_images != n_subjects:
            raise ValueError("n_subjects != number of images: {} != {}"
                             .format(n_subjects, n_images))
        return result.view(cls)


def mask_image(image: SpatialImage, mask: np.ndarray, data_type: type = None
               ) -> np.ndarray:
    """Mask image after optionally casting its type.

    Parameters
    ----------
    image
        Image to mask. Can include time as the last dimension.
    mask
        Mask to apply. Must have the same shape as the image data.
    data_type
        Type to cast image to.

    Returns
    -------
    np.ndarray
        Masked image.

    Raises
    ------
    ValueError
        Image data and masks have different shapes.
    """
    image_data = image.get_data()
    if image_data.shape[:3] != mask.shape:
        raise ValueError("Image data and mask have different shapes.")
    if data_type is not None:
        cast_data = image_data.astype(data_type)
    else:
        cast_data = image_data
    return cast_data[mask]


def multimask_images(images: Iterable[SpatialImage],
                     masks: Sequence[np.ndarray], image_type: type = None
                     ) -> Iterable[Sequence[np.ndarray]]:
    """Mask images with multiple masks.

    Parameters
    ----------
    images:
        Images to mask.
    masks:
        Masks to apply.
    image_type:
        Type to cast images to.

    Yields
    ------
    Sequence[np.ndarray]
        For each mask, a masked image.
    """
    for image in images:
        yield [mask_image(image, mask, image_type) for mask in masks]


def mask_images(images: Iterable[SpatialImage], mask: np.ndarray,
                image_type: type = None) -> Iterable[np.ndarray]:
    """Mask images.

    Parameters
    ----------
    images:
        Images to mask.
    mask:
        Mask to apply.
    image_type:
        Type to cast images to.

    Yields
    ------
    np.ndarray
        Masked image.
    """
    for images in multimask_images(images, (mask,), image_type):
        yield images[0]

def squareform_isfc(isfcs, iscs=None):

    """Converts square ISFCs to condensed ISFCs (and ISCs), and vice-versa
    If input is a 2- or 3-dimensional array of square ISFC matrices, converts
    this to the condensed off-diagonal ISFC values (i.e., the vectorized
    triangle) and the diagonal ISC values. In this case, input must be a
    single array of shape either n_voxels x n_voxels or n_subjects (or
    n_pairs) x n_voxels x n_voxels. The condensed ISFC values are vectorized
    according to scipy.spatial.distance.squareform, yielding n_voxels *
    (n_voxels - 1) / 2 values comprising every voxel pair. Alternatively, if
    input is an array of condensed off-diagonal ISFC values and an array of
    diagonal ISC values, the square (redundant) ISFC values are returned.
    This function mimics scipy.spatial.distance.squareform, but is intended
    to retain the diagonal ISC values.
    Parameters
    ----------
    isfcs : ndarray
        Either condensed or redundant ISFC values
    iscs: ndarray, optional
        Diagonal ISC values, required when input is condensed
    Returns
    -------
    isfcs : ndarray or tuple of ndarrays
        If condensed ISFCs are passed, a single redundant ISFC array is
        returned; if redundant ISFCs are passed, both a condensed off-
        diagonal ISFC array and the diagonal ISC values are returned
    """

    # Check if incoming ISFCs are square (redundant)
    if not type(iscs) == np.ndarray and isfcs.shape[-2] == isfcs.shape[-1]:
        if isfcs.ndim == 2:
            isfcs = isfcs[np.newaxis, ...]
        if isfcs.ndim == 3:
            iscs = np.diagonal(isfcs, axis1=1, axis2=2)
            isfcs = np.vstack([squareform(isfc, checks=False)[np.newaxis, :]
                               for isfc in isfcs])
        else:
            raise ValueError("Square (redundant) ISFCs must be square "
                             "with multiple subjects or pairs of subjects "
                             "indexed by the first dimension")
        if isfcs.shape[0] == iscs.shape[0] == 1:
            isfcs, iscs = isfcs[0], iscs[0]
        return isfcs, iscs

    # Otherwise, convert from condensed to redundant
    else:
        if isfcs.ndim == iscs.ndim == 1:
            isfcs, iscs = isfcs[np.newaxis, :], iscs[np.newaxis, :]
        isfcs_stack = []
        for isfc, isc in zip(isfcs, iscs):
            isfc_sq = squareform(isfc, checks=False)
            np.fill_diagonal(isfc_sq, isc)
            isfcs_stack.append(isfc_sq[np.newaxis, ...])
        isfcs = np.vstack(isfcs_stack)
        if isfcs.shape[0] == 1:
            isfcs = isfcs[0]
        return isfcs