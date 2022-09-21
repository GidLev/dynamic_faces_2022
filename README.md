# Fine-scale dynamics of functional connectivity in the face processing network during movie watching



Python implementation of the analysis from Levakov et al. 2022 paper, mainly the following:

* face_detection_cnn.py - Code for extracting face features from the movie frames
* frames_face_detected.mp4 - A movie depicting the face annotation
* faces_area.npy, n_faces.npy - Extracted face measures
* is_nts_ets_simulation.ipynb - Interactive notebook demonstrating the IS-N/ETS derivation and the IS edge seed correlation method
* is_edge_seed_corr.py - In preparation
* plot_utils.py - Functions used in is_nts_ets_simulation.ipynb
* isc_standalone.py - Inter-subject correlation standalone version with the IS-N/ETS implementations

## isc_standalone.py
A modified version from: https://github.com/snastase/isc-tutorial/blob/master/isc_tutorial/isc_standalone.py

The major change is the addition of functions for calculating inter-subject


The following main functions were added:

* isc_ets - Intersubject node time-series (IS-NTS)
* isfc_ets - Intersubject edge time-series (IS-ETS)

## Citing

If you use this code, please cite:

    Levakov, G., Sporns, O., & Avidan, G. (2022). Fine-scale dynamics of functional connectivity in the face processing network during movie watching. bioRxiv.
