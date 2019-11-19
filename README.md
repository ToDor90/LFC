# LFC
Full machine-learning representation and raw data for the static local field correction (LFC) of the warm dense electron gas.

Cite as:
T. Dornheim et al, J. Chem. Phys. 151, 194104 (2019)
https://doi.org/10.1063/1.5123013

The file GetLFC.py contains a defintion of the machine-learning representation of G(x,rs,theta) (with x=q/q_F, q_F being the Fermi wave number) using Keras (https://keras.io) and TensorFlow (https://tensorflow.org).
The function G(x,rs,theta) shows how to use the neural net to obtain the static LFC at some specific parameters.

The finite-size corrected PIMC raw data used to train the neural net can be found in the folder "PIMC_data".
