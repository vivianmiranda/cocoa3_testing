CAMB-MultiFluidDE
===================
Authors: João Victor Silva Rebouças, Vivian Miranda

Description and installation
=============================

CAMB is a cosmology code for calculating cosmological observables, including
CMB, lensing, source count and 21cm angular power spectra, matter power spectra, transfer functions
and background evolution. The goal of this modification is to provide CAMB a new dark energy interface that facilitates the implementation of multiple components in the dark sector.

So far, we have implemented two fluids: one late DE fluid modelled by a constant w, and one EDE fluid, using the Axion Effective Fluid model. In the future, more models will be implemented.

See `docs/Multifluid_test.ipynb` for a quick guide on how to use the multifluid model.

    pip install camb [--user]

If you want to work on the code, you can also just install in place without copying anything using::

    pip install -e . [--user]

You will need gfortran 6 or higher installed to compile. If you have gfortran installed, "python setup.py make"
(and other standard setup commands) will build the Fortran library on all systems (including Windows without directly using a Makefile).

Usage
=============================
In the `docs` folder you will find a Jupyter Notebook `multifluid_test.ipynb`. There, you'll see how to use this modification to study dark energy models composed by many fluids.

In Python, use the `MultifluidDE`

