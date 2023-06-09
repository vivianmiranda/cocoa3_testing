# Table of contents
1. [Overview of the Cobaya-CosmoLike Joint Architecture (Cocoa)](#overview)
3. [Installation of Cocoa's required packages](#required_packages)
    1. [Via Conda](#required_packages_conda)
    3. (expert) [Via Cocoa's internal cache](#required_packages_cache)
4. [Installation of Cobaya base code](#cobaya_base_code)
5. [Running Cobaya Examples](#cobaya_base_code_examples)
6. [Running Cosmolike projects](#running_cosmolike_projects)
7. [Creating Cosmolike projects (external readme)](Cocoa/projects/)
8. [Appendix](#appendix)
    1. [Proper Credits](#appendix_proper_credits)
    2. [The whovian-cocoa docker container](#appendix_jupyter_whovian)
    3. [Miniconda Installation](#overview_miniconda)
    4. [Compiling Boltzmann, CosmoLike and Likelihood codes separatelly](#appendix_compile_separatelly)
    5. [Warning about Weak Lensing YAML files in Cobaya](#appendix_example_runs)
    6. [Manual Blocking of Cosmolike Parameters](#manual_blocking_cosmolike)
    7. [Setting-up conda environment for Machine Learning emulators](#ml_emulators)
    8. [Adapting new modified CAMB/CLASS (external readme)](Cocoa/external_modules/code)
 
## Overview of the [Cobaya](https://github.com/CobayaSampler)-[CosmoLike](https://github.com/CosmoLike) Joint Architecture (Cocoa) <a name="overview"></a>

Cocoa allows users to run [CosmoLike](https://github.com/CosmoLike) routines inside the [Cobaya](https://github.com/CobayaSampler) framework. [CosmoLike](https://github.com/CosmoLike) can analyze data primarily from the [Dark Energy Survey](https://www.darkenergysurvey.org) and simulate future multi-probe analyses for LSST and Roman Space Telescope. Besides integrating [Cobaya](https://github.com/CobayaSampler) and [CosmoLike](https://github.com/CosmoLike), Cocoa introduces shell scripts and readme instructions that allow users to containerize [Cobaya](https://github.com/CobayaSampler). The container structure ensures that users will adopt the same compiler and libraries (including their versions), and that they will be able to use multiple [Cobaya](https://github.com/CobayaSampler) instances consistently. This readme file presents basic and advanced instructions for installing all [Cobaya](https://github.com/CobayaSampler) and [CosmoLike](https://github.com/CosmoLike) components.

## Installation of Cocoa's required packages <a name="required_packages"></a>

There are two installation methods. Users must choose one of them:

1. [Via Conda](#required_packages_conda) (easier, best)
2. [Via Cocoa's internal cache](#required_packages_cache) (slow, not advisable) 

We also provide the docker image whovian-cocoa to facilitate the installation of Cocoa on Windows and MacOS. For further instructions, refer to the Appendix [whovian-cocoa docker container](#appendix_jupyter_whovian).

### Via Conda <a name="required_packages_conda"></a>

We assume here the user has previously installed either [Minicoda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual). If this is not the case, then refer to the Appendix [Miniconda Installation](#overview_miniconda) for further instructions.

Type the following commands to create the cocoa Conda environment.

    $ conda create --name cocoa python=3.7 --quiet --yes \
       && conda install -n cocoa --quiet --yes  \
       'conda-forge::libgcc-ng=10.3.0' \
       'conda-forge::libstdcxx-ng=10.3.0' \
       'conda-forge::libgfortran-ng=10.3.0' \
       'conda-forge::gxx_linux-64=10.3.0' \
       'conda-forge::gcc_linux-64=10.3.0' \
       'conda-forge::gfortran_linux-64=10.3.0' \
       'conda-forge::openmpi=4.1.1' \
       'conda-forge::sysroot_linux-64=2.17' \
       'conda-forge::git=2.33.1' \
       'conda-forge::git-lfs=3.0.2' \
       'conda-forge::hdf5=1.10.6' \
       'conda-forge::git-lfs=3.0.2' \
       'conda-forge::cmake=3.21.3' \
       'conda-forge::boost=1.76.0' \
       'conda-forge::gsl=2.7' \
       'conda-forge::fftw=3.3.10' \
       'conda-forge::cfitsio=4.0.0' \
       'conda-forge::openblas=0.3.18' \
       'conda-forge::lapack=3.9.0' \
       'conda-forge::armadillo=10.7.3'\
       'conda-forge::expat=2.4.1' \
       'conda-forge::cython=0.29.24' \
       'conda-forge::numpy=1.21.4' \
       'conda-forge::scipy=1.7.2' \
       'conda-forge::pandas=1.3.4' \
       'conda-forge::mpi4py=3.1.2' \
       'conda-forge::matplotlib=3.5.1' \
       'conda-forge::astropy=4.3.1'
      
For those working on projects that utilize machine-learning-based emulators, the Appendix [Setting-up conda environment for Machine Learning emulators](#ml_emulators) provides additional commands for installing the necessary packages.

When adopting this installation method, users must activate the Conda environment whenever working with Cocoa, as shown below.

    $ conda activate cocoa
    
Furthermore, users must install GIT-LFS on the first loading of the Conda cocoa environment.

    $(cocoa) $CONDA_PREFIX/bin/git-lfs install

Users can now proceed to the section [Installation of Cobaya base code](#cobaya_base_code).

### (expert) Via Cocoa's internal cache <a name="required_packages_cache"></a>

This method is slow and not advisable. When Conda is unavailable, the user can still perform a local semi-autonomous installation on Linux based on a few scripts we implemented. We provide a local copy of almost all required packages on Cocoa's cache folder named *cocoa_installation_libraries*. We assume the pre-installation of the following packages:

   - [Bash](https://www.amazon.com/dp/B0043GXMSY/ref=cm_sw_em_r_mt_dp_x3UoFbDXSXRBT);
   - [Git](https://git-scm.com) v1.8+;
   - [Git LFS](https://git-lfs.github.com);
   - [gcc](https://gcc.gnu.org) v10.*;
   - [gfortran](https://gcc.gnu.org) v10.*;
   - [g++](https://gcc.gnu.org) v10.*;
   - [Python](https://www.python.org) v3.7.*;
   - [PIP package manager](https://pip.pypa.io/en/stable/installing/)
   - [Python Virtual Environment](https://www.geeksforgeeks.org/python-virtual-environment/)

To perform the local semi-autonomous installation, users must modify flags written on the file *set_installation_options* because the default behavior corresponds to an installation via Conda. First, select the environmental key `MANUAL_INSTALLATION` as shown below:

    [Extracted from set_installation_options script] 
    
    # --------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------
    # ----------------------- HOW COCOA SHOULD BE INSTALLED? -------------------------------
    # --------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------
    #export MINICONDA_INSTALLATION=1
    export MANUAL_INSTALLATION=1
    
Finally, set the following environmental keys
 
    [Extracted from set_installation_options script]
  
    elif [ -n "${MANUAL_INSTALLATION}" ]; then

        export GLOBAL_PACKAGES_LOCATION=/usr/local
        export PYTHON_VERSION=3
        export FORTRAN_COMPILER=gfortran

        export C_COMPILER=gcc
        export CXX_COMPILER=g++
        export GLOBALPYTHON3=python3
        export MPI_FORTRAN_COMPILER=mpif90
        export MPI_CXX_COMPILER=mpicc
        export MPI_CC_COMPILER=mpicxx

        # IF TRUE, THEN COCOA ADOPTS FFTW10. OTHERWISE, COCOA ADOPTS FFTW8
        #export FFTW_NEW_VERSION=1
        #export DONT_USE_SYSTEM_PIP_PACKAGES=1

        # IF TRUE, THEN COCOA WON'T INSTALL TENSORFLOW, KERAS and PYTORCH
        #export IGNORE_EMULATOR_PIP_PACKAGES=1

        #export IGNORE_DISTUTILS_INSTALLATION=1
        #export IGNORE_OPENBLAS_INSTALLATION=1
        #export IGNORE_XZ_INSTALLATION=1
        #export IGNORE_ALL_PIP_INSTALLATION=1
        #export IGNORE_CMAKE_INSTALLATION=1
        #export IGNORE_CPP_BOOST_INSTALLATION=1
        #export IGNORE_CPP_ARMA_INSTALLATION=1
        #export IGNORE_CPP_SPDLOG_INSTALLATION=1
        #export IGNORE_C_GSL_INSTALLATION=1
        #export IGNORE_C_CFITSIO_INSTALLATION=1
        #export IGNORE_C_FFTW_INSTALLATION=1
        #export IGNORE_OPENBLAS_INSTALLATION=1
        #export IGNORE_FORTRAN_LAPACK_INSTALLATION=1
 
Users can now proceed to the section [Installation of Cobaya base code](#cobaya_base_code)

## Installation of Cobaya base code <a name="cobaya_base_code"></a>

Assuming the user opted for the easier *Conda installation*, type:

    $ conda activate cocoa
    
    $(cocoa) $CONDA_PREFIX/bin/git-lfs clone https://github.com/CosmoLike/cocoa.git

to clone the repository. Cocoa developers with set ssh keys in GitHub should instead use the command

    $(cocoa) $CONDA_PREFIX/bin/git-lfs clone git@github.com:CosmoLike/cocoa.git

(**Warning**) We have a limited monthly quota in bandwidth for Git LFS files, and therefore we ask users to use good judgment in the number of times they clone files from Cocoa's main repository.
 
Cocoa is made aware of the chosen installation method of required packages via special environment keys located on the *Cocoa/set_installation_options* script, as shown below:

    [Extracted from set_installation_options script]
    # --------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------
    # ----------------------- HOW COCOA SHOULD BE INSTALLED? -------------------------------
    # --------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------
    export MINICONDA_INSTALLATION=1
    #export MANUAL_INSTALLATION=1
    
The user must uncomment the appropriate key (here, we assume `MINICONDA_INSTALLATION`), and then type the following command

    $(cocoa) cd ./Cocoa/
    $(cocoa) source setup_cocoa_installation_packages

The script `setup_cocoa_installation_packages` decompresses the data files, which only takes a few minutes, and installs any remaining necessary packages. Typical package installation time ranges, depending on the installation method, from a few minutes (installation via Conda) to more than one hour (installation via Cocoa's internal cache). It is important to note that our scripts never install packages on `$HOME/.local`. All requirements for Cocoa are installed at

    Cocoa/.local/bin
    Cocoa/.local/include
    Cocoa/.local/lib
    Cocoa/.local/share

This behavior is critical to enable users to work on multiple instances of Cocoa simultaneously.

Finally, type

    $(cocoa) source compile_external_modules
    
to compile CAMB, CLASS, Planck and Polychord. If the user wants to compile only a subset of these packages, then refer to the appendix [Compiling Boltzmann, CosmoLike and Likelihood codes separatelly](#appendix_compile_separatelly).

## Running Cobaya Examples <a name="cobaya_base_code_examples"></a>

Assuming the user opted for the easier *Conda installation* and located the terminal at the folder *where Cocoa was cloned*, this is how to run some example YAML files we provide (*no Cosmolike code involved*): 

**Step 1 of 5**: activate the conda environment

    $ conda activate cocoa
     
**Step 2 of 5**: go to the Cocoa main folder 

    $(cocoa) cd ./cocoa/Cocoa

**Step 3 of 5**: activate the private python environment

    $(cocoa) source start_cocoa

Users will see a terminal that looks like this: `$(Cocoa)(.local)`. *This is a feature, not a bug*! 

Why did we choose to have two separate bash environments? Users should be able to manipulate multiple Cocoa instances seamlessly, which is particularly useful when running chains in one instance while experimenting with code development in another. Consistency of the environment across all Cocoa instances is crucial, and the start_cocoa/stop_cocoa scripts handle the loading and unloading of environmental path variables for each Cocoa. All of them, however, depends on many of the same prerequisites, so it is advantageous to maintain the basic packages inside the shared conda cocoa environment. 

**Step 4 of 5**: select the number of OpenMP cores
    
    $(cocoa)(.local) export OMP_PROC_BIND=close; export OMP_NUM_THREADS=4

**Step 5 of 5**: run `cobaya-run` on a the first example YAML files we provide.

One model evaluation:

     $(cocoa)(.local) mpirun -n 1 --mca btl tcp,self --bind-to core:overload-allowed --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run ./projects/example/EXAMPLE_EVALUATE1.yaml -f
     
MCMC:

     $(cocoa)(.local) mpirun -n 4 --mca btl tcp,self --bind-to core:overload-allowed --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run ./projects/example/EXAMPLE_MCMC1.yaml -f

(**expert**) Why the `--mca btl tcp,self` flag? Conda-forge developers don't [compile OpenMPI with Infiniband compatibility](https://github.com/conda-forge/openmpi-feedstock/issues/38).

(**expert**) Why the `--bind-to core:overload-allowed --map-by numa:pe=${OMP_NUM_THREADS}` flag? This flag enables efficient hybrid MPI + OpenMP runs on NUMA architecture.

Once the work is done, type:

    $(cocoa)(.local) source stop_cocoa
    $(cocoa) conda deactivate cocoa

## Running Cosmolike projects <a name="running_cosmolike_projects"></a> 

The *projects* folder was designed to include Cosmolike projects. Similar to the previous section, we assume the user opted for the more direct *Conda installation* method. We also presume the user's terminal is in the folder where Cocoa was cloned.

**Step 1 of 5**: activate the Conda Cocoa environment
    
    $ conda activate cocoa

**Step 2 of 5**: go to the project folder (`./cocoa/Cocoa/projects`) and clone a Cosmolike project, with fictitious name `XXX`:
    
    $(cocoa) cd ./cocoa/Cocoa/projects
    $(cocoa) $CONDA_PREFIX/bin/git clone git@github.com:CosmoLike/cocoa_XXX.git XXX

By convention, the Cosmolike Organization hosts a Cobaya-Cosmolike project named XXX at `CosmoLike/cocoa_XXX`. However, our provided scripts and template YAML files assume the removal of the `cocoa_` prefix when cloning the repository. The prefix `cocoa_` on the Cosmolike organization avoids mixing Cobaya-Cosmolike projects with code meant to be run on the legacy CosmoLike code.

Example of cosmolike projects: [lsst_y1](https://github.com/CosmoLike/cocoa_lsst_y1).
 
**Step 3 of 5**: go back to Cocoa main folder, and activate the private python environment
    
    $(cocoa) cd ../
    $(cocoa) source start_cocoa
 
Remember to run the start_cocoa script only after cloning the project repository is essential. The script *start_cocoa* creates necessary symbolic links and also adds the *Cobaya-Cosmolike interface* of all projects to `LD_LIBRARY_PATH` and `PYTHONPATH` paths.

**Step 4 of 5**: compile the project
 
     $(cocoa)(.local) source ./projects/XXX/scripts/compile_XXX
  
 **Step 5 of 5**: select the number of OpenMP cores and run a template yaml file
    
    $(cocoa)(.local) export OMP_PROC_BIND=close; export OMP_NUM_THREADS=4
    $(cocoa)(.local) mpirun -n 1 --mca btl tcp,self --bind-to core --rank-by core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run ./projects/XXX/EXAMPLE_EVALUATE1.yaml -f

(**warning**) Be careful when creating YAML for weak lensing projects in Cobaya using the $\Omega_m/\Omega_b$ parameterization. See Appendix [warning about weak lensing YAML files](#appendix_example_runs) for further details.

## Appendix <a name="appendix"></a>

### Proper Credits <a name="appendix_proper_credits"></a>

The following is not an exhaustive list of the codes we use

- [Cobaya](https://github.com/CobayaSampler) is a framework developed by Dr. Jesus Torrado and Prof. Anthony Lewis

- [Cosmolike](https://github.com/CosmoLike) is a framework developed by Prof. Elisabeth Krause and Prof. Tim Eifler

- [CAMB](https://github.com/cmbant/CAMB) is a Boltzmann code developed by Prof. Anthony Lewis

- [CLASS](https://github.com/lesgourg/class_public) is a Boltzmann code developed by Prof. Julien Lesgourgues and Dr. Thomas Tram

- [Polychord](https://github.com/PolyChord/PolyChordLite) is a sampler code developed by Dr. Will Handley, Prof. Lasenby, and Prof. M. Hobson

By no means, we want to discourage people from cloning code from their original repositories. We've included these codes as compressed [xz file format](https://tukaani.org/xz/format.html) in our repository for convenience in the initial development. The work of those authors is extraordinary, and they must be properly cited.

### The whovian-cocoa docker container <a name="appendix_jupyter_whovian"></a>

We provide the docker image [whovian-cocoa](https://hub.docker.com/r/vivianmiranda/whovian-cocoa) to facilitate the installation of Cocoa on Windows and MacOS. This appendix assumes the users already have the docker engine installed on their local PC. For instructions on installing the docker engine in specific operating systems, please refer to [Docker's official documentation](https://docs.docker.com/engine/install/). 

To download and run the container for the first time, type:

    $ docker run --platform linux/amd64 --hostname cocoa --name cocoa2023 -it -p 8080:8888 -v $(pwd):/home/whovian/host/ -v ~/.ssh:/home/whovian/.ssh:ro vivianmiranda/whovian-cocoa

The flag `-v $(pwd):/home/whovian/host/` ensures that the files on the host computer, where the user should install Cocoa to avoid losing work in case the docker image needs to be updated, have been mounted to the directory `/home/whovian/host/`. 

    whovian@cocoa:~$ cd /home/whovian/host/; ls

the user should see the host files of the directory where the whovian-cocoa container was initialized.

When running the container the first time, the user needs to init conda with `conda init bash` followed by `source ~/.bashrc`, as shown below.

    whovian@cocoa:~$ conda init bash
    whovian@cocoa:~$ source ~/.bashrc

The container already comes with conda Cocoa environment pre-installed:

    $ whovian@cocoa:~$ conda activate cocoa

When the user exits the container, how to restart it? Type 
    
    $ docker start -ai cocoa2023

How to run Jupyter Notebooks remotely when using Cocoa within the whovian-cocoa container? First, type the following command:

    whovian@cocoa:~$ jupyter notebook --no-browser

The terminal will show a message similar to the following template:

    [... NotebookApp] Writing notebook server cookie secret to /home/whovian/.local/share/jupyter/runtime/notebook_cookie_secret
    [... NotebookApp] WARNING: The notebook server is listening on all IP addresses and not using encryption. This is not recommended.
    [... NotebookApp] Serving notebooks from local directory: /home/whovian/host
    [... NotebookApp] Jupyter Notebook 6.1.1 is running at:
    [... NotebookApp] http://f0a13949f6b5:8888/?token=XXX
    [... NotebookApp] or http://127.0.0.1:8888/?token=XXX
    [... NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).

Below, we assume the user runs the container in a server with the URL `your_sever.com`. We also presume the server can be accessed via ssh protocol. From a local PC, type:

    $ ssh your_username@your_sever.com -L 8080:localhost:8080

Finally, go to a browser and type `http://localhost:8080/?token=XXX`, where `XXX` is the previously saved token.

### Miniconda Installation <a name="overview_miniconda"></a>

Download and run Miniconda installation script (please adapt `CONDA_DIR`):

    export CONDA_DIR=/gpfs/home/vinmirandabr/miniconda

    mkdir $CONDA_DIR

    wget https://repo.continuum.io/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh
    
    /bin/bash Miniconda3-py38_4.12.0-Linux-x86_64.sh -f -b -p $CONDA_DIR

After installation, users must source conda configuration file

    source $CONDA_DIR/etc/profile.d/conda.sh \
    && conda config --set auto_update_conda false \
    && conda config --set show_channel_urls true \
    && conda config --set auto_activate_base false \
    && conda config --prepend channels conda-forge \
    && conda config --set channel_priority strict \
    && conda init bash
    
### Compiling Boltzmann, CosmoLike and Likelihood codes separatelly <a name="appendix_compile_separatelly"></a>

To avoid excessive compilation times during development, users can use specialized scripts located at `Cocoa/installation_scripts/` that compile only a specific module. A few examples of these scripts are: 

    $(cocoa)(.local) source ./installation_scripts/compile_class
    $(cocoa)(.local) source ./installation_scripts/compile_camb
    $(cocoa)(.local) source ./installation_scripts/compile_planck
    $(cocoa)(.local) source ./installation_scripts/compile_act
    $(cocoa)(.local) source ./installation_scripts/setup_polychord
    
### Warning about Weak Lensing YAML files in Cobaya <a name="appendix_example_runs"></a>

The CosmoLike pipeline takes $\Omega_m$ and $\Omega_b$, but the CAMB Boltzmann code only accepts $\Omega_c h^2$ and $\Omega_b h^2$ in Cobaya. Therefore, there are two ways of creating YAML compatible with CAMB and Cosmolike: 

1. CMB parameterization and  $\Omega_m$ and $\Omega_b$ as derived parameters.

        omegabh2:
            prior:
                min: 0.01
                max: 0.04
            ref:
                dist: norm
                loc: 0.022383
                scale: 0.005
            proposal: 0.005
            latex: \Omega_\mathrm{b} h^2
        omegach2:
            prior:
                min: 0.06
                max: 0.2
            ref:
                dist: norm
                loc: 0.12011
                scale: 0.03
            proposal: 0.03
            latex: \Omega_\mathrm{c} h^2
        mnu:
            value: 0.06
            latex: m_\\nu
        omegam:
            latex: \Omega_\mathrm{m}
        omegamh2:
            derived: 'lambda omegam, H0: omegam*(H0/100)**2'
            latex: \Omega_\mathrm{m} h^2
        omegab:
            derived: 'lambda omegabh2, H0: omegabh2/((H0/100)**2)'
            latex: \Omega_\mathrm{b}
        omegac:
            derived: 'lambda omegach2, H0: omegach2/((H0/100)**2)'
            latex: \Omega_\mathrm{c}

2. Weak Lensing parameterization and  $\Omega_c h^2$ and $\Omega_b h^2$ as derived parameters.

Adopting $\Omega_m$ and $\Omega_b$ as main MCMC parameters can create a silent bug in Cobaya. The problem occurs when the option `drop: true` is absent in $\Omega_m$ and $\Omega_b$ parameters, and there are no expressions that define the derived $\Omega_c h^2$/$\Omega_b h^2$ parameters. The bug is silent because the MCMC runs without any warnings, but the CAMB Boltzmann code does not update the cosmological parameters at every MCMC iteration. As a result, the resulting posteriors are flawed, but they may seem reasonable to those unfamiliar with the issue. It's important to be aware of this bug to avoid any potential inaccuracies in the results. 

The correct way to create YAML files with $\Omega_m$ and $\Omega_b$ as main MCMC parameters is exemplified below

        omegab:
            prior:
                min: 0.03
                max: 0.07
            ref:
                dist: norm
                loc: 0.0495
                scale: 0.004
            proposal: 0.004
            latex: \Omega_\mathrm{b}
            drop: true
        omegam:
            prior:
                min: 0.1
                max: 0.9
            ref:
                dist: norm
                loc: 0.316
                scale: 0.02
            proposal: 0.02
            latex: \Omega_\mathrm{m}
            drop: true
        mnu:
            value: 0.06
            latex: m_\\nu
        omegabh2:
            value: 'lambda omegab, H0: omegab*(H0/100)**2'
            latex: \Omega_\mathrm{b} h^2
        omegach2:
            value: 'lambda omegam, omegab, mnu, H0: (omegam-omegab)*(H0/100)**2-(mnu*(3.046/3)**0.75)/94.0708'
            latex: \Omega_\mathrm{c} h^2
        omegamh2:
            derived: 'lambda omegam, H0: omegam*(H0/100)**2'
            latex: \Omega_\mathrm{m} h^2

### Manual Blocking of Cosmolike Parameters <a name="manual_blocking_cosmolike"></a>

Cosmolike Weak Lensing pipeline contains parameters with different speed hierarchies. For example, Cosmolike execution time is reduced by approximately 50% when fixing the cosmological parameters. When varying only multiplicative shear calibration, Cosmolike execution time is reduced by two orders of magnitude. 

Cobaya can't automatically handle parameters associated with the same likelihood that have different speed hierarchies. Luckily, we can manually impose the speed hierarchy in Cobaya using the `blocking:` option. The only drawback of this method is that parameters of all adopted likelihoods need to be manually specified, not only the ones required by Cosmolike.

In addition to that, Cosmolike can't cache the intermediate products of the last two evaluations, which is necessary to exploit optimizations associated with dragging (`drag: True`). However, Cosmolike caches the intermediate products of the previous evaluation, thereby enabling the user to take advantage of the slow/fast decomposition of parameters in Cobaya's main MCMC sampler. 

Below we provide an example YAML configuration for an MCMC chain that with DES 3x2pt likelihood.

        likelihood: 
            des_y3.des_3x2pt:
            path: ./external_modules/data/des_y3
            data_file: DES_Y1.dataset
         
         (...)
         
        sampler:
            mcmc:
                covmat: "./projects/des_y3/EXAMPLE_MCMC22.covmat"
                covmat_params:
                # ---------------------------------------------------------------------
                # Proposal covariance matrix learning
                # ---------------------------------------------------------------------
                learn_proposal: True
                learn_proposal_Rminus1_min: 0.03
                learn_proposal_Rminus1_max: 100.
                learn_proposal_Rminus1_max_early: 200.
                # ---------------------------------------------------------------------
                # Convergence criteria
                # ---------------------------------------------------------------------
                # Maximum number of posterior evaluations
                max_samples: .inf
                # Gelman-Rubin R-1 on means
                Rminus1_stop: 0.02
                # Gelman-Rubin R-1 on std deviations
                Rminus1_cl_stop: 0.2
                Rminus1_cl_level: 0.95
                # ---------------------------------------------------------------------
                # Exploiting Cosmolike speed hierarchy
                # ---------------------------------------------------------------------
                measure_speeds: False
                drag: False
                oversample_power: 0.2
                oversample_thin: True
                blocking:
                - [1,
                    [
                        As_1e9, H0, omegab, omegam, ns
                    ]
                  ]
                - [4,
                    [
                        DES_DZ_S1, DES_DZ_S2, DES_DZ_S3, DES_DZ_S4, DES_A1_1, DES_A1_2,
                        DES_B1_1, DES_B1_2, DES_B1_3, DES_B1_4, DES_B1_5,
                        DES_DZ_L1, DES_DZ_L2, DES_DZ_L3, DES_DZ_L4, DES_DZ_L5
                    ]
                  ]
                - [25,
                    [
                        DES_M1, DES_M2, DES_M3, DES_M4, DES_PM1, DES_PM2, DES_PM3, DES_PM4, DES_PM5
                    ]
                  ]
                # ---------------------------------------------------------------------
                max_tries: 10000
                burn_in: 0
                Rminus1_single_split: 4
                
### Setting-up conda environment for Machine Learning emulators <a name="ml_emulators"></a>

If the user wants to add Tensorflow, Keras and Pytorch for an emulator-based project via Conda, then type

        $ conda activate cocoa 
      
        $(cocoa) $CONDA_PREFIX/bin/pip install --no-cache-dir \
            'pyDOE2==1.2.1' \
            'gpy==1.10' \
            'tensorflow-cpu==2.8.0' \
            'keras==2.8.0' \
            'keras-preprocessing==1.1.2' \
            'torch==1.11.0+cpu' \
            'torchvision==0.12.0+cpu' -f https://download.pytorch.org/whl/torch_stable.html

In case there are GPUs available, the following commands will install the GPU version of 
Tensorflow, Keras and Pytorch.

        $(cocoa) CONDA_PREFIX/bin/pip install --no-cache-dir \
            'pyDOE2==1.2.1' \
            'gpy==1.10' \
            'tensorflow==2.8.0' \
            'keras==2.8.0' \
            'keras-preprocessing==1.1.2' \
            'torch==1.11.0' \
            'torchvision==0.12.0' -f https://download.pytorch.org/whl/torch_stable.html

Based on our experience, we recommend utilizing the GPU versions to train the emulator while using the CPU versions to run the MCMCs. This is because our supercomputers possess a greater number of CPU-only nodes. It may be helpful to create two separate conda environments for this purpose. One could be named `cocoa` (CPU-only), while the other could be named `cocoaemu` and contain the GPU versions of the machine learning packages.

For users that opted for the manual installation via Cocoa's internal cache, commenting out the environmental flags shown below, located at *set_installation_options* script, will enable the installation of machine-learning-related libraries via pip.  

        # IF TRUE, THEN COCOA WON'T INSTALL TENSORFLOW, KERAS and PYTORCH
        #export IGNORE_EMULATOR_CPU_PIP_PACKAGES=1
        #export IGNORE_EMULATOR_GPU_PIP_PACKAGES=1

Unlike most installed pip prerequisites, cached at `cocoa_installation_libraries/pip_cache.xz`, installing the Machine Learning packages listed above will require an active internet connection.
