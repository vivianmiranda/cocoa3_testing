# Table of contents
1. [Overview of the Cobaya-CosmoLike Joint Architecture (Cocoa)](#overview)
3. [Installation of Cocoa's required packages](#required_packages)
    1. [Via Conda](#required_packages_conda)
    3. [Via Cocoa's internal cache](#required_packages_cache)
4. [Installation of Cobaya base code](#cobaya_base_code)
5. [Running Cobaya Examples](#cobaya_base_code_examples)
6. [Running Cosmolike projects](#running_cosmolike_projects)
7. [Creating Cosmolike projects](#creating_cosmolike_projects)
8. [Appendix](#appendix)
    1. [Proper Credits](#appendix_proper_credits)
    2. [The whovian-cocoa docker container](#appendix_jupyter_whovian)
    3. [Miniconda Installation](#overview_miniconda)
    4. [Compiling Boltzmann, CosmoLike and Likelihood codes separatelly](#appendix_compile_separatelly)
    5. [Warning about Weak Lensing YAML files](#appendix_example_runs)
9. [The projects folder (external readme)](https://github.com/SBU-UNESP-2022-COCOA/cocoa2/tree/main/Cocoa/projects)
10. [Adapting new modified CAMB/CLASS (external readme)](https://github.com/SBU-UNESP-2022-COCOA/cocoa2/tree/main/Cocoa/external_modules/code)
 
## Overview of the [Cobaya](https://github.com/CobayaSampler)-[CosmoLike](https://github.com/CosmoLike) Joint Architecture (Cocoa) <a name="overview"></a>

Cocoa allows users to run [CosmoLike](https://github.com/CosmoLike) routines inside the [Cobaya](https://github.com/CobayaSampler) framework. [CosmoLike](https://github.com/CosmoLike) can analyze data primarily from the [Dark Energy Survey](https://www.darkenergysurvey.org) and simulate future multi-probe analyses for LSST and Roman Space Telescope. Besides integrating [Cobaya](https://github.com/CobayaSampler) and [CosmoLike](https://github.com/CosmoLike), Cocoa introduces shell scripts and readme instructions that allow users to containerize [Cobaya](https://github.com/CobayaSampler). The container structure ensures that users will adopt the same compiler and libraries (including their versions), and that they will be able to use multiple [Cobaya](https://github.com/CobayaSampler) instances consistently. This readme file presents basic and advanced instructions for installing all [Cobaya](https://github.com/CobayaSampler) and [CosmoLike](https://github.com/CosmoLike) components.

## Installation of Cocoa's required packages <a name="required_packages"></a>

There are two installation methods. Users must choose one of them:

1. [Via Conda](#required_packages_conda) (easier, best)
2. [Via Cocoa's internal cache](#required_packages_cache) (slow, not advisable) 

We also provide the docker image whovian-cocoa to promote the installation of Cocoa on Windows/MacOS. See Appendix [The whovian-cocoa docker container](#appendix_jupyter_whovian) for further instructions.

### Via Conda <a name="required_packages_conda"></a>

We assume here the user has previously installed either [Minicoda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/individual). If this is not the case, then refer to the [Miniconda Installation](#overview_miniconda) appendix for further instructions.

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
       'conda-forge::matplotlib=3.5.0' \
       'conda-forge::astropy=4.3.1'
      
If the user wants to add Tensorflow, Keras and Pytorch for an emulator-based project, then type
 
      $ conda activate cocoa 
      
      $(cocoa) $CONDA_PREFIX/bin/pip install --no-cache-dir \
        'tensorflow-cpu==2.8.0' \
        'keras==2.8.0' \
        'keras-preprocessing==1.1.2' \
        'torch==1.11.0+cpu' \
        'torchvision==0.12.0+cpu' -f https://download.pytorch.org/whl/torch_stable.html
 
With this installation method, users must activate the Conda environment whenever working with Cocoa, as shown below 

    $ conda activate cocoa
    
When loading conda cocoa environment for the first time, users must also install git-lfs

    $(cocoa) $CONDA_PREFIX/bin/git-lfs install

**Users can now proceed to the section [Installation of Cobaya base code](#cobaya_base_code)** 

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
   
Our scripts never install packages on `$HOME/.local`. All requirements for Cocoa are installed at

    Cocoa/.local/bin
    Cocoa/.local/include
    Cocoa/.local/lib
    Cocoa/.local/share

**Users can now proceed to the section [Installation of Cobaya base code](#cobaya_base_code)**

## Installation of Cobaya base code <a name="cobaya_base_code"></a>

Assuming the user opted for the easier *Conda installation*, type:

    $ conda activate cocoa
    
    $(cocoa) $CONDA_PREFIX/bin/git-lfs clone https://github.com/CosmoLike/cocoa.git

to clone the repository. Cocoa developers with set ssh keys in GitHub should instead use the command

    $(cocoa) $CONDA_PREFIX/bin/git-lfs clone git@github.com:CosmoLike/cocoa.git

(**Warning**) We have a limited monthly quota in bandwidth for Git LFS files, and therefore we ask users to use good judgment in the number of times they clone files from Cocoa's main repository.
 
Cocoa is made aware of the chosen installation method of required packages via special environment keys located on the *set_installation_options* script (Cocoa/ subdirectory), as shown below:

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

    $(cocoa) source setup_cocoa_installation_packages

This script decompresses the data files and installs any remaining necessary packages. File decompression should only take a few minutes. Depending on the installation method, package installation time ranges from a few minutes (installation via Conda) to more than one hour (installation via Cocoa's internal cache). Finally, type

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

(**expert**) Why did we choose to have two environments? The Conda environment should be the same for all Cocoa instances, with [start_cocoa](https://github.com/SBU-Josh/cocoa2/blob/main/Cocoa/start_cocoa)/[stop_cocoa](https://github.com/SBU-Josh/cocoa2/blob/main/Cocoa/stop_cocoa) scripts loading/unloading the appropriate `LD_LIBRARY_PATH`, `CPATH`, `C_INCLUDE_PATH`, `CPLUS_INCLUDE_PATH` and `PATH` variables associated with each instance. While users may be running chains in one Cocoa instance, they can install and use a second Cocoa instantiation to make experimental changes.

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

Example of cosmolike projects: [lsst_y1](https://github.com/CosmoLike/cocoa_lsst_y1), [des_y3](https://github.com/CosmoLike/cocoa_des_y3).
 
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

## Creating Cosmolike projects <a name="creating_cosmolike_projects"></a> 

The `XXX` project needs to have more or less the following structure

    +-- cocoa_des_y3
    |    +-- likelihood
    |    |   +-- _cosmolike_prototype_base.py
    |    |   +-- des_3x2pt.py
    |    |   +-- des_3x2pt.yaml
    |    |   +-- des_2x2pt.py
    |    |   +-- des_3x2pt.yaml
    |    |   +-- des_cosmic_shear.py
    |    |   +-- des_cosmic_shear.yaml
    |    +-- scripts
    |    |   +-- compile_des_y3
    |    |   +-- start_des_y3
    |    |   +-- stop_des_y3
    |    +-- data
    |    |   +-- DES.paramnames
    |    |   +-- DES_Y3.dataset
    |    |   +-- datavector.txt
    |    |   +-- covariance.txt
    |    |   +-- nzlens.txt
    |    |   +-- nzsource.txt
    |    |   +-- mask.mask
    |    +-- interface
    |    |   +-- MakefileCosmolike
    |    |   +-- cosmolike_des_y3_interface.py
    |    |   +-- interface.cpp
    |    |   +-- interface.hpp
    |    +-- chains
    |    |   +-- README

We expect to have a public project that can be used as a template for new Cosmolike projects in the near future.

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

We provide the docker image whovian-cocoa to promote the installation of Cocoa on Windows/MacOS. This docker image assumes the user already has the docker engine installed on their local PC. For instructions on installing the docker engine on your specific operating system, please refer to [Docker's official documentation](https://docs.docker.com/engine/install/). 

To download and run the container for the first time, type:

    $ docker run --platform linux/amd64 --hostname cocoa --name cocoa2023 -it -p 8080:8888 -v $(pwd):/home/whovian/host/ -v ~/.ssh:/home/whovian/.ssh:ro vivianmiranda/whovian-cocoa

The flag `-v $(pwd):/home/whovian/host/` ensures that the files on the host computer where the user plans to install Cocoa have been properly mounted to the directory `/home/whovian/host/`.  After executing the command

    whovian@cocoa:~$ cd /home/whovian/host/; ls

the user should see the host files on the screen.

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
    
### Examples of Weak Lensing Runs <a name="appendix_example_runs"></a>

We have provided examples of Weak Lensing runs within specific projects such as:

- [LSST-Y1](https://github.com/CosmoLike/cocoa_lsst_y1/blob/main/EXAMPLE_MCMC1.yaml)
- [DES-Y3](https://github.com/CosmoLike/cocoa_des_y3/blob/main/EXAMPLE_MCMC1.yaml)

(**Warning**) The CosmoLike pipeline takes $\Omega_m$ and $\Omega_b$ where as camb **ONLY accepts** $\Omega_c h^2$ and $\Omega_b h^2$. In the YAMLs we provided, we included the option `drop: true` and the derived expressions to convert from $\Omega_m$/$\Omega_b$ to $\Omega_c h^2$/$\Omega_b h^2$. **Be aware that if this conversion is not present in the YAML file, a silent/terrible bug will be created**: the chains will continue to run without the matter density being updated on CAMB. Always ensure that the conversion is included in your YAML file when adopting the  $\Omega_m$/$\Omega_b$ parameterization.
