# Simulation-based Inference for Model Parameterization on Analog Neuromorphic Hardware

In this repository we collect all code which is needed to replicate the results in [Kaiser et al. (2023)](#Kaiser2023simulation).

The repository is structured as follows:
- `src/py/paper_sbi`: Python library.
- `src/py/paper_sbi/scripts`: Experiment and visualization scripts.
- `tests`: Hardware and software tests.

To generate an approximated posterior the script `attenuation_sbi.py` can be executed.
It takes experiment data from which a target observation can be extracted as an argument.
Such experiment data can be recorded with `record_variations.py` which can be found in the repository `model-hw-mc-attenuation`.

Further scripts for visualizing the posteriors or samples drawn from the posterior can be found in the repository `paramopt`.

## Example: How to execute the SNPE algorithm and visualize posterior samples
In order to perform emulations on the BrainScaleS-2 system, you need access to the cluster of the Electronic Vision(s) group.
Please contact us, if you want access.
Without hardware access you can still execute the simulations in [Arbor](https://arbor-sim.org/).

In the following example we will perform an experiment similar to Figure 3 in [Kaiser et al. (2023)](#Kaiser2023simulation).
We will restrict the parameter space to two dimension and look at the decay constant as an observable.
Furthermore, we will perform three approximation rounds with 50 simulations in each round.
In contrast to the publication, we use the default neural density estimator in the current example.

As stated in [How to build](#build-and-runtime-dependencies), all runtime dependencies are encapsulated in a [Singularity Container](https://sylabs.io/docs/).
All commands in the following list have to be executed in the appropriate singularity container (`singularity exec --app dls /containers/stable/latest ...`).
When you want to perform experiments on BrainScaleS-2 you have to allocate hardware resources (`srun -p cube --wafer XX --fpga-without-fpga X ...`)

1. Build/install the software as described in [How to build](#how-to-build).
1. Record a target:
    - execute `record_variations.py` (`record_variations_arbor.py` for arbor); these scripts are part of the repository `model-hw-mc-attenuation`,
    - the file `attenuation_variations.pkl` contains the experiment results.
1. Approximation of the posterior:
    - execute `attenuation_sbi.py attenuation_variations.pkl -observation length_constant -n_sim_first 50 -n_sim_rest 50 -n_rounds 3 --global_parameters`,
    - the file `sbi_samples.pkl` contains the samples which were drawn while executing the SNPE algorithm,
    - the file `posteriors.pkl` contains the approximated posteriors of each round.
1. Draw samples from the last approximated posterior:
    - execute `sbi_draw_posterior_samples.py posteriors.pkl sbi_samples.pkl` (script is part of the repository `paramopt`),
    - the file `posterior_samples_2.pkl` contains samples drawn from the last posterior.
1. Visualize the drawn posterior:
    - execute `plot_sbi_pairplot.py posterior_samples_2.pkl` (script is part of the repository `paramopt`),
    - the file `pairplot.png` contains a pairplot of the posterior samples (Figure 3D in [Kaiser et al. (2023)](#Kaiser2023simulation)).


## How to build
### Build- and runtime dependencies
All build- and runtime dependencies are encapsulated in a [Singularity Container](https://sylabs.io/docs/).
If you want to build this project outside the Electronic Vision(s) cluster, please start by downloading the most recent version from [here](https://openproject.bioai.eu/containers/).

For all following steps, we assume that the most recent Singularity container is located at `/containers/stable/latest` – you are free to choose any other path.

### Github-based build
To build this project from public resources, adhere to the following guide:

```shell
# 1) Most of the following steps will be executed within a singularity container
#    To keep the steps clutter-free, we start by defining an alias
shopt -s expand_aliases
alias c="singularity exec --app dls /containers/stable/latest"

# 2) Add the cross-compiler and toolchain for the embedded processor to your environment
#    If you don't have access to the module, you may build it as noted here:
#    https://github.com/electronicvisions/oppulance
module load ppu-toolchain

# 2) Prepare a fresh workspace and change directory into it
mkdir workspace && cd workspace

# 3) Fetch a current copy of the symwaf2ic build tool
git clone https://github.com/electronicvisions/waf -b symwaf2ic symwaf2ic

# 4) Build symwaf2ic
c make -C symwaf2ic
ln -s symwaf2ic/waf

# 5) Setup your workspace and clone all dependencies (--clone-depth=1 to skip history)
c ./waf setup --repo-db-url=https://github.com/electronicvisions/projects --project=model-paper-mc-sbi --project=paramopt --project=model-hw-mc-attenuation

# 6) Build the project
#    Adjust -j1 to your own needs, beware that high parallelism will increase memory consumption!
c ./waf configure
c ./waf build -j1

# 7) Install the project to ./bin and ./lib
c ./waf install

# 8) If you run programs outside waf, you'll need to add ./lib and ./bin to your path specifications
export SINGULARITYENV_PREPEND_PATH=`pwd`/bin:$SINGULARITYENV_PREPEND_PATH
export SINGULARITYENV_LD_LIBRARY_PATH=`pwd`/lib:$SINGULARITYENV_LD_LIBRARY_PATH
export LD_LIBRARY_PATH=`pwd`/lib:$LD_LIBRARY_PATH
export PYTHONPATH=`pwd`/lib:$PYTHONPATH
export PATH=`pwd`/bin:$PATH
```

### On the Electronic Vision(s) Cluster

* Work on the frontend machine, `helvetica`. You should have received instructions how to connect to it.
* Follow [aforementioned instructions](#github-based-build) with the following simplifications
  * Replace **steps 3) and 4)** by `module load waf`
  * Make sure to run **step 6)** within a respective slurm allocation: Prefix `srun -p compile -c8`; depending on your shell, you might need to roll out the `c`-alias.
  * Replace **step 8)** by `module load localdir`.

### Build from internal sources

If you have access to the internal *Gerrit* server, you may drop the `--repo-db-url`-specification in **step 5)** of [aforementioned instructions](#github-based-build).

## Acknowledgment
The software in this repository has been developed by staff and students
of Heidelberg University as part of the research carried out by the
Electronic Vision(s) group at the Kirchhoff-Institute for Physics.
The research is funded by Heidelberg University, the State of
Baden-Württemberg, the European Union Sixth Framework Programme no.
15879 (FACETS), the Seventh Framework Programme under grant agreements
no 604102 (HBP), 269921 (BrainScaleS), 243914 (Brain-i-Nets), the
Horizon 2020 Framework Programme under grant agreement 720270, 785907, 945539 (HBP) as
well as from the Manfred Stärk Foundation.

## License
```
Experiment Code for "Simulation-based Inference for Model Parameterization on 
Analog Neuromorphic Hardware" ('model-paper-mc-sbi')

Copyright (C) 2023–2023 Electronic Vision(s) Group
                        Kirchhoff-Institute for Physics
                        Ruprecht-Karls-Universität Heidelberg
                        69120 Heidelberg
                        Germany

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
USA
```

## References
<a id="Kaiser2023simulation">Kaiser et al. (2023)</a> 
"Simulation-based Inference for Model Parameterization on Analog Neuromorphic Hardware"
In: arXiv preprint. doi: [10.48550/arXiv.2303.16056](https://doi.org/10.48550/arXiv.2303.16056).
