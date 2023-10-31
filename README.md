# Alpaca Use Case

Repository containing the code necessary to reproduce the results of the 
Alpaca toolbox manuscript.


## Table of contents
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Code repository](#code-repository)
  - [How to run](#how-to-run)
  - [Outputs](#outputs)
  - [Acknowledgments](#acknowledgments)
  - [License](#license)


## Prerequisites


### Clone the repository to a local folder

This repository must be cloned to a local folder. This can be done using the 
`git` CLI client:

```bash
git clone [URL redacted for double-blind review]
```


### Data

To run the analyses, the public experimental datasets availabe at 
[https://doi.gin.g-node.org/10.12751/g-node.f83565](https://doi.gin.g-node.org/10.12751/g-node.f83565)
must be downloaded.

The scripts use the datasets in the NIX format (including the 30 kHz neural
signal), and the versioned files are accessible using the links:

* **i140703-001.nix**: <https://gin.g-node.org/INT/multielectrode_grasp/raw/to_nix/datasets_nix/i140703-001.nix>
* **l101210-001.nix**: <https://gin.g-node.org/INT/multielectrode_grasp/raw/to_nix/datasets_nix/l101210-001.nix>

You can also follow the [instructions on the GIN repository](https://gin.g-node.org/INT/multielectrode_grasp)
to download the files to a local repository folder using the `gin` client.

The NIX files must be downloaded/copied into the folder `/data` with respect
to the root of this repository. This allows running the analyses using the
`bash` scripts that are provided with each Python script. If downloaded using
the `gin` client, a symbolic link can be created to the path where the GIN
repository was cloned in your system (subfolder `datasets_nix`):

```bash
ln -s /path/to/multielectrode_grasp/datasets_nix ./data
```


### Requirements

Project requires Python 3.9 and the following packages:

- conda
- pip
- scipy
- numpy
- matplotlib
- nixio
- neo
- elephant
- odml
- alpaca

The code was run using Ubuntu 18.04.6 LTS 64-bit and `conda` 22.9.0.


## Installation

The required environments can be created using `conda`, using the templates in
the `/environment` folder. For instructions on how to install `conda` in your
system, please check the `conda` [documentation](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). 

For convenience, all necessary environments can be created using a `bash`
script (this rewrites existing versions of the environments):

```bash
cd environment
./build_envs.sh
```

For visualization of provenance graphs as GEXF files, Gephi 0.9.7 (build 
202208031831) was used. The instructions for downloading and installing are
found in the Installation [URL redacted for double-blind review] section of the 
[URL redacted for double-blind review].


## Code repository

The code is organized into subfolders inside the `/code` folder:

* `no_provenance`: this is the original PSD analysis presented as use case
   in the paper. The analysis is implemented in `psd_by_trial_type.py`. The
   flowchart presented in Figure 4 was constructed based on this script.

* `provenance`: `psd_by_trial_type.py` is the code in `no_provenance` modified 
  to use Alpaca to track provenance. The `generate_gexf_from_prov.py` script 
  generates several visualization graphs, with different levels of 
  simplification, using the provenance information saved as a Turtle file.

* `smoothed_plot`: `psd_by_trial_type.py` contains a modification of the 
  original analysis to produce a smoothed version of the PSD plot. The 
  `generate_gexf_from_prov.py` script generates simplified visualization 
  graphs from the provenance information saved as a Turtle file.


## How to run

To run the code, the correct environment must be activated with `conda` and
the scripts run using the provided `bash` scripts:

* `no_provenance`:

   To run the analysis:

   ```bash
   cd code/no_provenance
   conda activate no_provenance
   ./run.sh
   ```

* `provenance`:

   To run the analysis:

   ```bash
   cd code/provenance
   conda activate provenance
   ./run.sh
   ```
  
   After the analysis is run, generate the GEXF graphs to visualize provenance
   with Gephi:

   ```bash
   ./visualize_provenance.sh
   ```

* `smoothed_plot`:

   To run the analysis and generate the GEXF graphs at the end:

   ```bash
   cd code/smoothed_plot
   conda activate provenance
   ./run.sh
   ```

## Outputs

The outputs presented in the paper are available in the `/outputs` folder.
**[Files were removed for double-blind review, as they identified the authors]**

The bash scripts write files to the `/outputs` folder, with respect to
the root of this repository. There are subfolders with the same names as
in `/code`, each with the respective outputs:

* [outputs/no_provenance](outputs/no_provenance): plot in 
  `R2G_PSD_all_subjects.png` is the Figure 3 in the paper.  
* [outputs/provenance](outputs/provenance): one of the outputs is the
  `R2G_PSD_all_subjects.ttl` file, that contains the provenance information 
  used to generate the graphs presented in Figures 11 and 12 in the paper
  (specific details below). Several visualization graphs as GEXF files are 
  provided. The plot in `R2G_PSD_all_subjects.png` is the analysis output, 
  equivalent to the one generated by `/code/no_provenance`.
* [outputs/smoothed_plot](outputs/smoothed_plot): the outputs are presented as 
  Figure 13 in the paper. The output file `R2G_PSD_all_subjects.png` was used 
  in Figure 13A. The `R2G_PSD_all_subjects.ttl` file was used to generate the 
  visualization graph presented in Figure 13B.  


The specific GEXF graph outputs used for the figures in the paper are:

* Figure 11A: [outputs/provenance/R2G_PSD_all_subjects_full.gexf](outputs/provenance/R2G_PSD_all_subjects_full.gexf)
* Figure 11B (top): [outputs/provenance/R2G_PSD_all_subjects_full.gexf](outputs/provenance/R2G_PSD_all_subjects_full.gexf)
* Figure 11B (bottom): [outputs/provenance/R2G_PSD_all_subjects.gexf](outputs/provenance/R2G_PSD_all_subjects.gexf)
* Figure 11C: [outputs/provenance/R2G_PSD_all_subjects.gexf](outputs/provenance/R2G_PSD_all_subjects.gexf)
* Figure 11D: [outputs/provenance/R2G_PSD_all_subjects.gexf](outputs/provenance/R2G_PSD_all_subjects.gexf)
* Figure 11E: [outputs/provenance/R2G_PSD_all_subjects.gexf](outputs/provenance/R2G_PSD_all_subjects.gexf)
* Figure 12A: [outputs/provenance/R2G_PSD_all_subjects_simplified_Q_shape_units_function.gexf](outputs/provenance/R2G_PSD_all_subjects_simplified_Q_shape_units_function.gexf)
* Figure 12B: [outputs/provenance/R2G_PSD_all_subjects_simplified_Q_units.gexf](outputs/provenance/R2G_PSD_all_subjects_simplified_Q_units.gexf)
* Figure 13B: [outputs/smoothed_plot/R2G_PSD_all_subjects_simplified.gexf](outputs/smoothed_plot/R2G_PSD_all_subjects_simplified.gexf)


### Logs

For each analysis script run, the respective folder in `/outputs` will 
contain text files to further document the execution:
* `environment.txt`: details on the Python and package version information;
* `psd_by_trial_type.out`: STDOUT and STDERR output of the script execution.


## Acknowledgments

[redacted for double-blind review]


## License

BSD 3-Clause License
