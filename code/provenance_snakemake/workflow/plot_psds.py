import argparse
from pathlib import Path
from datetime import datetime
import logging
import pickle
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from numpy import mean
from scipy.stats import sem

from alpaca import Provenance, activate, save_provenance, alpaca_setting
from alpaca.utils.files import get_file_name


# Apply the decorator to the functions used

mean = Provenance(inputs=['a'])(mean)

sem = Provenance(inputs=['a'])(sem)


# Setup logging
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s")


@Provenance(inputs=[], file_input=['file_name'])
def load_psd_data(file_name):
    """
    Reads data of the PSD stored in a file.
    """
    with open(file_name, "rb") as file:
        psd_data = pickle.load(file)

    return (psd_data['frequencies'], psd_data['psds'],
            psd_data['subject_name'])


def get_subject_psd_data(psd_data_files):
    subject_psd_data = defaultdict(dict)
    for file in psd_data_files:
        stem = str(file.stem)
        session, trial_type = stem.split("_")
        subject_psd_data[session][trial_type] = file
    return subject_psd_data


@Provenance(inputs=[])
def create_main_plot_objects(n_subjects, title):
    """
    Creates the plotting grid and figure/axes, with proper distribution and
    area sizes, and sets fixed attributes such as axes' labels.
    """

    # Set the style of plot components
    # Font sizes and style
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['axes.titlesize'] = 30
    plt.rcParams['axes.labelsize'] = 25
    plt.rcParams['figure.titlesize'] = 40
    plt.rcParams['figure.titleweight'] = 'bold'
    plt.rcParams['legend.fontsize'] = 20

    # Line widths
    plt.rcParams['axes.linewidth'] = 1
    plt.rcParams['xtick.major.size'] = 2
    plt.rcParams['ytick.major.size'] = 2

    fig, axes = plt.subplots(1, n_subjects, sharey=True, figsize=(18, 9),
                             constrained_layout=False)

    fig.suptitle(title)

    return fig, axes


@Provenance(inputs=['axes', 'freqs', 'psds', 'sem'])
def plot_lfp_psd(axes, freqs, psds, sem, label, sem_multiplier=1.96,
                 freq_range=None, **kwargs):
    """
    Plot the mean `psds` and the SEM `sem` in `axes`. The interval is
    defined as multiples of the SEM with the parameter `sem_multiplier`.
    The label for the legend is defined in `label`.
    If specified, the list in `freq_range`  will be used to select the
    appropriate range in the frequency axis (closed interval).
    """

    indexes = np.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))

    plot_freqs = freqs[indexes]
    plot_means = psds[indexes]
    plot_upper = plot_means.magnitude + (sem_multiplier * sem[indexes])
    plot_lower = plot_means.magnitude - (sem_multiplier * sem[indexes])
    axes.semilogy(plot_freqs, plot_means, label=label, **kwargs)
    axes.fill_between(plot_freqs, plot_lower, plot_upper,
                      color=kwargs['color'], alpha=0.10, lw=0)
    axes.set_ylabel(f"Power [{psds.dimensionality.latex}]")
    axes.set_xlabel(f"Frequency [{freqs.dimensionality}]")
    return axes


@Provenance(inputs=['axes', 'title'])
def set_title(axes, title):
    """
    Set the title of the plot in `axes`.
    """
    if isinstance(title, list) and len(title) == 1:
        title = title[0]
    axes.set_title(title)


@Provenance(inputs=['fig'], file_output=['file_name'])
def save_plot(fig, file_name, **kwargs):
    """
    Save the plot in `fig` to the file `file_name`.
    """
    fig.savefig(file_name, **kwargs)


def main(psd_data_files, output_file, trial_types):

    # Use builtin hash for matplotlib objects
    alpaca_setting('use_builtin_hash_for_module', ['matplotlib'])
    alpaca_setting('authority', "fz-juelich.de")

    # Activate provenance tracking
    activate()

    logging.info(f"Processing files: {','.join(map(str, psd_data_files))}")

    # Get number of subjects and sort the input files
    subject_psd_data = get_subject_psd_data(psd_data_files)
    n_subjects = len(subject_psd_data.keys())

    # Generate the plotting objects
    fig, axes = create_main_plot_objects(n_subjects,
                                         "PSD for each trial type")

    # Iterate over PSD data files of each subject
    for sub_idx, subject in enumerate(subject_psd_data.keys()):

        for trial_type, color in trial_types.items():
            trial_data_file = subject_psd_data[subject][trial_type]
            frequencies, all_psds_stacked, subject_name = \
                load_psd_data(trial_data_file)

            # Summarize PSDs
            psd_mean = mean(all_psds_stacked, axis=0)
            psd_sem = sem(all_psds_stacked, axis=0)

            # Plot
            plot_lfp_psd(axes[sub_idx], frequencies, psd_mean, psd_sem,
                         color=color, lw=1, label=trial_type,
                         freq_range=[0, 49])

        # Add legend and title
        axes[sub_idx].legend()
        set_title(axes[sub_idx], subject_name)

    # Save output file
    logging.info(f"Saving output to {output_file}")
    save_plot(fig, output_file, format='png', dpi=300)

    # Save provenance information as Turtle file
    prov_file_format = "ttl"
    prov_file = get_file_name(output_file, extension=prov_file_format)
    save_provenance(prov_file, file_format=prov_file_format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--trial_types', type=str, required=True)
    parser.add_argument('input', metavar='input', nargs="+")
    args = parser.parse_args()

    psd_data_files = [Path(arg).expanduser().absolute() for arg in args.input]

    # Colors for the plot passed as a list
    # trial_type1=color_code;trial_type2=color_code...
    trial_types = dict(item.split('=')
                       for item in args.trial_types.split(';'))

    output_file = Path(args.output_file).expanduser().absolute()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    start = datetime.now()
    logging.info(f"Start time: {start}")

    # Run the analysis
    main(psd_data_files, output_file, trial_types)

    end = datetime.now()
    logging.info(f"End time: {end}; Total processing time:{end - start}")
