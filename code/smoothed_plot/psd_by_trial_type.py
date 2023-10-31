import argparse
from pathlib import Path
from datetime import datetime
import logging

import quantities as pq
from quantities import Quantity

import neo
from neo.utils import get_events, cut_segment_by_epoch, add_epoch

from elephant.signal_processing import butter
from elephant.spectral import welch_psd

import numpy as np
import matplotlib.pyplot as plt

from numpy import mean, linspace
from scipy.interpolate import make_interp_spline, BSpline
from scipy.stats import sem

from alpaca import Provenance, activate, save_provenance, alpaca_setting
from alpaca.utils.files import get_file_name


# Apply the decorator to the funtions used

butter = Provenance(inputs=['signal'])(butter)

welch_psd = Provenance(inputs=['signal'])(welch_psd)

add_epoch = Provenance(inputs=['segment', 'event1', 'event2'])(add_epoch)

get_events = Provenance(inputs=['container'],
                        container_output=True)(get_events)

cut_segment_by_epoch = Provenance(inputs=['seg', 'epoch'],
                                  container_output=True)(cut_segment_by_epoch)

mean = Provenance(inputs=['a'])(mean)

sem = Provenance(inputs=['a'])(sem)

neo.AnalogSignal.downsample = Provenance(inputs=['self'])(neo.AnalogSignal.downsample)

linspace = Provenance(inputs=[])(linspace)

make_interp_spline = Provenance(inputs=['x', 'y'])(make_interp_spline)

BSpline.__call__ = Provenance(inputs=['self', 'x'])(BSpline.__call__)

Quantity.__new__ = Provenance(inputs=['data'])(Quantity.__new__)


@Provenance(inputs=[], file_input=['file_name'])
def load_data(file_name):
    """
    Reads all blocks in the NIX data file `file_name`.
    """
    path = Path(file_name).expanduser().absolute()
    session = neo.NixIO(str(path))
    block = session.read_block()
    return block


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


@Provenance(inputs=['signal'])
def select_channels(signal, skip_channels):
    """
    Selects all the channels from the AnalogSignal `signal` that are not
    in `skip_channels`. Channels are identified by the array annotations.
    """
    if skip_channels is None:
        return signal

    mask = ~np.isin(signal.array_annotations['channel_names'],
                    [f"chan{ch}" for ch in skip_channels])
    return signal[:, mask]


@Provenance(inputs=['axes', 'freqs', 'psds', 'error_low', 'error_high'])
def plot_lfp_psd(axes, freqs, psds, error_low, error_high, label,
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
    plot_upper = error_high[indexes]
    plot_lower = error_low[indexes]
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


@Provenance(inputs=['arrays'])
def vstack_quantities(*arrays):
    """
    Performs stacking of Quantity arrays, as ordinary `np.vstack` removes
    the unit information.
    """
    if len(arrays) == 1:
        # If a single element was passed
        if (isinstance(arrays[0], list) and
                isinstance(arrays[0][0], pq.Quantity)):
            # Expand if it is a list of Quantities
            *arrays, = arrays[0]
        elif isinstance(arrays[0], pq.Quantity):
            # Otherwise, if a single Quantity was passed, just return it
            return arrays[0]

    # Check if all units are the same, and all dimensions agree
    unit = arrays[0].units
    simplified_unit = unit.simplified
    dimensions = arrays[0].shape
    for array in arrays[1:]:
        if (array.units.simplified != simplified_unit or
                array.shape != dimensions):
            raise TypeError("Arrays do not agree in units or dimensionality")

    stacked = np.vstack(arrays) * unit
    return stacked


@Provenance(inputs=['psd_mean', 'psd_sem'])
def error_line(psd_mean, psd_sem, multiplier):
    return psd_mean.magnitude + (multiplier * psd_sem)


def main(session_files, output_dir, skip_channels):

    # Use builtin hash for matplotlib objects
    alpaca_setting('use_builtin_hash_for_module', ['matplotlib'])
    alpaca_setting('authority', "fz-juelich.de")

    # Activate provenance tracking
    activate()

    logging.info(f"Processing files: {','.join(session_files)}")

    # Labels of the different trial types in the data, and the colors
    # that will be used for plotting
    trial_types = {'PGLF': 'k', 'PGHF': 'b', 'SGLF': 'r', 'SGHF': 'g'}

    # List with trial types strings and number of subjects
    all_trial_types = list(trial_types.keys())
    n_subjects = len(session_files)

    # Generate the plotting objects
    fig, axes = create_main_plot_objects(n_subjects,
                                         "PSD for each trial type")

    # Iterate over each input file, taking the requested list of channels
    # to be ignored
    for sub_idx, (subject_file, skip) in enumerate(zip(session_files,
                                                       skip_channels)):

        logging.info(f"Processing {subject_file}; skipping channels {skip}")

        # Load the Neo Block with the data
        block = load_data(subject_file)

        # Iterate over each trial type
        for trial_idx, trial_type in enumerate(all_trial_types):

            # Select the trials for the analysis
            # Raw data is in the last AnalogSignal
            logging.info(f"Extracting starting events for {trial_type}")

            events = get_events(block.segments[0],
                                trial_event_labels='CUE-OFF',
                                performance_in_trial_str='correct_trial',
                                belongs_to_trialtype=trial_type)[0]

            logging.info(f"Extracting trial epochs for {trial_type}")

            trial_epochs = add_epoch(block.segments[0], events,
                                     pre=0 * pq.ms, post=500 * pq.ms,
                                     attach_result=False)

            logging.info(f"Cutting trials of type {trial_type}")

            trial_segments = cut_segment_by_epoch(block.segments[0],
                                                  trial_epochs)

            all_freqs = []
            all_psds = []

            logging.info("Computing PSD")

            # Iterate over each trial, and compute the PSDs
            for trial in trial_segments:

                signal = select_channels(trial.analogsignals[-1], skip)

                filtered_signal = butter(signal,
                                         lowpass_frequency=250 * pq.Hz)

                downsampled_signal = filtered_signal.downsample(60)

                freqs, psd = welch_psd(downsampled_signal,
                                       frequency_resolution=2 * pq.Hz)

                avg_psd = mean(psd, axis=0)

                all_freqs.append(freqs)
                all_psds.append(avg_psd)

                del filtered_signal

            # Combine the PSDs of each trial and summarize
            all_psds_stacked = vstack_quantities(*all_psds)
            psd_units = all_psds_stacked.units
            psd_mean = mean(all_psds_stacked, axis=0)
            psd_sem = sem(all_psds_stacked, axis=0)
            frequencies = all_freqs[0]

            level = 1.96
            error_low = error_line(psd_mean, psd_sem, -level)
            error_high = error_line(psd_mean, psd_sem, +level)

            mean_spline = make_interp_spline(frequencies, psd_mean)
            error_low_spline = make_interp_spline(frequencies,
                                                  error_low)
            error_high_spline = make_interp_spline(frequencies,
                                                   error_high)

            frequencies = linspace(0, 50, 500)
            frequencies = Quantity(frequencies, units='Hz')

            psd_mean = mean_spline(frequencies)
            psd_mean = Quantity(psd_mean, units=psd_units)

            error_low = error_low_spline(frequencies)
            error_high = error_high_spline(frequencies)

            # Plot
            plot_lfp_psd(axes[sub_idx], frequencies, psd_mean, error_low,
                         error_high, color=trial_types[trial_type], lw=1,
                         label=trial_type, freq_range=[0, 49])

        # Add legend and title
        axes[sub_idx].legend()
        set_title(axes[sub_idx], block.annotations['subject_name'])

    # Save output file
    out_file = output_dir / "R2G_PSD_all_subjects.png"
    logging.info(f"Saving output to {out_file}")
    save_plot(fig, out_file, format='png', dpi=300)

    # Save provenance information as Turtle file
    prov_file_format = "ttl"
    prov_file = get_file_name(out_file, extension=prov_file_format)
    save_provenance(prov_file, file_format=prov_file_format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--skip_channels', type=str, required=False)
    parser.add_argument('input', metavar='input', nargs=2)
    args = parser.parse_args()

    session_files = [str(Path(arg).expanduser().absolute())
                     for arg in args.input]

    output_dir = Path(args.output_path).expanduser().absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    # `skip_channels` argument contains the list of channels to be skipped
    # Format: `subject_1_list;subject_2_list`
    # For each subject, the list is separated by commas:
    # `channel#,channel#,...,channel#`, where `channel#` is a positive int
    # If all channels are used for a subject, use `0` instead of a list
    # Subject order is the same as the files in `input`
    skip = [None, None]
    if args.skip_channels:
        subject_1, subject_2 = args.skip_channels.split(';')
        skip = list()
        for skip_list in (subject_1, subject_2):
            skip_channels = skip_list.split(',')
            skip.append(list(map(int, skip_channels)))
            if not skip[-1][-1]:  # Last element zero? Use all channels
                skip[-1] = None

    start = datetime.now()
    print(f"Start time: {start}")

    # Run the analysis
    main(session_files, output_dir, skip)

    end = datetime.now()
    print(f"End time: {end}; Total processing time:{end - start}")
