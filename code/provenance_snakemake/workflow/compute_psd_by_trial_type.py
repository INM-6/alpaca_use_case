import argparse
from pathlib import Path
from datetime import datetime
import logging
import pickle

import quantities as pq

import neo
from neo.utils import get_events, cut_segment_by_epoch, add_epoch

from elephant.signal_processing import butter
from elephant.spectral import welch_psd

import numpy as np
from numpy import mean

from alpaca import Provenance, activate, save_provenance, alpaca_setting


# Apply the decorator to the functions used

butter = Provenance(inputs=['signal'])(butter)

welch_psd = Provenance(inputs=['signal'])(welch_psd)

add_epoch = Provenance(inputs=['segment', 'event1', 'event2'])(add_epoch)

get_events = Provenance(inputs=['container'],
                        container_output=True)(get_events)

cut_segment_by_epoch = Provenance(inputs=['seg', 'epoch'],
                                  container_output=True)(cut_segment_by_epoch)

neo.AnalogSignal.downsample = Provenance(inputs=['self'])(neo.AnalogSignal.downsample)

mean = Provenance(inputs=['a'])(mean)


# Setup logging
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s")


@Provenance(inputs=[], file_input=['file_name'])
def load_data(file_name):
    """
    Reads all blocks in the NIX data file `file_name`.
    """
    path = Path(file_name).expanduser().absolute()
    session = neo.NixIO(str(path))
    block = session.read_block()
    return block


@Provenance(inputs=['frequencies', 'psds', 'subject_name'],
            file_output=['file_name'])
def save_psd_data(file_name, frequencies, psds, subject_name=None):
    data = {'frequencies': frequencies,
            'psds': psds,
            'subject_name': subject_name}

    with open(file_name, 'wb') as out:
        pickle.dump(data, out, protocol=pickle.HIGHEST_PROTOCOL)


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


def main(subject_file, output_dir, skip_channels, trial_types):

    alpaca_setting('authority', "fz-juelich.de")

    # Activate provenance tracking
    activate()

    # Process session file
    logging.info(f"Processing {subject_file}; skipping channels {skip}")

    session_code = str(subject_file.stem)

    # Load the Neo Block with the data
    block = load_data(subject_file)

    # Iterate over each trial type
    for trial_idx, trial_type in enumerate(trial_types):

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

            signal = select_channels(trial.analogsignals[-1], skip_channels)

            filtered_signal = butter(signal,
                                     lowpass_frequency=250 * pq.Hz)

            downsampled_signal = filtered_signal.downsample(60)

            freqs, psd = welch_psd(downsampled_signal,
                                   frequency_resolution=2 * pq.Hz)

            avg_psd = mean(psd, axis=0)

            all_freqs.append(freqs)
            all_psds.append(avg_psd)

            del filtered_signal

        # Combine the PSDs of each trial
        all_psds_stacked = vstack_quantities(*all_psds)

        # Save data to pickle file
        output_file = (output_dir / f"{session_code}_{trial_type}.pickle")

        logging.info(f"Saving output to {output_file}")

        frequencies = all_freqs[0]
        save_psd_data(output_file, frequencies, all_psds_stacked,
                      block.annotations['subject_name'])

    # Save provenance information as Turtle file
    prov_file_format = "ttl"
    prov_file = output_dir / f"{session_code}.{prov_file_format}"
    save_provenance(prov_file, file_format=prov_file_format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--skip_channels', type=str, required=False)
    parser.add_argument('--trial_types', type=str, required=True)
    parser.add_argument('input', metavar='input', type=str)
    args = parser.parse_args()

    session_file = Path(args.input)

    output_dir = Path(args.output_path).expanduser().absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    # `skip_channels` argument contains the channels to be skipped
    # A list is passed, with the channel numbers separated by commas:
    # `channel#,channel#,...,channel#`, where `channel#` is a positive int
    # If all channels are used for the subject, use `0` instead of a list
    skip = None
    if args.skip_channels:
        skip_channels = args.skip_channels.split(',')
        skip = list(map(int, skip_channels))
        if not skip[-1]:  # Last element zero? Use all channels
            skip = None

    trial_types = args.trial_types.split(',')

    start = datetime.now()
    logging.info(f"Start time: {start}")

    # Run the analysis
    main(session_file, output_dir, skip, trial_types)

    end = datetime.now()
    logging.info(f"End time: {end}; Total processing time:{end - start}")
