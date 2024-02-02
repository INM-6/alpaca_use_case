import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from alpaca import ProvenanceGraph, AlpacaProvDocument


def add_suffix_to_gexf_file(source, suffix):
    return source.parent / (source.stem + f"_{suffix}.gexf")


def merge_graphs(merged_file, *source_files):
    doc = AlpacaProvDocument()
    for file in source_files:
        file_path = file.expanduser().absolute()
        doc.read_records(file_path, file_format="turtle")
    doc.serialize(merged_file)


def main(source, output):
    output_path = Path(output).expanduser().absolute()

    with TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir) / "temp.ttl"
        merge_graphs(temp_file, *source)
        prov_graph = ProvenanceGraph(temp_file,
                                     attributes=('dtype', 'shape', 'name',
                                                 'units', 'name', 'description',
                                                 'file_origin', 't_start',
                                                 't_stop', "_sampling_rate"),
                                     annotations=('subject_name', 'channel_names',
                                                  'performance_in_trial_str',
                                                  'trial_event_labels',
                                                  'belongs_to_trialtype'),
                                     remove_none=True,
                                     strip_namespace=True)

    # We are condensing all container memberships except for the Segment
    print("Condensing memberships")
    prov_graph.condense_memberships(preserve=['Segment'])

    # Save the GEXF graph that can be read by Gephi
    print("Saving GEXF")
    prov_graph.save_gexf(output_path)

    print("Simplifying graph")
    prov_graph.aggregate({'Quantity': ('units', )}, use_function_parameters=False,
                         output_file=output_path.parent / (output_path.stem + "_simplified.gexf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    source_files = [file
                    for file in Path(args.source_path).expanduser().absolute().iterdir()
                    if file.suffix == ".ttl"]

    main(source_files, args.output)
