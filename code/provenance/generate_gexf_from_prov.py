import argparse
from pathlib import Path

from alpaca import ProvenanceGraph


def add_suffix_to_gexf_file(source, suffix):
    return source.parent / (source.stem + f"_{suffix}.gexf")


def main(source, output):
    source_path = Path(source).expanduser().absolute()
    output_path = Path(output).expanduser().absolute()

    prov_graph = ProvenanceGraph(source_path,
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

    # Save full graph before transformations
    print("Saving full graph")
    full_graph_path = add_suffix_to_gexf_file(output_path, "full")
    prov_graph.save_gexf(full_graph_path)

    # We are condensing all container memberships except for the Segment
    print("Condensing memberships")
    prov_graph.condense_memberships(preserve=['Segment'])

    # Save the condensed graph
    print("Saving GEXF")
    prov_graph.save_gexf(output_path)

    # Generate and save simplifications
    print("Simplifying graph")
    prov_graph.aggregate({}, use_function_parameters=False,
                         output_file=add_suffix_to_gexf_file(output_path, "simplified"))

    prov_graph.aggregate({'Quantity': ('shape', 'units', )}, use_function_parameters=True,
                         output_file=add_suffix_to_gexf_file(output_path, "simplified_Q_shape_units_function"))

    prov_graph.aggregate({'Quantity': ('shape', 'units', )}, use_function_parameters=False,
                         output_file=add_suffix_to_gexf_file(output_path, "simplified_Q_shape_units.gexf"))

    prov_graph.aggregate({'Quantity': ('units', ), 'AnalogSignal': ('shape',)}, use_function_parameters=True,
                         output_file=add_suffix_to_gexf_file(output_path, "simplified_Q_units_AS_shape_function"))

    prov_graph.aggregate({'Quantity': ('units', )}, use_function_parameters=False,
                         output_file=add_suffix_to_gexf_file(output_path, "simplified_Q_units_function"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('source', metavar='source', nargs='?')
    parser.add_argument('output', metavar='output', nargs='?')
    args = parser.parse_args()

    main(args.source, args.output)
