import argparse
from pathlib import Path

from alpaca import ProvenanceGraph


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
    parser.add_argument('source', metavar='source', nargs='?')
    parser.add_argument('output', metavar='output', nargs='?')
    args = parser.parse_args()

    main(args.source, args.output)
