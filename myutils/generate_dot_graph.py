import os
import sys
import json
import logging
import argparse


DEFAULT_COLOUR_CODE = "c0c0c0"

RELATIONSHIP_COLOUR_CODES = {
    "plays": "dc143c",
    "interacts_with": "ffd700",
    "on": "6b8e23",
    "inside_of": "bc8f8f",
    "holds": "4682b4",
    "is": "ed4545",
    "at": "4287f5",
    "wears": "6a5acd",
    "hits": "cd853f",
    "under": "000000"
}


def generate_label_name(label_name, label_alias_dict):

    if label_name in label_alias_dict:
        label_name = label_alias_dict[label_name]

    return label_name


def generate_bounding_box_moniker(bounding_box_record):

    return str(bounding_box_record["x"]) + ":" + \
           str(bounding_box_record["y"]) + "(" + \
           str(bounding_box_record["h"]) + "x" + \
           str(bounding_box_record["w"]) + ")"


def add_bounding_box(bounding_boxes_dict, bounding_box_record):
    """
    Adds the given bounding box name to the dictionary if not already present.

    The value is a unique identifier.

    Args:
        bounding_boxes_dict(dict): Where the key is the object name and the value is the unique id
        bounding_box_record(str): Object name
    """

    bounding_box_name = generate_bounding_box_moniker(bounding_box_record)

    if bounding_box_name not in bounding_boxes_dict:
        bounding_boxes_dict[bounding_box_name] = len(bounding_boxes_dict)

    return


def add_label(label_dict, label_name, label_alias_dict):
    """
    Adds the given label to the label dictionary.

    If the label is not already present, the value is a unique identifier

    Args:
        label_dict(dict): Where the key is the object name and the value is the unique id
        label_name(str): Object name
        label_alias_dict(dict):
    """

    if label_name in label_alias_dict:
        label_name = label_alias_dict[label_name]

    if label_name not in label_dict:
        label_dict[label_name] = len(label_dict)

    return


def generate_graph_identifiers(visual_genome_image_record, label_alias_dict):
    """
    Creates a dictionary containing each label and bounding box in this visual genome dictionary.

    Args:
        visual_genome_image_record(dict):

    Returns:
        dict: Where the key is the object name and the value is the unique id
    """

    graph_identifier_dict = {
        "labels": {},
        "bounding_boxes": {}
    }

    for relationship_record in visual_genome_image_record["relationships"]:

        add_label(graph_identifier_dict["labels"], relationship_record["object"]["name"], label_alias_dict)
        add_bounding_box(graph_identifier_dict["bounding_boxes"], relationship_record["object"])

        add_label(graph_identifier_dict["labels"], relationship_record["subject"]["name"], label_alias_dict)
        add_bounding_box(graph_identifier_dict["bounding_boxes"], relationship_record["subject"])

    return graph_identifier_dict


def generate_relationship_dicts(visual_genome_record, graph_identifier_dict, label_alias_dict):

    bounding_boxes_dict = {}
    relationships_dict = {}

    for bounding_box_name in graph_identifier_dict["bounding_boxes"]:

        bounding_boxes_dict[bounding_box_name] = {
            "primary_attribute": None,
            "attributes": {}
        }

    for relationship_record in visual_genome_record["relationships"]:

        object_label_name = generate_label_name(relationship_record["object"]["name"], label_alias_dict)
        object_bounding_box_name = generate_bounding_box_moniker(relationship_record["object"])

        subject_label_name = generate_label_name(relationship_record["subject"]["name"], label_alias_dict)
        subject_bounding_box_name = generate_bounding_box_moniker(relationship_record["subject"])

        relationship_type = relationship_record["predicate"]

        bounding_boxes_dict[object_bounding_box_name]["primary_attribute"] = object_label_name

        if relationship_type == "is":

            try:
                bounding_boxes_dict[object_bounding_box_name]["attributes"][subject_label_name] += 1
            except KeyError:
                bounding_boxes_dict[object_bounding_box_name]["attributes"][subject_label_name] = 1

        else:

            bounding_boxes_dict[subject_bounding_box_name]["primary_attribute"] = subject_label_name

            try:
                bounding_box_relationships = relationships_dict[object_bounding_box_name]
            except KeyError:
                relationships_dict[object_bounding_box_name] = {}
                bounding_box_relationships = relationships_dict[object_bounding_box_name]

            try:
                bounding_box_relationships[subject_bounding_box_name].append(relationship_type)
            except KeyError:
                bounding_box_relationships[subject_bounding_box_name] = [relationship_type]

    return bounding_boxes_dict, relationships_dict


def group_by_label(visual_genome_record, label_alias_dict):

    graph_identifier_dict = generate_graph_identifiers(visual_genome_record, label_alias_dict)

    bounding_box_labels, relationship_dict =\
        generate_relationship_dicts(visual_genome_record, graph_identifier_dict, label_alias_dict)

    return graph_identifier_dict, bounding_box_labels, relationship_dict


def balance_attributes_dict(attributes_dict):

    for label_name, label_record in attributes_dict.items():

        sorted_sub_label_dict = sorted(label_record["labels"].items(), key=lambda x: len(x[1]), reverse=True)

        for sub_label_name, sub_label_bounding_box_dict in sorted_sub_label_dict:

            if len(sub_label_bounding_box_dict) == 0:
                continue

            label_record["attributes"][sub_label_name] = {
                "bounding_boxes": [],
                "labels": {},
                "attributes": {}
            }

            for bounding_box_name in sub_label_bounding_box_dict.keys()[:]:

                # Remove it from this label
                if bounding_box_name in label_record["bounding_boxes"]:
                    label_record["bounding_boxes"].remove(bounding_box_name)

                sub_attribute_record = label_record["attributes"][sub_label_name]

                referenced_in_other_tags = False

                for other_label_name, other_label_bounding_box_dict in label_record["labels"].items():

                    if sub_label_name == other_label_name:
                        continue

                    if bounding_box_name in other_label_bounding_box_dict:

                        referenced_in_other_tags = True

                        try:
                            sub_attribute_bounding_box_dict = sub_attribute_record["labels"][other_label_name]
                        except KeyError:
                            sub_attribute_record["labels"][other_label_name] = {}
                            sub_attribute_bounding_box_dict = sub_attribute_record["labels"][other_label_name]

                        try:
                            sub_attribute_bounding_box_dict[bounding_box_name] += 1
                        except KeyError:
                            sub_attribute_bounding_box_dict[bounding_box_name] = 1

                        del other_label_bounding_box_dict[bounding_box_name]

                #
                if not referenced_in_other_tags:
                    sub_attribute_record["bounding_boxes"].append(bounding_box_name)

                del sub_label_bounding_box_dict[bounding_box_name]

        del label_record["labels"]

        balance_attributes_dict(label_record["attributes"])

    return


def output_sub_graphs(label_sub_graphs, image_dict, depth):

    indent = "    " * (depth + 1)
    dot_graph_str = ""

    for label_name, label_record in label_sub_graphs.items():

        dot_graph_str += indent + "subgraph cluster_%s {\n" % image_dict["labels"][label_name]
        dot_graph_str += indent + "style = rounded;\n"
        dot_graph_str += indent + "label = \"%s\";\n" % label_name
        dot_graph_str += indent + "\n"

        for bounding_box_name in label_record["bounding_boxes"]:

            bounding_box_id = image_dict["bounding_boxes"][bounding_box_name]

            dot_graph_str += indent + "    box%d[fontname=\"Consolas\", label=<\n" % bounding_box_id
            dot_graph_str += indent + "    <table border=\"0\" cellborder=\"0\" cellspacing=\"0\">\n"
            dot_graph_str += indent + "    <tr>\n"
            dot_graph_str += indent + "    <td>%s</td>\n" % bounding_box_name.replace("(", "<br/>(")
            dot_graph_str += indent + "    </tr>\n"
            dot_graph_str += indent + "    </table>> shape=box];\n"
            dot_graph_str += indent + "    \n"

        dot_graph_str += output_sub_graphs(label_record["attributes"], image_dict, depth + 1)

        dot_graph_str += indent + "};\n\n"

    return dot_graph_str


def convert_record_to_grouped_dot_graph(visual_genome_record, label_alias_dict):
    """

    Args:
        visual_genome_record:
        label_alias_dict:

    Returns:
        str
    """
    dot_graph_str = "digraph image_%s {\n" % visual_genome_record["image_id"]
    dot_graph_str += "\n"

    graph_identifier_dict, bounding_box_labels, relationship_dict =\
        group_by_label(visual_genome_record, label_alias_dict)

    dot_graph_sub_graphs = {}

    for bounding_box_name, attributes_record in bounding_box_labels.items():

        try:
            primary_attribute_record = dot_graph_sub_graphs[attributes_record["primary_attribute"]]

        except KeyError:
            dot_graph_sub_graphs[attributes_record["primary_attribute"]] = {
                "bounding_boxes": [],
                "labels": {},
                "attributes": {}
            }
            primary_attribute_record = dot_graph_sub_graphs[attributes_record["primary_attribute"]]

        primary_attribute_record["bounding_boxes"].append(bounding_box_name)

        for label_name in attributes_record["attributes"]:

            try:
                primary_attribute_bounding_box_dict = primary_attribute_record["labels"][label_name]
            except KeyError:
                primary_attribute_record["labels"][label_name] = {}
                primary_attribute_bounding_box_dict = primary_attribute_record["labels"][label_name]

            try:
                primary_attribute_bounding_box_dict[bounding_box_name] += 1
            except KeyError:
                primary_attribute_bounding_box_dict[bounding_box_name] = 1

    balance_attributes_dict(dot_graph_sub_graphs)

    print json.dumps(dot_graph_sub_graphs, indent=4)

    dot_graph_str += output_sub_graphs(dot_graph_sub_graphs, graph_identifier_dict, 0)

    for bounding_box_name, relationship_dict in relationship_dict.items():

        object_id = "box" + str(graph_identifier_dict["bounding_boxes"][bounding_box_name])

        for subject_bounding_box_name, relationship_list in relationship_dict.items():

            subject_id = "box" + str(graph_identifier_dict["bounding_boxes"][subject_bounding_box_name])

            for relationship_type in relationship_list:

                try:
                    colour_code = RELATIONSHIP_COLOUR_CODES[relationship_type]
                except KeyError:
                    colour_code = DEFAULT_COLOUR_CODE

                dot_graph_str += "%s->%s [ label=\"%s\", color=\"#%s\" ]\n" % \
                                 (object_id, subject_id, relationship_type,
                                  colour_code)

    dot_graph_str += "}"

    return dot_graph_str


def convert_record_to_dot_graph(visual_genome_record, class_mid_dict):
    """

    Args:
        visual_genome_record:
        class_mid_dict:

    Returns:
        str
    """
    dot_graph_str = "digraph image_%s {\n" % visual_genome_record["image_id"]

    graph_identifier_dict, bounding_box_labels, relationship_dict =\
        group_by_label(visual_genome_record, class_mid_dict)

    for label_name, label_id in graph_identifier_dict["labels"].items():

        if label_name in class_mid_dict:
            label_name = class_mid_dict[label_name]

        dot_graph_str += "label%d[fontname=\"Consolas\", label=<\n" % label_id
        dot_graph_str += "<table border=\"0\" cellborder=\"0\" cellspacing=\"0\">\n"
        dot_graph_str += "<tr>\n"
        dot_graph_str += "<td>%s</td>\n" % label_name
        dot_graph_str += "</tr>\n"
        dot_graph_str += "</table>> shape=doublecircle];\n"

    for bounding_box_name, bounding_box_id in graph_identifier_dict["bounding_boxes"].items():

        dot_graph_str += "box%d[fontname=\"Consolas\", label=<\n" % bounding_box_id
        dot_graph_str += "<table border=\"0\" cellborder=\"0\" cellspacing=\"0\">\n"
        dot_graph_str += "<tr>\n"
        dot_graph_str += "<td>%s</td>\n" % bounding_box_name.replace("(", "<br/>(")
        dot_graph_str += "</tr>\n"
        dot_graph_str += "</table>> shape=box];\n"

    for bounding_box_name, label_dict in bounding_box_labels.items():

        colour_code = RELATIONSHIP_COLOUR_CODES["is"]

        object_id = "box" + str(graph_identifier_dict["bounding_boxes"][bounding_box_name])
        subject_id = "label" + str(graph_identifier_dict["labels"][label_dict["primary_attribute"]])

        dot_graph_str += "%s->%s [ label=\"%s\", color=\"#%s\" ]\n" % \
                         (object_id, subject_id, "is", colour_code)

        for label_name in label_dict["attributes"]:

            subject_id = "label" + str(graph_identifier_dict["labels"][label_name])

            dot_graph_str += "%s->%s [ label=\"%s\", color=\"#%s\" ]\n" % \
                             (object_id, subject_id, "is", colour_code)

    for bounding_box_name, relationship_dict in relationship_dict.items():

        object_id = "box" + str(graph_identifier_dict["bounding_boxes"][bounding_box_name])

        for subject_bounding_box_name, relationship_list in relationship_dict.items():

            subject_id = "box" + str(graph_identifier_dict["bounding_boxes"][subject_bounding_box_name])

            for relationship_type in relationship_list:

                try:
                    colour_code = RELATIONSHIP_COLOUR_CODES[relationship_type]
                except KeyError:
                    colour_code = "4287f5"

                dot_graph_str += "%s->%s [ label=\"%s\", color=\"#%s\" ]\n" % \
                                 (object_id, subject_id, relationship_type,
                                  colour_code)

    dot_graph_str += "}"

    return dot_graph_str


def main():

    class_mid_dict = {}

    # Initialise the logging level
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format="")
    logger = logging.getLogger(os.path.basename(__file__))

    # Initialise the argument parsing object
    parser = argparse.ArgumentParser(description='Converts an Annotation CSV file to a Visual Genome JSON file.')

    parser.add_argument('json_file_path', metavar='json_file_path', type=str,
                        help='path to the Visual Genome JSON file')

    parser.add_argument('dot_file_path', metavar='dot_file_path', type=str,
                        help='path to output the DOT graph file')

    parser.add_argument('-c', '--class-descriptor-path', dest='class_descriptor_path', type=str, default=None,
                        help='')

    parser.add_argument('-g', '--group-by-label', dest='do_group_label', action='store_const',
                        const=True, default=False,
                        help='simplify code after emulation')

    # Parse the script's arguments
    args = parser.parse_args()

    # Basic sanity tests
    if not os.path.exists(args.json_file_path):
        logger.error("Failed to find the JSON path: %s" % args.json_file_path)
        return

    if args.class_descriptor_path is not None:

        if not os.path.exists(args.class_descriptor_path):
            logger.error("Failed to find the class descriptor path: %s" % args.class_descriptor_path)
            return

        with open(args.class_descriptor_path) as fh:

            for line in fh.read().splitlines():
                key, value = line.split(",", 1)
                class_mid_dict[key] = value

    with open(args.json_file_path) as fh:
        visual_genome_record_list = json.load(fh)

    for visual_genome_record in visual_genome_record_list:

        with open(args.dot_file_path, "wb") as fh:

            if args.do_group_label:
                fh.write(convert_record_to_grouped_dot_graph(visual_genome_record, class_mid_dict))

            else:
                fh.write(convert_record_to_dot_graph(visual_genome_record, class_mid_dict))

        break

    return


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print "\n\nAborted."
