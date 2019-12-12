from __future__ import print_function, absolute_import
import os
import sys
import csv
import json
import argparse
import logging
import pandas as pd
import struct 
import numpy as np


"""
Trying to reformat the OpenImage dataset into the Relationship Genome dataset from: https://visualgenome.org/
based on: 
https://github.com/danfeiX/scene-graph-TF-release/tree/master/data_tools
"""

# CSV Column Names
COLUMN_NAME_IMAGE_ID = "ImageID"
COLUMN_NAME_LABEL_1 = "LabelName1"
COLUMN_NAME_LABEL_2 = "LabelName2"
COLUMN_NAME_X_MIN_1 = "XMin1"
COLUMN_NAME_X_MAX_1 = "XMax1"
COLUMN_NAME_Y_MIN_1 = "YMin1"
COLUMN_NAME_Y_MAX_1 = "YMax1"
COLUMN_NAME_X_MIN_2 = "XMin2"
COLUMN_NAME_X_MAX_2 = "XMax2"
COLUMN_NAME_Y_MIN_2 = "YMin2"
COLUMN_NAME_Y_MAX_2 = "YMax2"
COLUMN_NAME_RELATIONSHIP = "RelationshipLabel"


def read_csv_file(csv_file_path):
    """
    Reads the given CSV file and converts each row into a dictionary where
    the key is the column header.

    Args:
        csv_file_path(str): Path to the CSV file to read (must exist)

    Returns:
        list: List of record dictionaries
    """
    csv_records = []

    with open(csv_file_path) as csv_file:

        for row in csv.DictReader(csv_file):
            csv_records.append(row)

    return csv_records


def convert_to_visual_genome_dict(annotation_record_list):
    """
    Converts Annotation records into a Visual Genome structured dictionary.

    Args:
        annotation_record_list(list): List of Annotation records

    Example:
        "1fe6f315288fe145": {
            "image_id": "1fe6f315288fe145",
            "relationships": [
                {
                    "object": {
                        "h": 0.23666667000000008,
                        "name": "/m/01mzpv",
                        "w": 0.10875,
                        "x": "0",
                        "y": "0.7625"
                    },
                    "predicate": "is",
                    "subject": {
                        "h": 0.23666667000000008,
                        "name": "/m/05z87",
                        "w": 0.10875,
                        "x": "0",
                        "y": "0.7625"
                    }
                }
            ]
        },

    Returns:
        dict: Where the key is the ImageId and the value is the Relationship dictionary
    """
    visual_genome_dict = {}
    records = len(annotation_record_list)
    print("Generating new relationship records...")
    for idx, annotation_record in enumerate(annotation_record_list):
        print("Processing {}/{}".format(idx+1, records), end='\r')
        if annotation_record["RelationshipLabel"] == 'is':
            continue
        image_id = annotation_record[COLUMN_NAME_IMAGE_ID]
        
        try:
            image_dict = visual_genome_dict[image_id]

        except KeyError:

            # No Visual Genome dictionary for
            visual_genome_dict[image_id] = {
                "image_id": image_id,
                "relationships": [],
                "nodes": [],
                "object": [],
                "subject": [],
                "edge_list": [],
                "edge_labels": [],
            }

            image_dict = visual_genome_dict[image_id]
        
        # Generate unique identifiers for nodes
        obj = int(float(annotation_record[COLUMN_NAME_X_MIN_2])*100)+int(float(annotation_record[COLUMN_NAME_Y_MIN_2])*100)
        sub = int(float(annotation_record[COLUMN_NAME_X_MIN_1])*100)+int(float(annotation_record[COLUMN_NAME_Y_MIN_1])*100)

        # Update relationships
        image_dict["relationships"].append(
            {
                "predicate": annotation_record["RelationshipLabel"],
                "object": {
                    "x": float(annotation_record[COLUMN_NAME_X_MIN_1]),
                    "w": float(annotation_record[COLUMN_NAME_X_MAX_1]) - float(annotation_record[COLUMN_NAME_X_MIN_1]),
                    "y": float(annotation_record[COLUMN_NAME_Y_MIN_1]),
                    "h": float(annotation_record[COLUMN_NAME_Y_MAX_1]) - float(annotation_record[COLUMN_NAME_Y_MIN_1]),
                    "name": annotation_record[COLUMN_NAME_LABEL_1],
                    "node": obj
                    # "synsets": annotation_dict[UNKNOWN],
                },
                "subject": {
                    "x": float(annotation_record[COLUMN_NAME_X_MIN_2]),
                    "w": float(annotation_record[COLUMN_NAME_X_MAX_2]) - float(annotation_record[COLUMN_NAME_X_MIN_2]),
                    "y": float(annotation_record[COLUMN_NAME_Y_MIN_2]),
                    "h": float(annotation_record[COLUMN_NAME_Y_MAX_2]) - float(annotation_record[COLUMN_NAME_Y_MIN_2]),
                    "name": annotation_record[COLUMN_NAME_LABEL_2],
                    "node": sub
                    # "synsets": annotation_dict[UNKNOWN],
                }
            }
        )
        # Make list of distinct nodes
        image_dict['nodes'].append(obj)
        image_dict['nodes'].append(sub)
        image_dict['nodes'] = list(set(image_dict['nodes']))

        # Set up for edge list
        image_dict['object'].append(obj)
        image_dict['subject'].append(sub)
        image_dict['edge_list'].append((obj, sub))
        image_dict['edge_labels'].append( annotation_record["RelationshipLabel"])
    return visual_genome_dict


def process_image_data(csv_file, image_dir):
    print("Generating image meta_data...")
    images_had = [im.replace('.jpg', '') for im in os.listdir(image_dir)]
    df = pd.read_csv(csv_file)
    if len(images_had) < 1:
        import sys; sys.exit()
    df['coco_id'] = [None]*len(df)
    df['flickr_id'] = [None]*len(df)
    df = df.rename(columns={'ImageID': 'image_id', 'OriginalURL': 'url'})
    df = df.set_index('image_id')
    df = df.ix[images_had]
    height = list()
    width = list()
    for idx, image in enumerate(df.index.values[:]):
        print("Processing {}/{}".format(idx+1, len(df)), end='\r')
        try:
            filename = os.path.join(image_dir, image+'.jpg')
            if os.path.exists(filename):
                h, w = read_size(filename)
                height.append(h)
                width.append(w)
            else:
                df = df.drop(image, axis=0)
        except Exception as e:
            import pdb; pdb.set_trace()
    df['height'] = height
    df['width'] = width
    df = df[['width', 'height', 'url', 'coco_id', 'flickr_id']]
    image_data = df.to_dict(orient='index')
    # return list(image_data.values())
    return image_data

def read_size(image_path):
    with open(image_path, 'rb') as fhandle:
        try:
            # fhandle.seek(0) # Read 0xff next
            size = 2
            ftype = 0
            while not 0xc0 <= ftype <= 0xcf:
                fhandle.seek(size, 1)
                byte = fhandle.read(1)
                while ord(byte) == 0xff:
                    byte = fhandle.read(1)
                ftype = ord(byte)
                size = struct.unpack('>H', fhandle.read(2))[0] - 2
            # We are at a SOFn block
            fhandle.seek(1, 1)  # Skip `precision' byte.
            height, width = struct.unpack('>HH', fhandle.read(4))
            return height, width
        except Exception as e: #IGNORE:W0703
            print("Its windardium Leviosa not LEVIOSAR")
            print(e)
            return



def main():

    # Initialise the logging level
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format="")
    logger = logging.getLogger(os.path.basename(__file__))

    # Initialise the argument parsing object
    parser = argparse.ArgumentParser(description='Converts an Annotation CSV file to a Visual Genome JSON file.')

    parser.add_argument('csv_file_path', metavar='csv_file_path', type=str,
                        help='path to the Annotations CSV file to convert')

    parser.add_argument('json_file_path', metavar='json_file_path', type=str,
                        help='path to output the Visual Genome JSON file')

    # Parse the script's arguments
    args = parser.parse_args()

    # Basic sanity tests
    if not os.path.exists(args.csv_file_path):
        logger.error("Failed to find the CSV path: %s" % args.csv_file_path)
        return

    # First read in the Annotation records
    annotation_records = read_csv_file(args.csv_file_path)

    # Convert them to a Visual Genome dictionary
    visual_genome_dict = convert_to_visual_genome_dict(annotation_records)

    # Write out the results - Visual Genome uses a list of Image records hence .values()
    with open(args.json_file_path, "w") as fh:
        fh.write(json.dumps(visual_genome_dict.values(), indent=4, sort_keys=True))

    return


if __name__ == '__main__':
    # root = '/lustre/fs0/groups/course.cap6614/Data/OpenImages2019_ONLYVisualRelationship/'
    root = '/home/mschiappa/Desktop/VisualRelationshipsDetection/data/open-images'

    for select in ['train', 'validation']:
        root = '/home/mschiappa/Desktop/VisualRelationshipsDetection/data/open-images'
        # val_vrd = 'challenge-2019-validation-vrd.csv'
        # train_vrd = 'challenge-2019-train-vrd.csv'
        # train_metadata = 'train-images-boxable-with-rotation.csv'
        # val_metadata = 'validation-images-with-rotation.csv'
        root = os.path.join(root, select)
        # Validation dataset
        try:
            
            # First read in image metadata 
            # # image_dir = os.path.join(root, select)
            # csv_file_path = 'validation-images-with-rotation.csv' if select is 'validation' else 'train-images-boxable-with-rotation.csv'
            # csv_file = os.path.join(root, csv_file_path)
            
            # # Write out results with height and width now included
            # with open(os.path.join(root, select+'_image_data.json'), 'w') as f:
            #     # json.dump(process_image_data(csv_file, image_dir), f)
            #     json.dump(process_image_data(csv_file, os.path.join(root, 'images')), f)

            # First read in the Annotation records
            csv_file = os.path.join(root, 'challenge-2019-{}-vrd.csv'.format(select))
            annotation_records = read_csv_file(csv_file)
            df = pd.read_csv(csv_file)
            # Convert them to a Visual Genome dictionary
            visual_genome_dict = convert_to_visual_genome_dict(annotation_records)

            # Write out the results - Visual Genome uses a list of Image records hence .values()
            json_file_path = os.path.join(root, select+'_relationships.json')
            with open(json_file_path, "w") as fh:
                fh.write(json.dumps(visual_genome_dict, indent=4, sort_keys=True))
        except KeyboardInterrupt:
            print("\n\nAborted.")

    
    # try:
    
        # # First read in the Annotation records
        # csv_file = '/home/mschiappa/Desktop/VisualRelationshipsDetection/data/open-images/train/train-annotations-vrd.csv'
        # annotation_records = read_csv_file(csv_file)

        # # Convert them to a Visual Genome dictionary
        # visual_genome_dict = convert_to_visual_genome_dict(annotation_records)

        # # Write out the results - Visual Genome uses a list of Image records hence .values()
        # json_file_path = '/home/mschiappa/Desktop/VisualRelationshipsDetection/data/open-images/train/train_relationships.json'
        # with open(json_file_path, "w") as fh:
        #     fh.write(json.dumps(visual_genome_dict, indent=4, sort_keys=True))

        # First read in image metadata 
    #     image_dir = '/home/mschiappa/Desktop/VisualRelationshipsDetection/data/open-images/train/images'
    #     csv_file = '/home/mschiappa/Desktop/VisualRelationshipsDetection/data/open-images/train/train-images-boxable-with-rotation.csv'
        
    #     # Write out results with height and width now included
    #     with open('/home/mschiappa/Desktop/VisualRelationshipsDetection/data/open-images/train/train_image_data.json', 'w') as f:
    #         f.write(json.dump(process_image_data(csv_file, image_dir), f))
    # except KeyboardInterrupt:
        # print("\n\nAborted.")
    # except Exception as e:
    #     print(e)