import yaml
import pandas as pd
import networkx as nx

def get_annotation_by_folder(folder_name, annotation_csv):
    df = pd.read_csv(annotation_csv, delimiter='|')
    result = df[df['id'].str.contains(folder_name)]
    if not result.empty:
        return result.iloc[0]['annotation']
    else:
        return "Folder not found in CSV."


def load_config(path="./Configs/Base.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def create_graph(keypoints):
    # Create a graph: 17 for body, 21 for left hand and 21 for right hand.
    graph = nx.Graph()
    graph.add_nodes_from(keypoints)  # 59 nodes
    
    edges = [
        # TODO: Change face representation to add more details.
        (0, 1), (1, 2), (2, 3), (3, 7),  # right eye
        (0, 4), (4, 5), (5, 6), (6, 8),  # left eye
        (9, 10), # mouth
        (11, 12), (11, 13), (12, 14), (12, 16), (11, 15), (15, 16), (14, 17), (13, 38),  #body
        (17, 18), (18, 19), (19, 20), (20, 21),  # left_thumb
        (17, 22), (22, 23), (23, 24), (24, 25),  # left_index
        (22, 26), (26, 27), (27, 28), (28, 29),  # left_middle
        (26, 30), (30, 31), (31, 32), (32, 33),  # left_ring
        (17, 34), (30, 34), (34, 35), (35, 36), (36, 37),  # left_pinky
        (38, 39), (39, 40), (40, 41), (41, 42), # right_thumb
        (38, 43), (43, 44), (44, 45), (45, 46),  # right_index
        (43, 47), (47, 48), (48, 49), (49, 50),  # right_middle
        (47, 51), (51, 52), (52, 53), (53, 54),  # right_ring
        (38, 55), (51, 55), (55, 56), (56, 57), (57, 58)  # right_pinky
    ]
    
    graph.add_edges_from(edges)
    
    return graph