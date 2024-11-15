import yaml
import pandas as pd

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

if __name__ == "__main__":
    print([(None, None)] * 21)