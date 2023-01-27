import pickle 


def save_file(filename, file):
    """
    Save file in pickle format
    Args:
        file (any object): Can be any Python object. We would normally use this to save the
        processed Pytorch dataset
        filename (str): Name of the file
    """

    with open(filename, 'wb') as f:
        pickle.dump(file, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_file(filename):
    """
    Load a pickle file
    Args:
        filename (str): Name of the file
    Returns (Python obj): Returns the loaded pickle file
    """
    with open(filename, 'rb') as f:
        file = pickle.load(f)

    return file