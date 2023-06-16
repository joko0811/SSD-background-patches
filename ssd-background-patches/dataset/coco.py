def load_class_names(path):
    with open(path, "r") as fp:
        names = fp.read().splitlines()
    return names
