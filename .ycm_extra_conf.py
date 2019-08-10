import os

def Settings( **kwargs ):
    not_found = 'not_found'
    conda_prefix = os.getenv('CONDA_PREFIX', not_found)
    conda_bin_path = "/usr/bin/python3"
    if conda_prefix != not_found:
        conda_bin_path = "{}/bin/python".format(conda_prefix)
    return {
        'interpreter_path': conda_bin_path
    }
