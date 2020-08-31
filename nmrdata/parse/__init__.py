try:
    import rdkit
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        'To use parsing, you must install rdkit and extra dependencies with pip install nmrdata[parse]')
