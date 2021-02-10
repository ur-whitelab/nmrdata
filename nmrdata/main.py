
import click
from .validation import *


@click.group()
def nmrdata():
    pass


nmrdata.add_command(count_records)
nmrdata.add_command(validate_peaks)
nmrdata.add_command(validate_embeddings)
nmrdata.add_command(validate_nlist)
nmrdata.add_command(count_names)
nmrdata.add_command(write_peak_labels)
nmrdata.add_command(find_pairs)
