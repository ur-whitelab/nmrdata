import click
from .protein_tfrecords import parse_refdb


@click.group()
def nmrparse():
    pass


nmrparse.add_command(parse_refdb)
