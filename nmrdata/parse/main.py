import click
from .protein_tfrecords import parse_refdb
from .metabolite_tfrecords import parse_metabolites


@click.group()
def nmrparse():
    pass


nmrparse.add_command(parse_refdb)
nmrparse.add_command(parse_metabolites)
