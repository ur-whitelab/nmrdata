import click
from .protein_tfrecords import parse_refdb
from .shiftml_tfrecords import parse_shiftml
from .metabolite_tfrecords import parse_metabolites
from .parse_universe import parse_universe


@click.group()
def nmrparse():
    pass


nmrparse.add_command(parse_refdb)
nmrparse.add_command(parse_metabolites)
nmrparse.add_command(parse_shiftml)
