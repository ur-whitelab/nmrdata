import click

try:
    import pdbfixer
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        'To use parsing, you must install openmm and extra dependencies with pip install nmrdata[parse]')


from .protein_tfrecords import parse_refdb
from .shiftml_tfrecords import parse_shiftml
from .metabolite_tfrecords import parse_metabolites


@click.group()
def nmrparse():
    pass


@click.command()
@click.argument('input_pdb')
@click.argument('output_pdb')
def clean_pdb(input_pdb, output_pdb):
    from pdbfixer import PDBFixer
    from simtk.openmm.app import PDBFile
    fixer = PDBFixer(filename=input_pdb)
    # we want to add missing atoms,
    # but not replace missing residue. We'd
    # rather just ignore those
    fixer.findMissingResidues()
    # remove the missing residues
    fixer.missingResidues = []
    # remove water!
    fixer.removeHeterogens(False)
    fixer.findMissingAtoms()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    PDBFile.writeFile(fixer.topology, fixer.positions, open(output_pdb, 'w'))


nmrparse.add_command(parse_refdb)
nmrparse.add_command(parse_metabolites)
nmrparse.add_command(parse_shiftml)
nmrparse.add_command(clean_pdb)
