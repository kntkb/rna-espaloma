#!/usr/bin/env python

import os, sys
import numpy as np
import h5py
import torch
import espaloma as esp
import qcportal as ptl
import click
from collections import Counter
from openff.toolkit.topology import Molecule
from openff.qcsubmit.results import BasicResultCollection
from simtk import unit
from simtk.unit import Quantity
from matplotlib import pyplot as plt



def get_graph(record, idx):
    smi = record["smiles"][0].decode('UTF-8')
    offmol = Molecule.from_smiles(smi, allow_undefined_stereo=True)
    offmol.compute_partial_charges_am1bcc()   # https://docs.openforcefield.org/projects/toolkit/en/0.9.2/api/generated/openff.toolkit.topology.Molecule.html
    charges = offmol.partial_charges.value_in_unit(esp.units.CHARGE_UNIT)
    g = esp.Graph(offmol)
    
    energy = record["dft_total_energy"][int(idx)]
    grad = record["dft_total_gradient"][int(idx)]
    xyz = record["conformations"][int(idx)]

    # energy is already hartree
    g.nodes["g"].data["u_ref"] = torch.tensor(
        [
            Quantity(
                energy,
                esp.units.HARTREE_PER_PARTICLE,
            ).value_in_unit(esp.units.ENERGY_UNIT)
        ],
        dtype=torch.get_default_dtype(),
    )[None, :]

    g.nodes["n1"].data["xyz"] = torch.tensor(
        np.stack(
            [
                Quantity(
                    xyz,
                    unit.bohr,
                ).value_in_unit(esp.units.DISTANCE_UNIT)
            ],
            axis=1,
        ),
        requires_grad=True,
        dtype=torch.get_default_dtype(),
    )

    g.nodes["n1"].data["u_ref_prime"] = torch.stack(
        [
            torch.tensor(
                Quantity(
                    grad,
                    esp.units.HARTREE_PER_PARTICLE / unit.bohr,
                ).value_in_unit(esp.units.FORCE_UNIT),
                dtype=torch.get_default_dtype(),
            )
        ],
        dim=1,
    )
    
    g.nodes['n1'].data['q_ref'] = c = torch.tensor(charges, dtype=torch.get_default_dtype(),).unsqueeze(-1)
    
    return g




def load_from_hdf5(kwargs):
    filename = kwargs["hdf5"]
    key = kwargs["keyname"]
    idx = kwargs["conf_id"]
    output_prefix = kwargs["output_prefix"]

    hdf = h5py.File(filename)
    record = hdf[key]

    gs = []
    g = get_graph(record, idx)
    gs.append(g)
            
    ds = esp.data.dataset.GraphDataset(gs)
    ds.save(output_prefix)





@click.command()
@click.option("--hdf5", required=True, help='hdf5 filename')
@click.option("--keyname", required=True, help='keyname of the hdf5 group')
@click.option("--conf_id", required=True, help='conformation index')
@click.option("--output_prefix", required=True, help='output directory to save graph data')
def cli(**kwargs):
    print(kwargs)
    load_from_hdf5(kwargs)



if __name__ == "__main__":
    cli()
