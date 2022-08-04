#!/usr/bin/env python
# coding: utf-8

import os, sys, shutil
import pathlib
import glob as glob
import numpy as np
import re
import warnings
import mdtraj as md
import click
import openmmtools as mmtools
from openmm.app import *
from openmm import *
from openmm.unit import *
#from simtk.unit import Quantity
from openff.toolkit.utils import utils as offutils
#from openff.units.openmm import to_openmm
from sys import stdout
from openmm.app import PDBFile
from pdbfixer import PDBFixer
from mdtraj.reporters import NetCDFReporter



def create_position_restraint(position, restraint_atom_indices):
    """
    heavy atom restraint
    """
    force = CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k", 10.0*kilocalories_per_mole/angstroms**2)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    for i in restraint_atom_indices:
        atom_crd = position[i]
        force.addParticle(i, atom_crd.value_in_unit(nanometers))
    return force



def run(**options):
    print(options)
    prmtopfile = options["prmtop"]
    inpcrdfile = options["inpcrd"]
    output_prefix = options["output_prefix"]
    timestep = 2 * femtoseconds
    hmass = 1 * amu
    temperature = 300 * kelvin
    pressure = 1 * atmosphere
    nonbonded_cutoff = 9 * angstrom
    nsteps_min = 100
    nsteps_eq = 100
    nsteps_prod = 5000
    checkpoint_frequency = 10
    logging_frequency = 10
    netcdf_frequency = 10


    platform = mmtools.utils.get_fastest_platform()
    platform_name = platform.getName()
    print("fastest platform is ", platform_name)
    if platform_name == "CUDA":
        # Set CUDA DeterministicForces (necessary for MBAR)
        platform.setPropertyDefaultValue('DeterministicForces', 'true')
        platform.setPropertyDefaultValue('Precision', 'mixed')
    else:
        #raise Exception("fastest platform is not CUDA")
        warnings.warn("fastest platform is not CUDA")

    
    prmtop = AmberPrmtopFile(prmtopfile)
    inpcrd = AmberInpcrdFile(inpcrdfile)

    system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=nonbonded_cutoff, constraints=HBonds, rigidWater=True, hydrogenMass=hmass)
    integrator = LangevinMiddleIntegrator(temperature, 1/picosecond, timestep)
    simulation = Simulation(prmtop.topology, system, integrator, platform)
    
    simulation.context.setPositions(inpcrd.positions)
    if inpcrd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)


    # Define reporter
    #simulation.reporters.append(PDBReporter('/Users/takabak/Desktop/dump.pdb', options["netcdf_frequency"]))
    simulation.reporters.append(NetCDFReporter(os.path.join(output_prefix, 'traj.nc'), netcdf_frequency))
    simulation.reporters.append(CheckpointReporter(os.path.join(output_prefix, 'checkpoint.chk'), checkpoint_frequency))
    simulation.reporters.append(StateDataReporter(os.path.join(output_prefix, 'reporter.log'), logging_frequency, step=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, volume=True, density=True, speed=True))


    # Minimization
    restraint_atom_indices = [ a.index for a in prmtop.topology.atoms() if a.residue.name in ['A', 'C', 'U', 'T'] and a.element.symbol != 'H' ]
    restraint_index = system.addForce(create_position_restraint(inpcrd.positions, restraint_atom_indices))

    simulation.minimizeEnergy(maxIterations=nsteps_min)
    minpositions = simulation.context.getState(getPositions=True).getPositions()    
    PDBFile.writeFile(prmtop.topology, minpositions, open(os.path.join(output_prefix, 'min.pdb'), 'w'))   


    # Equilibration
    # Heating
    n = 50
    for i in range(n):
        temp = temperature * i / n
        simulation.context.setVelocitiesToTemperature(temp)     # initialize velocity
        integrator.setTemperature(temp)    # set target temperature
        simulation.step(int(nsteps_eq/n))

    # NVT
    integrator.setTemperature(temperature)
    simulation.step(nsteps_eq)

    # NPT
    system.removeForce(restraint_index)
    system.addForce(MonteCarloBarostat(pressure, temperature))
    simulation.context.reinitialize(preserveState=True)
    simulation.step(nsteps_prod)


    """
    Save state as XML
    """
    state = simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True, getForces=True)

    # Save and serialize the final state
    with open(os.path.join(output_prefix, "state.xml"), "w") as wf:
        xml = XmlSerializer.serialize(state)
        wf.write(xml)

    # Save and serialize integrator
    with open(os.path.join(output_prefix, "integrator.xml"), "w") as wf:
        xml = XmlSerializer.serialize(integrator)
        wf.write(xml)

    # Save the final state as a PDB
    with open(os.path.join(output_prefix, "state.pdb"), "w") as wf:
        PDBFile.writeFile(
            simulation.topology,
            simulation.context.getState(
                getPositions=True,
                enforcePeriodicBox=True).getPositions(),
                file=wf,
                keepIds=True
        )

    # Save and serialize system
    system.setDefaultPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    with open(os.path.join(output_prefix, "system.xml"), "w") as wf:
        xml = XmlSerializer.serialize(system)
        wf.write(xml)


@click.command()
@click.option('--prmtop', default="input.prmtop", help='path to prmtopfile')
@click.option('--inpcrd', default="input.inpcrd", help='path to inpcrdfile')
@click.option('--output_prefix', default=".", help='path to output files')
def cli(**kwargs):
    run(**kwargs)


if __name__ == "__main__":
    cli()