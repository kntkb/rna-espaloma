#!/usr/bin/env python
# coding: utf-8

import os, sys, shutil
import pathlib
import glob as glob
import numpy as np
import re
import warnings
import mdtraj as md
import yaml
from openmm.app import *
from openmm import *
from openmm.unit import *
from openff.toolkit.utils import utils as offutils
from openff.units.openmm import to_openmm
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


def run(**kwargs):
    """
    """
    DEFAULT_OPTIONS = {
        'filepath': '.',
        'outpath': '.',
        'prmtopfile': 'input.prmtop',
        'inpcrdfile': 'input.inpcrd',
        'nsteps_min': 100,
        'nsteps_eq': 100,
        'nsteps_prod': 100,
        'checkpoint_interval': 100,
        'logging_frequency': 10,
        'netcdf_frequency': 10,
        'timestep': 2 * femtoseconds,
        'hmass': 1 * amu,
        'temperature': 300 * kelvin,
        'pressure': 1 * bar,
        'nonbonded_cutoff': 10 * angstrom,
    }


    # any better way to convert strings into openmm quantity?
    for k, v in kwargs.items():
        if k in [ "timestep", "temperature", "pressure", "nonbonded_cutoff" ]:
            converted = offutils.string_to_quantity(v)
            kwargs.update({ k: to_openmm(converted) })
        if k in [ "hmass" ]:
            m = v.split('*')[0].strip()
            kwargs.update({ k: float(m) * amu })
    options = DEFAULT_OPTIONS.copy()
    options.update(**kwargs)
    #print(options)


    """
    setup system
    """
    prmtop = AmberPrmtopFile(os.path.join(options["filepath"], options["prmtopfile"]))
    inpcrd = AmberInpcrdFile(os.path.join(options["filepath"], options["inpcrdfile"]))
    system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=options["nonbonded_cutoff"], constraints=HBonds, rigidWater=True, hydrogenMass=options["hmass"])

    integrator = LangevinMiddleIntegrator(options["temperature"], 1/picosecond, options["timestep"])
    simulation = Simulation(prmtop.topology, system, integrator)
    simulation.context.setPositions(inpcrd.positions)
    if inpcrd.boxVectors is not None:
        simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)


    ###
    ### minimize
    ###
    # http://getyank.org/latest/api/multistate_api/index.html?highlight=minimi#yank.multistate.multistatesampler.MultiStateSampler.minimize
    #simulation.minimizeEnergy(maxIterations=100, tolerance=1.0*kilojoules_per_mole/nanometers)

    restraint_atom_indices = [ a.index for a in prmtop.topology.atoms() if a.residue.name in ['A', 'C', 'U', 'T'] and a.element.symbol != 'H' ]
    restraint_index = system.addForce(create_position_restraint(inpcrd.positions, restraint_atom_indices))

    if options["nsteps_min"] > 0:
        simulation.minimizeEnergy(maxIterations=options["nsteps_min"])
        minpositions = simulation.context.getState(getPositions=True).getPositions()    
        PDBFile.writeFile(prmtop.topology, minpositions, open(os.path.join(options["outpath"], "min.pdb"), 'w'))   



    """
    reporter
    """
    #simulation.reporters.append(PDBReporter('/Users/takabak/Desktop/dump.pdb', options["netcdf_frequency"]))
    simulation.reporters.append(NetCDFReporter(os.path.join(options["outpath"], 'traj.nc'), options["netcdf_frequency"]))
    simulation.reporters.append(CheckpointReporter(os.path.join(options["outpath"], 'checkpoint.chk'), options["checkpoint_frequency"]))
    simulation.reporters.append(StateDataReporter(os.path.join(options["outpath"], 'reporter.log'), options["logging_frequency"], step=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True, temperature=True, volume=True, density=True, speed=True))


    """
    equilibrate
    """
    if options["nsteps_eq"] > 0:
        # 1) heating
        n = 50
        for i in range(n):
            temp = options["temperature"] * i / n
            simulation.context.setVelocitiesToTemperature(temp)     # initialize velocity
            integrator.setTemperature(temp)
            simulation.step(int(options["nsteps_eq"]/n))

        # 2) nvt
        integrator.setTemperature(options["temperature"])
        simulation.step(options["nsteps_eq"])

        # 3) npt
        system.removeForce(restraint_index)
        system.addForce(MonteCarloBarostat(options["pressure"], options["temperature"]))
        simulation.context.reinitialize(preserveState=True)
        simulation.step(options["nsteps_eq"])


    """
    production
    """
    if options["nsteps_prod"] > 0:
        integrator.setTemperature(options["temperature"])   # initialize velocity
        simulation.step(options["nsteps_prod"])



if __name__ == "__main__":
    with open("input.yml") as f:
        kwargs = yaml.safe_load(f)
        
    run(**kwargs)
