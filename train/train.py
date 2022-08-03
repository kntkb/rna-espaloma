#!/usr/bin/env python

import os, sys
import numpy as np
import torch
import espaloma as esp
import qcportal as ptl
from collections import Counter
from openff.toolkit.topology import Molecule
from openff.qcsubmit.results import BasicResultCollection
from dgllife.utils import EarlyStopping
from simtk import unit
from simtk.unit import Quantity
from matplotlib import pyplot as plt




def get_graph(mol, energy, grad):
    offmol = Molecule.from_qcschema(mol, allow_undefined_stereo=True)   # convert to OpenFF Molecule object
    g = esp.Graph(offmol)

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
                    mol.geometry,
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

    return g



def load_from_qcachive():
    client = ptl.FractalClient()
    collection = client.get_collection(collection_type, name)
    recs_wb97m = collection.get_records(method='wb97m-d3bj', basis='def2-tzvppd', program='psi4', keywords='wb97m-d3bj/def2-tzvppd')
    
    gs = []
    for i in range(len(recs_wb97m)):
        if recs_wb97m.iloc[i].record.status == 'COMPLETE':
            print("#{}: {}".format(i, recs_wb97m.iloc[i].name))
            
            mol = client.query_molecules(recs_wb97m.iloc[i].record.molecule)[0]
            energy = recs_wb97m.iloc[i].record.properties.scf_total_energy
            grad = recs_wb97m.iloc[i].record.return_result
            
            g = get_graph(mol, energy, grad)
            gs.append(g)
            
    ds = esp.data.dataset.GraphDataset(gs)
    ds.save("mydata")

    return ds



def load_data(source):
    collection_type = "Dataset"
    name = "RNA Single Point Dataset v1.0"

    if source == "download":
        ds = load_from_qcachive(collection_type, name)
    else:
        ds = esp.data.dataset.GraphDataset.load("mydata")

    return ds




def train(espaloma_model, epochs, loss_fn):
    #https://lifesci.dgl.ai/api/utils.pipeline.html
    stopper = EarlyStopping(mode='lower', patience=10, filename='checkpoint.pt')
    optimizer = torch.optim.Adam(espaloma_model.parameters(), 1e-4)

    for epoch in range(epochs):
        #for g in ds_tr:
        for g in ds_tr_loader:
            optimizer.zero_grad()
            if torch.cuda.is_available():
                #g.heterograph = g.heterograph.to("cuda:0")
                g.to("cuda:0")
            #g = espaloma_model(g.heterograph)
            g = espaloma_model(g)
            loss = loss_fn(g)
            loss.backward()
            optimizer.step()
            torch.save(espaloma_model.state_dict(), "%s.pt" % epoch)

        early_stop = stopper.step(loss.detach().data, espaloma_model)
        if early_stop:
            break

    torch.save(espaloma_model.state_dict(), "espaloma_model.pt")




def inspect_matrix(epochs, loss_fn, g_tr, g_vl, g_te):
    inspect_metric = loss_fn

    loss_tr = []
    loss_vl = []
    loss_te = []

    with torch.no_grad():
        for idx_epoch in range(epochs):
            espaloma_model.load_state_dict(
                torch.load("%s.pt" % idx_epoch)
            )

            if torch.cuda.is_available():
                g_tr = g_tr.to("cuda:0")
                g_vl = g_vl.to("cuda:0")
                g_te = g_te.to("cuda:0")

            espaloma_model(g_tr)
            loss_tr.append(inspect_metric(g_tr).item())

            espaloma_model(g_vl)
            loss_vl.append(inspect_metric(g_vl).item())

            espaloma_model(g_te)
            loss_te.append(inspect_metric(g_te).item())


    loss_tr = np.array(loss_tr) * 627.5
    loss_vl = np.array(loss_vl) * 627.5
    loss_te = np.array(loss_te) * 627.5
    
    return loss_tr, loss_vl, loss_te



def inspect_matrix_alter(epochs, loss_fn, g_tr, g_vl, g_te):
    #inspect_metric = esp.metrics.center(torch.nn.MSELoss(), reduction="mean") # use mean-squared error loss
    inspect_metric = loss_fn

    loss_tr = []
    loss_vl = []
    loss_te = []

    with torch.no_grad():
        for idx_epoch in range(epochs):
            espaloma_model.load_state_dict(
                torch.load("%s.pt" % idx_epoch)
            )

            # training set performance
            u = []
            u_ref = []
            for g in ds_tr:
                if torch.cuda.is_available():
                    g.heterograph = g.heterograph.to("cuda:0")
                espaloma_model(g.heterograph)
                u.append(g.nodes['g'].data['u'])
                #u_ref.append(g.nodes['g'])
                u_ref.append(g.nodes['g'].data['u_ref'])
            #u = torch.cat(u, dim=0)
            #u_ref = torch.cat(u_ref, dim=0)
            u = torch.cat(u, dim=1)
            u_ref = torch.cat(u_ref, dim=1)
            loss_tr.append(inspect_metric(u, u_ref))


            # validation set performance
            u = []
            u_ref = []
            for g in ds_vl:
                if torch.cuda.is_available():
                    g.heterograph = g.heterograph.to("cuda:0")
                espaloma_model(g.heterograph)
                u.append(g.nodes['g'].data['u'])
                #u_ref.append(g.nodes['g'])
                u_ref.append(g.nodes['g'].data['u_ref'])
            #u = torch.cat(u, dim=0)
            #u_ref = torch.cat(u_ref, dim=0)
            u = torch.cat(u, dim=1)
            u_ref = torch.cat(u_ref, dim=1)
            loss_vl.append(inspect_metric(u, u_ref))
            
            
            # test set performance
            u = []
            u_ref = []
            for g in ds_te:
                if torch.cuda.is_available():
                    g.heterograph = g.heterograph.to("cuda:0")
                espaloma_model(g.heterograph)
                u.append(g.nodes['g'].data['u'])
                #u_ref.append(g.nodes['g'])
                u_ref.append(g.nodes['g'].data['u_ref'])
            #u = torch.cat(u, dim=0)
            #u_ref = torch.cat(u_ref, dim=0)
            u = torch.cat(u, dim=1)
            u_ref = torch.cat(u_ref, dim=1)
            loss_te.append(inspect_metric(u, u_ref))


    # hartee to kcal/mol
    loss_tr = np.array(loss_tr) * 627.5
    loss_vl = np.array(loss_vl) * 627.5
    loss_te = np.array(loss_te) * 627.5

    return loss_tr, loss_vl, loss_te



def plot(loss_tr, loss_vl, loss_te, ofilename):
    plt.plot(loss_tr, label="train")
    plt.plot(loss_vl, label="valid")
    plt.plot(loss_te, label="test")
    plt.yscale("log")
    plt.legend()
    plt.savefig(ofilename)






if __name__ == "__main__":
    """
    define
    """
    source = "local"  # "local", "download"
    tr_ratio, vl_ratio, te_ratio = 1, 1, 8
    epochs = 10
    random_seed = 2666
    batch_size = 128



    """
    load data
    """
    ds = load_data(source)



    """
    define model
    """
    representation = esp.nn.Sequential(
        layer=esp.nn.layers.dgl_legacy.gn("SAGEConv"),   # use SAGEConv implementation in DGL
        config=[128, "relu", 128, "relu", 128, "relu"],  # 3 layers, 128 units, ReLU activation
    )


    readout = esp.nn.readout.janossy.JanossyPooling(
        in_features=128, config=[128, "relu", 128, "relu", 128, "relu"],
        out_features={                  # define modular MM parameters Espaloma will assign
            1: {"e": 1, "s": 1},        # atom hardness and electronegativity
            2: {"log_coefficients": 2}, # bond linear combination, enforce positive
            3: {"log_coefficients": 2}, # angle linear combination, enforce positive
            4: {"k": 6},                # torsion barrier heights (can be positive or negative)
        },
    )

    espaloma_model = torch.nn.Sequential(
        representation, readout, esp.nn.readout.janossy.ExpCoefficients(),
        esp.mm.geometry.GeometryInGraph(),
        esp.mm.energy.EnergyInGraph(),
        esp.nn.readout.charge_equilibrium.ChargeEquilibrium(),
    )



    """
    split dataset
    """
    ds.shuffle(seed=random_seed)
    ds_tr, ds_vl, ds_te = ds.split([tr_ratio, vl_ratio, te_ratio])
    print("train:    ", len(ds_tr))
    print("validate: ", len(ds_vl))
    print("test:     ", len(ds_te))

    ds_tr_loader = ds_tr.view(batch_size=batch_size, shuffle=True)
    g_tr = next(iter(ds_tr.view(batch_size=len(ds_tr))))
    g_vl = next(iter(ds_vl.view(batch_size=len(ds_vl))))
    g_te = next(iter(ds_te.view(batch_size=len(ds_te))))



    """
    train
    """
    loss_fn = esp.metrics.GraphMetric(
            base_metric=torch.nn.MSELoss(), # use mean-squared error loss
            between=["u", "u_ref"],         # between predicted and QM energies
            level="g",                      # compare on graph level
    )

    """
    loss_fn = [
        esp.metrics.GraphMetric(
            base_metric=torch.nn.MSELoss(), # use mean-squared error loss
            between=["u", "u_ref"],         # between predicted and QM energies
            level="g",                      # compare on graph level
        )
        esp.metrics.GraphMetric(
            base_metric=torch.nn.MSELoss(), # use mean-squared error loss
            between=["q", "q_hat"],         # between predicted and reference charges
            level="n1",                     # compare on node level 
        )
    ]
    """

    train(espaloma_model, epochs, loss_fn)

    loss_tr, loss_vl, loss_te = inspect_matrix(epochs, loss_fn, g_tr, g_vl, g_te)
    plot(loss_tr, loss_vl, loss_te, "plot_loss.png")

    loss_tr, loss_vl, loss_te = inspect_matrix_alter(epochs, loss_fn, g_tr, g_vl, g_te)
    plot(loss_tr, loss_vl, loss_te, "plot_loss_alter.png")
