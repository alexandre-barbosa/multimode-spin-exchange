# -*- coding: utf-8 -*-

""" Defines functions for manipulating and analysing simulation results. """

import csv
from dynamics import CollectiveSpinsPDE
from pde import FileStorage, ScalarField


def write_magnitudes(read_filename, write_filename="magnitudes.csv",
                     params=CollectiveSpinsPDE()):
    """
    Write the spatial average of the spin operator magnitudes to a .csv file.

    :read_filename: file containing operator magnitudes in each point and time
    :write_filename: magnitudes are saved with this parameter as a filename
    :params: instance of CollectiveSpinsPDE class with simulation parameters
    :return: none
    """
    reader = FileStorage(read_filename, write_mode="read_only")

    f = open(write_filename, "a")
    f.write("t, S, K \n")

    for time, collection in reader.items():
        S, K = collection.fields
        f.write(f"{time}, {S.magnitude*S.magnitude}, \
                {K.magnitude*K.magnitude} \n")

    f.close()


def write_parameters(params, write_filename="parameters.txt"):
    """
    Write the simulation parameters to a .txt file.

    :params: instance of CollectiveSpinsPDE class with simulation parameters
    :write_filename: simulation parameters are saved with this as a filename
    :return: none
    """
    attrs = vars(params)

    f = open(write_filename, "a")
    f.write(', '.join("%s: %s" % item for item in attrs.items()))
    f.close()


def efficiency(filename, params, write_filename='efficiency.csv'):
    """
    Calculate the efficiency of the spin exchange and write it to a file.

    :params: instance of CollectiveSpinsPDE class with simulation parameters
    :write_filename: simulation parameters are saved with this as a filename
    :return: calculated efficiency of the spin-exchange mechanism
    """
    efficiency = 0
    with open(filename, 'r') as csvfile:
        magnitudes = csv.reader(csvfile, delimiter=',')
        headers = next(magnitudes)  # ignore file header
        for row in magnitudes:
            try:
                if float(row[0]) >= params.tpulse+params.tdark+params.t0:
                    if float(row[1]) > efficiency:
                        efficiency = float(row[1])
            except ValueError:
                print("Invalid result for efficiency.")
                pass

    print(f"Exchange Efficiency: {efficiency}")

    f = open(write_filename, "a")
    f.write(f"{params.J/params.gamma_s}, {efficiency} \n")
    f.close()

    return efficiency
