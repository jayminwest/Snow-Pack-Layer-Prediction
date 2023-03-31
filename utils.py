"""
Jaymin West
Spring, 2023

This file contains utility functions for the Snowpack Analysis project.
"""
import xmltodict

def snowpilot_xml_to_dict(fname):
    """
    Parses the snowpilot xml file and returns a dictionary of the data.
    """
    with open(fname, 'r', encoding='utf-8') as file:
        sp_xml = file.read()

    sp_xml = xmltodict.parse(sp_xml)

    return sp_xml['Pit_Observation']
