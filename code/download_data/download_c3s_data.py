#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download c3s two-sat product vDT2018

Main link: 
https://cds.climate.copernicus.eu/cdsapp#!/dataset/satellite-sea-level-global?tab=overview

To download data I am following these steps: 
https://confluence.ecmwf.int/display/CKB/How+to+install+and+use+CDS+API+on+macOS

------

To be run in the virtual environment venv_c3s

info here: https://confluence.ecmwf.int/display/CKB/How+to+install+and+use+CDS+API+on+macOS

Check if you are in the correct environment 
writing these commands on the console: 

import sys
sys.executable
"""

import cdsapi
import numpy as np

c = cdsapi.Client()

# Set the path where you want to store the downloaded data
output_path   = '/Users/bbarcelo/HOME_SCIENCE/Scripts/2023_EKE_repositories/EKE_all_two_sat_adapted_for_paper/code/raw_c3s_data/vDT2018/'

# Set the years you want data from
all_years     = np.arange(1993,2021)

# Download one file per year
for syear in all_years: 
    
    selected_year = str(syear)
    
    print(' ')
    print('----------------------')
    print(selected_year)
    print('----------------------')
    print(' ')
    
    c.retrieve(
        'satellite-sea-level-global',
        {
            'variable': 'daily',
            'year': selected_year,
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'version': 'vDT2018',
            'format': 'tgz',
        },
        output_path + selected_year + '.tar.gz')




