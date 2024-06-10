import subprocess as sp
import datetime as datetime
import os
from ftplib import FTP
import xarray as xr


def download_raw_data_ftp(
    username,
    password,
    year=1993,
    month=1,
    product="all_sat",
):
    

    if product == 'all_sat':
        #print('Retrieving Altimetric L4 data for {}-{}-{}...'.format(year, month, day))
        service_id = 'SEALEVEL_GLO_PHY_L4_MY_008_047'
        product_id = 'cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D'
        lon = 'longitude'
        lat = 'latitude'


    elif product == 'two_sat':
        #print('Retrieving Altimetric L4 data for {}-{}-{}...'.format(year, month, day))
        service_id = 'SEALEVEL_GLO_PHY_CLIMATE_L4_MY_008_057'
        product_id = 'c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D'
        lon = 'longitude'
        lat = 'latitude'

    # Navigate to the directory of interest
    ftp = FTP('my.cmems-du.eu', username, password)

    fold = f'Core/{service_id}/{product_id}/{year}/{month:02d}/' 
    ftp.cwd(fold)

    files = ftp.nlst()

    for i, file in enumerate(files):
        out_fold = f'../../temp/{product}/'
        path = out_fold+file
        if not os.path.exists(out_fold):
            os.makedirs(out_fold)
        if not os.path.isfile(path):
            with open(path, 'wb') as f:
                ftp.retrbinary("RETR " + file, f.write)

        
    ds = xr.open_mfdataset(out_fold + '*.nc', concat_dim='time', combine='nested')


    # select only 'ugosa' and 'vgosa' variables
    ds = ds[['ugosa', 'vgosa']]

    new_out_fold = f'../../raw_data/{product}/'
    if not os.path.exists(new_out_fold):
        os.makedirs(new_out_fold)
    
    ds.to_netcdf(new_out_fold+f'{product}_{year}_{month:02d}.nc')

    # close dataset
    ds.close()

    for file in os.listdir(out_fold):
        os.remove(os.path.join(out_fold, file))

    return 