
import numpy as np
import os
import utils


def compute_eke(ds, out_file, out_folder='../processed_data/global_EKE/',  region=False):

    """
    Compute Eddy Kinetic Energy [cm2 s-2] from geostrophic velocity anomalies (ugosa, vgosa)
    
    Results saved in :out_folder: as netcdf files.
    
    """

    # From m to cm
    ds['ugosa'] = ds['ugosa']*1e2
    ds['vgosa'] = ds['vgosa']*1e2

    EKE = (1/2)*((ds['ugosa'])**2 + (ds['vgosa'])**2)  # cm2 s-2
    EKE = EKE.astype(np.float32)

    EKE = EKE.to_dataset(name="EKE")
    EKE = EKE.rename({'latitude': 'lat',
                        'longitude': 'lon'})
    EKE.attrs['title'] = "Eddy Kinetic Energy"
    EKE['EKE'].attrs['units'] = "cm^2 s^{-2}"
    EKE['EKE'].attrs['long_name'] = "Eddy Kinetic Energy from geostrophic velocity anomalies (ugosa, vgosa)"
    EKE = EKE.where(abs(EKE.lat) < 65, drop=True)

    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in EKE.data_vars}

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    EKE.to_netcdf(out_folder + out_file,
                encoding=encoding)

    return


def compute_EKE_tseries(ds, mask=None):

    area_array = utils.area(ds.lat, ds.lon)
    

    if mask is not None:
        ds = ds.where(mask==1)
        area_array = area_array.where(mask==1)
        area_sum = area_array.sum(['lat', 'lon'])

    eke_tseries = ((ds.EKE)*area_array)/area_sum
    eke_tseries = eke_tseries.sum(['lat', 'lon']) 
    eke_tseries = eke_tseries.to_dataset(name="EKE").compute()

    return eke_tseries




