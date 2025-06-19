import numpy as np
import xarray as xr
from scipy import ndimage
import os
import my_pymannkendall     as mk


def dirtodict(dirPath):
    d = {}
    for i in [os.path.join(dirPath, i) for i in os.listdir(dirPath)
              if os.path.isdir(os.path.join(dirPath, i))]:
        d[os.path.basename(i)] = dirtodict(i)
    d['.files'] = [os.path.join(dirPath, i) for i in os.listdir(dirPath)
                   if os.path.isfile(os.path.join(dirPath, i))]
    return d


region_dic = {'GWSE': [[-0.593297, 42.156397], [21.390730, 65.139224]],
              'AC': [[-49.705567, -3.992485], [-10.668587, 88.354845]],
              'KE': [[27.214451, 126.707357], [44.797421, 178.966069]],
              'GS': [[24.655584, -79.479982], [52.309994, -25.198502]],
              'LC': [[17.631824, -98.185393], [32.110687, -81.340371]],
              'BMC': [[-53.048704, -59.082424], [-32.907111, -27.689903]],
              'EAC': [[-43.201439, 143.651991], [-21.725291, 167.912370]]}


def smooth_regions(da, lim = 1000):
  
    np_da = da.values

    # Create a structure for the morphological operations
    struct = ndimage.generate_binary_structure(2, 2)

    # Fill the holes
    np_da_filled = ndimage.binary_fill_holes(np_da, structure=struct).astype(int)

    # Label each separate region
    label_np_da, num_features = ndimage.label(np_da_filled)

    # Create a size map
    sizes = ndimage.sum(np_da_filled, label_np_da, range(num_features + 1))

    # Remove small regions that have sizes less than a threshold 
    mask_sizes = sizes > lim
    np_da_clean = mask_sizes[label_np_da]

    smooth_np_da = ndimage.binary_closing(np_da_clean, structure=struct).astype(int)

    # Convert the numpy array back to DataArray
    da_filled_clean = xr.DataArray(smooth_np_da, coords=da.coords, dims=da.dims)


    return da_filled_clean

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the haversine distance between two sets of coordinates.
    
    More info: https://en.wikipedia.org/wiki/Haversine_formula
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    r = 6371 # Earth radius in km 
    
    distance = r * c

    return distance

def get_area_haversine(lat, lon):
    """
    Compute area of a rectilinear grid on a sphere using the haversine formula.
    """
    
    lat2d, lon2d = np.meshgrid(lat, lon, indexing='ij')

    lat1 = lat2d[:-1, :-1]    
    lat2 = lat2d[1:, :-1]
    
    lon1 = lon2d[:-1, :-1] 
    lon2 = lon2d[:-1, 1:]
 
    dlat = haversine(lon1, lat1, lon1, lat2)
    dlon = haversine(lon1, lat1, lon2, lat1)

    
    # Add NaN at the end of each array axis to have the same 
    # shape as lat and lon.
    new_value = np.nan
    
    # Add a NaN value at the end of the latitudinal axis
    dy = np.append(dlat, np.full((1, dlat.shape[1]), new_value), axis=0)
    dx = np.append(dlon, np.full((1, dlon.shape[1]), new_value), axis=0)
    
    # Add a NaN value at the end of the longitudinal axis
    dyy = np.append(dy, np.full((dy.shape[0], 1), new_value), axis=1)
    dxx = np.append(dx, np.full((dx.shape[0], 1), new_value), axis=1)

    return dxx * dyy

def get_mean_haversine(da, mask):
    
    area            = get_area_haversine(mask['lat'], mask['lon'])
    da_area         = xr.Dataset(coords={'lat': mask.lat, 'lon': mask.lon})
    da_area['area'] = (('lat', 'lon'), area)
    area            = da_area['area'].where(mask==1)
    area_sum        = area.sum(['lat', 'lon']).compute()
    area_rat        = area/area_sum
    ds_r            = da.where(mask==1)
    ds_r            = (ds_r * area_rat).sum(['lat', 'lon']).compute()
    val             = ds_r.values

    return val

def get_significant_trend_area_ratio_haversine(eke_trend, masks, region):

    sign_trend_reg      = eke_trend.sign.where(masks[region].notnull())
    sign_trend_reg_pixs = sign_trend_reg.where(sign_trend_reg==1)

    area            = get_area_haversine(masks['lat'], masks['lon'])
    da_area         = xr.Dataset(coords={'lat': masks.lat, 'lon': masks.lon})
    da_area['area'] = (('lat', 'lon'), area)
    area            = da_area['area'].where(masks[region]==1)
    area_sum        = area.sum(['lat', 'lon']).compute()

    sign_trend_reg_area = (sign_trend_reg_pixs * area).sum(['lat', 'lon']).compute()

    ratio = np.float32(sign_trend_reg_area/area_sum)

    return ratio


def crop(ds, coords, drop=False, 
         coord_names = ['lat', 'lon']):

    ds_crop = ds.where((ds[coord_names[0]] > coords[0][0]) &
                       (ds[coord_names[0]] < coords[1][0]) &
                       (ds[coord_names[1]] > coords[0][1]) &
                       (ds[coord_names[1]] < coords[1][1]), drop=drop)

    return ds_crop


def mk_test(ts):
    
    if np.isnan(ts).all():
        return np.nan, np.nan
    results = mk.yue_wang_modification_test(ts)
    slope, p_value, intercept, n_ns = results.slope, results.p, results.intercept, results.n_ns
    
    # Compute standard error of the slope
    SE_slope = standard_error_of_slope(ts, slope, intercept, n_ns)

    return slope, p_value, SE_slope

def standard_error_of_slope(ts, slope, intercept, n_ns):
    
    '''
    Compute standard error of the slope using the effective 
    sample size (ESS).

    The standard error of the slope is calculated as the 
    residual standard error divided by the square root of 
    the sum of squared differences in the independent 
    variable (James et al., 2023), considering the effective 
    sample size of the time series (Stan Development 
    Team, 2021; Martínez-Moreno et al., 2021).

    info about standard error of the slope (James et al., 2023): 
    https://www.statlearning.com/ (eq. 3.8)

    info about ESS here (Stan Development Team, 2021): 
    https://mc-stan.org/docs/2_21/reference-manual/effective-sample-size-section.html 

    '''

    ''' 
    Preprocessing the time series to skip nans
    (as in mk.yue_wang_modification_test) 
    '''
    ts_pp, c = mk.__preprocessing(ts)
    ts_p, n  = mk.__missing_values_analysis(ts_pp, method = 'skip')

    ''' 1) Calculate residual standard error '''

    # Create a time array (assuming time points are consecutive integers)
    time = np.arange(len(ts_p))

    # Calculate the predicted values using the Sen's slope
    y_pred = slope * time + intercept 

    # Calculate the residuals
    residuals = ts_p - y_pred

    # Calculate the Sum of Squared Residuals (RSS)
    RSS = np.sum(residuals**2)

    # Compute effective sample size (ESS)
    ESS  = len(ts_p)/n_ns

    print(' ')
    print('len(ts)', len(ts))
    print('len(ts_p)', len(ts_p))
    print('n_ns', n_ns)
    print('ESS', ESS)

    # Compute residual standard error

    RSE = np.sqrt(RSS/(ESS-2)) 

    ''' 2) Compute square root of the sum of squared 
    differences in the independent variable '''

    denom = np.sqrt(np.sum((time - np.mean(time))**2))

    ''' 3) Compute standard error '''
    
    SE_slope = RSE / denom

    return SE_slope

