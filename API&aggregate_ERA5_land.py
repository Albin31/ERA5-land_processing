import cdsapi 
import xarray as xr
from calendar import monthrange
from datetime import datetime
import glob
import pandas as pd 


def api_era5_land(target_path,
                  start_year=2009,
                  start_month=1,
                  end_year=2019,
                  end_month=12,
                  variables=["2m_temperature",], 
                  area=[90, -180, -90, 180], # [north, west, south, east]
                  ):
    """
    Download ERA5-Land data from the Copernicus Climate Data Store (CDS) for a specified period and area.
    Parameters:
        target_path (str): Path to save the downloaded data.
        start_year (int): Start year for the data download (included).
        start_month (int): Start month for the data download (included).
        end_year (int): End year for the data download (included).
        end_month (int): End month for the data download (included).
        variables (list): List of variables to download. Default is ["2m_temperature"].
        area (list): Area to download data for, specified as [north, west, south, east].    
    """

    # Generate list of (year, month) tuples from start to end
    start_date = datetime(start_year, start_month, 1)
    end_date = datetime(end_year, end_month, 1)

    date_list = []
    current = start_date
    while current <= end_date:
        date_list.append((current.year, current.month))
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)

    # Set up CDS client and output file
    client = cdsapi.Client()

    for year, month in date_list:
        days_in_month = monthrange(year, month)[1]
        days_list = [f"{day:02d}" for day in range(1, days_in_month + 1)]
        month_str = f"{month:02d}"
        year_str = str(year)

        request = {
            "variable": variables,
            "year": year_str,
            "month": month_str,
            "day": days_list,
            "time": [f"{h:02}:00" for h in range(24)],
            "data_format": "grib",
            "download_format": "unarchived",
            "area": area,
        }

        target = f"{target_path}/{year}_months{month_str}_europe.grib"
        client.retrieve("reanalysis-era5-land", request, target)
        print(f"Data downloaded to {target}")


def aggregate_era5_land_timeseries(era5_land_data_path_inputs,
                      era5_land_data_path_outputs,
                      start_year=2009,
                      end_year=2019,
                      save=True):
    """
    Process ERA5-Land data from GRIB files into a single xarray Dataset.
    Parameters:
        era5_land_data_path_inputs (str): Path to the input GRIB files.
        era5_land_data_path_outputs (str): Path to save the processed NetCDF file.
        start_year (int): Start year for processing (included).
        end_year (int): End year for processing (included).
        save (bool): Whether to save the processed dataset to a NetCDF file.
    Returns:
        ds_variable (xarray.Dataset): Processed dataset containing the variables from the GRIB files.
    """

    ds_variable = xr.Dataset()
    for year in range(start_year, end_year+1):
        grib_pattern = f"{era5_land_data_path_inputs}/{year}_months*_europe.grib"
        grib_files = sorted(glob.glob(grib_pattern))

        if not grib_files:
            print(f"No file for year {year} found in {era5_land_data_path_inputs}")
            continue
        
        # Open the GRIB files 
        print(f"Opening GRIB files for year {year}...")
        ds = xr.open_mfdataset(grib_files, engine='cfgrib', combine='by_coords', chunks={'time': 100})
        
        print(f"Processing year {year} ...")
        # Stack the variables time and step into a single dimension valid_time
        ds_stacked = ds.stack(datetime=("time", "step"))
        ds_stacked = ds_stacked.assign_coords(valid_time=("datetime", ds["valid_time"].values.ravel()))
        ds_ready = ds_stacked.swap_dims({"datetime": "valid_time"})
        ds_ready = ds_ready.drop_vars("datetime")
        
        print(f"Filtering data for year {year} ...")
        ds_ready = ds_ready.where(ds_ready.valid_time.dt.year == year, drop=True)        
        ds_ready = ds_ready.sortby("valid_time")
        ds_ready = ds_ready.dropna(dim="valid_time", how="all") #to try
        
        print("Concatenating datasets...")
        if ds_variable:
            ds_variable = xr.concat([ds_variable, ds_ready], dim="valid_time")
        else:
            ds_variable = ds_ready
    if save:
        ds_variable.to_netcdf(f"{era5_land_data_path_outputs}/inflows_var_era5_{start_year}to{end_year}.nc", format='NETCDF4', engine='netcdf4')
    return ds_variable


