



import pandas as pd
import xarray as xr
import numpy as np
import dask as da
import tqdm as tqdm
import os
from pathlib import Path
import sys
import datetime

import matplotlib.pyplot as plt

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import matplotlib.image as imag


import aws_processing.loading
import aws_processing.remapping
import cartopy.crs as ccrs

def print_colocations(df,scream=True):
    sample_count=0

    #build a list of collocations
    collocations_list = []

    #df = df_all.copy()
    #df['time'] = pd.to_datetime(df['time'])
    #df = df[df['time'].dt.date == datetime.date(2025, 7, 5)].reset_index(drop=True)
    #print(f"Rows on July 5th: {len(df)}")

    #iterate through the dataframe and print each collocation until
    for index, row in df.iterrows():
        #print only if the next time is more than 1 hour apart + converts time to pandas datetime
        actual_time= pd.to_datetime(row['time'])
        if actual_time.date() != datetime.date(2025, 10, 31):
            continue
        next_time= pd.to_datetime(df['time'][index+1]) if index < len(df)-1 else None
        #print(row['time'])
        if next_time is None:
            continue
        # compute time_delta as difference between two timestamps
        time_delta = next_time - actual_time
        # print only if time delta is more than 1 hour   
        if time_delta > pd.Timedelta(hours=0.1) and index < len(df)-1:
            if scream == True:
                print(f"Collocation {sample_count+1}:")
                print(f"  Time: {row['time']}, Sat_Lat_1: {row['sat_1_lat']}, Sat_Lon_1: {row['sat_1_lon']}, Sat_Lat_2: {row['sat_2_lat']}, Sat_Lon_2: {row['sat_2_lon']}, Distance_km: {row['distance_km']}, delay: {row['sat_2_time_delay_mins']}")
                print("") 
            sample_count += 1
            collocations_list.append(row)
    
    print(f"Total collocations printed: {sample_count}") 
    return pd.DataFrame(collocations_list) 


import pandas as pd

def print_colocations_for_cloud_signal(df_all, scream=True):
    df = df_all.copy()
    df['time'] = pd.to_datetime(df['time'])
    print(df.columns.tolist())
    # --- filters ---
    target = pd.Timestamp('2025-10-31').date()
    mask = (
        (df['time'].dt.date == target)
    )
    #mask = (
    #    df['time'].dt.day.isin([5, 15, 25])          # 5th, 15th or 25th
    #    & (df['time'] >= pd.Timestamp('2025-07-01'))  # July 2025 onwards
    #    & (df['distance_km'] < 15)        # zero delay
    #)
    df = df[mask].reset_index(drop=True)

    sample_count = 0
    collocations_list = []

    for index, row in df.iterrows():
        actual_time = row['time']
        next_time = df['time'][index + 1] if index < len(df) - 1 else None
        if next_time is None:
            continue

        time_delta = next_time - actual_time

        if time_delta > pd.Timedelta(hours=0.1):
            if scream:
                print(f"Collocation {sample_count + 1}:")
                print(f"  Time: {row['time']}, Sat_Lat_1: {row['sat_1_lat']}, "
                      f"Sat_Lon_1: {row['sat_1_lon']}, Sat_Lat_2: {row['sat_2_lat']}, "
                      f"Sat_Lon_2: {row['sat_2_lon']}, Distance_km: {row['distance_km']}, "
                      f"delay: {row['sat_2_time_delay_mins']}")
                print("")
            sample_count += 1
            collocations_list.append(row)

    print(f"Total collocations printed: {sample_count}")
    return pd.DataFrame(collocations_list)


def earthcare_load_spec(time__ea,aws_lat,product_type='/L2b/ACM_CAP_2B/'):
    """Load nearest EarthCare file based on time_ea within ±1 hour window.
    
    Args:
        time__ea: colocation timestamp of EarthCare (str or datetime)
        aws_lat: latitude of the Arctic Weather Satellite swath ID 68
    
    Returns:
        xr.DataTree with EarthCare data
    """
    # Convert time to pandas Timestamp and add delay
    time_ea = pd.to_datetime(time__ea)
    
    # Define search window: -1 hour to +0 (time_ea)
    search_start = time_ea - pd.Timedelta(hours=0.5)
    search_end = time_ea + pd.Timedelta(hours=0.5)


    ea_time_min= time_ea - pd.Timedelta(minutes=5)
    ea_time_max= time_ea + pd.Timedelta(minutes=5)
    
    # Set base path and end letter based on location
    #base_path = Path('/data/s13/EarthCare/L2b/ACM_CAP_2B/')
    if time_ea>=pd.Timestamp("2025-12-01"):
            base_path = Path('/data/s13/EarthCare'+product_type+'BC/')
    if time_ea<pd.Timestamp("2025-12-01"):
            base_path = Path('/data/s13/EarthCare'+product_type+'BA/')    
    end_let = 'G' if aws_lat < 0 else 'C'
    
    # Search across potentially multiple days (if search spans midnight)
    search_dates = pd.date_range(start=search_start.date(), end=search_end.date(), freq='D')
    
    all_h5_files = []
    
    for search_date in search_dates:
        year_str = search_date.strftime('%Y')
        month_str = search_date.strftime('%m')
        day_str = search_date.strftime('%d')
        
        search_dir = base_path / year_str / month_str / day_str
        
        if search_dir.exists():
            # Find all h5 files ending with end_let
            h5_files = list(search_dir.glob(f'*{end_let}/*.h5'))
            all_h5_files.extend(h5_files)
    
    if not all_h5_files:
        print(f"No EarthCare files found between {search_start} and {search_end}")
        return None
    
    # Extract timestamps from folder names and find closest to time_ea
    closest_file = None
    min_time_diff = None
    
    for h5_file in all_h5_files:
        folder_name = h5_file.parent.name
        try:
            # Parse start timestamp from folder name (format: ECA_EXBA_ACM_CAP_2B_20250707T102155Z_...)
            start_time_str = folder_name.split('_')[5]  # Extract timestamp
            start_time = pd.to_datetime(start_time_str, format='%Y%m%dT%H%M%SZ')
            
            # Check if file's start time falls within search window
            if search_start <= start_time <= search_end:
                time_diff = abs((time_ea - start_time).total_seconds())
                
                if min_time_diff is None or time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_file = h5_file
        except Exception as e:
            print(f"Error parsing {folder_name}: {e}")
            continue
    
    if closest_file is None:
        print(f"No suitable EarthCare file found within window {search_start} to {search_end}")
        return None
    
    print(f"Loading: {closest_file}")
    print(f"Time difference: {min_time_diff} seconds")
    ea_data = xr.open_datatree(str(closest_file), engine='h5netcdf')

    ea_data_time_select=np.where((ea_data.ScienceData.time>= ea_time_min) & (ea_data.ScienceData.time<=ea_time_max))[0]
    print(ea_data_time_select.size)

    #node = ea_data['ScienceData'].data_vars
    #node

    #print(node)

    ea_data=ea_data.sel(along_track=ea_data_time_select)


    
    return ea_data


def handy_colocation_list_selector(number_of_collocation,coloc_list):
    #grep the desired colocation from the list
    region=None
    if number_of_collocation<=len(coloc_list):
        selected_coloc=coloc_list.iloc[number_of_collocation]
        print(f"Selected Collocation {number_of_collocation}:")
        print(f"  Time: {selected_coloc['time']}, Sat_Lat_1: {selected_coloc['sat_1_lat']}, Sat_Lon_: {selected_coloc['sat_1_lon']}, Sat_Lat_2: {selected_coloc['sat_2_lat']}, Sat_Lon_2: {selected_coloc['sat_2_lon']}, Distance_km: {selected_coloc['distance_km']}, delay: {selected_coloc['sat_2_time_delay_mins']}")
        if selected_coloc['sat_1_lat']>60:
            print("This colocation is in the Arctic region.")
            region='Arctic'
        elif selected_coloc['sat_1_lat']<-60:
            print("This colocation is in the Antarctic region.")
            region='Antarctic'
        else:
            print("This colocation is not in the polar regions.")
            region='Non-polar'

        print()
        
    else:
        print(f"Error: number_of_collocation {number_of_collocation} exceeds available collocations {len(coloc_list)}")
        
    time= selected_coloc['time']
    delay= selected_coloc['sat_2_time_delay_mins']
    return region,time,delay



def convert_timestamp_format(timestamp_str, margin_minutes=5.5):
    """Convert '2025-07-01 00:49:20' to ISO format with time slice ±N minutes.
    
    Returns a slice object suitable for aws_processing.loading.get_files_l1b(timerange=...)
    """
    # Parse the input string to a pandas Timestamp
    ts = pd.to_datetime(timestamp_str)
    
    # Compute bounds
    lower = ts - pd.Timedelta(minutes=margin_minutes)
    upper = ts + pd.Timedelta(minutes=margin_minutes)
    
    # Format as ISO strings (with T separator)
    lower_str = lower.strftime('%Y-%m-%dT%H:%M:%S')
    upper_str = upper.strftime('%Y-%m-%dT%H:%M:%S')
    
    # Return as slice object
    return slice(lower_str, upper_str)


def AWS_time_select(time__):


    timerange=convert_timestamp_format(time__)
    files = aws_processing.loading.get_files_l1b(timerange=timerange)
    ds = aws_processing.loading.load_multiple_files_l1b(files)
    ds = ds.sel(time=timerange)

    ds___ = aws_processing.remapping.remap_interp(
            ds,
            remap_to_ch="AWS33",
            method="nearest",
            fill_distance=True)
    
    ds___
    
    return ds___





def plot_colocation_figures(aws_datatree,earthcare_datatree,time,delay,
                            Antarctica_or_Arctic,sterni,aws_channel_group=3,central_point=68):
    

#plot colocation lat-lon maps for either Antarctica or Arctic

    time_ts = pd.to_datetime(time)
    delay__=delay

    lower_bound = np.datetime64(time_ts - pd.Timedelta(minutes=5.5))
    upper_bound = np.datetime64(time_ts + pd.Timedelta(minutes=5.5))

    lower_bound_earthcare = np.datetime64(time_ts - pd.Timedelta(minutes=5.5)+ pd.Timedelta(minutes=delay__))
    upper_bound_earthcare = np.datetime64(time_ts + pd.Timedelta(minutes=5.5)+ pd.Timedelta(minutes=delay__))


    
        

    earth_times = earthcare_datatree.ScienceData.time.values

    print(earth_times.size)

    # use elementwise & (or np.logical_and) instead of Python 'and'
    
    if isinstance(aws_datatree,xr.DataTree):
        aws_times = aws_datatree.data.navigation.time_startscan_utc_earthview.values
        
    if isinstance(aws_datatree,xr.Dataset):
        aws_times=aws_datatree.time.values


    aws_idx = np.where((aws_times >= lower_bound) & (aws_times <= upper_bound))[0]
    earth_idx = np.where((earth_times >= lower_bound_earthcare) & (earth_times <= upper_bound_earthcare))[0]
    


        
    if isinstance(aws_datatree,xr.DataTree):
        aws_latitudes = aws_datatree.data.navigation.aws_lat[aws_idx,:,aws_channel_group].values
        aws_longitudes = aws_datatree.data.navigation.aws_lon[aws_idx,:,aws_channel_group].values

    if isinstance(aws_datatree,xr.Dataset):
        aws_latitudes = aws_datatree.aws_lat[aws_idx,:].values
        aws_longitudes = aws_datatree.aws_lon[aws_idx,:].values






    plt.figure(4,(10,8))
    if Antarctica_or_Arctic=='Arctic':
        ax=plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=150))
        ax.set_extent([np.min(earthcare_datatree.ScienceData.longitude[earth_idx])-5,
                        np.max(earthcare_datatree.ScienceData.longitude[earth_idx])+5,
                          np.min(earthcare_datatree.ScienceData.latitude[earth_idx])-2,
                            90], ccrs.PlateCarree())
    else:
        ax=plt.axes(projection=ccrs.SouthPolarStereo(central_longitude=150))
        ax.set_extent([np.min(earthcare_datatree.ScienceData.longitude[earth_idx])-5,
                        np.max(earthcare_datatree.ScienceData.longitude[earth_idx])+5,
                        -90,
                          np.max(earthcare_datatree.ScienceData.latitude[earth_idx])+2], ccrs.PlateCarree())  
    ax.plot(aws_longitudes[:,central_point],
            aws_latitudes[:,central_point],'d',color=sterni[0],markersize=2,linestyle='None',label='AWS trajectory '+str(central_point), transform=ccrs.PlateCarree())
    ax.plot(aws_longitudes[:,central_point-40],
            aws_latitudes[:,central_point-40],'d',color=sterni[1],markersize=2,linestyle='None',label='AWS trajectory '+str(central_point)+'-40', transform=ccrs.PlateCarree())
    ax.plot(aws_longitudes[:,central_point+40],
            aws_latitudes[:,central_point+40],'d',color=sterni[2],markersize=2,linestyle='None',
            label='AWS trajectory '+str(central_point)+'+40', transform=ccrs.PlateCarree())
  
    ax.plot(earthcare_datatree.ScienceData.longitude[earth_idx],earthcare_datatree.ScienceData.latitude[earth_idx],'d',color=sterni[3],markersize=2,linestyle='None',
            label='EARTHCARE trajectory', transform=ccrs.PlateCarree())
    ax.coastlines(color='black',linewidth=2)
    ax.gridlines()
    plt.grid(True)
    plt.title('Satellite orbits on '+str(aws_times[int(np.round(np.median(aws_idx)))]), fontsize=16)
    plt.xlabel('longitude',fontsize=12)
    plt.ylabel('latitude',fontsize=12)
    plt.legend(fontsize=14)


    #plot time series of time vs latitude
    plt.figure(5,(10,6))
    plt.plot(aws_times[aws_idx],
             aws_latitudes[:,central_point],'x',color=sterni[0],label='AWS trajectory '+str(central_point))
    plt.plot(earthcare_datatree.ScienceData.time[earth_idx],
             earthcare_datatree.ScienceData.latitude[earth_idx],'x',color=sterni[3],label='EarthCare trajectory')               
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.title('Time series of Latitude vs Time', fontsize=16)
    plt.xlabel('Time',fontsize=14)
    plt.ylabel('Latitude',fontsize=14)
    #plot time series of time vs longitude
    plt.figure(6,(10,6))
    plt.plot(aws_times[aws_idx],
             aws_longitudes[:,central_point],'x',color=sterni[0],label='AWS trajectory '+str(central_point))
    plt.plot(aws_times[aws_idx],
             aws_longitudes[:,central_point-50],'x',color=sterni[1],label='AWS trajectory '+str(central_point)+'-50')
    plt.plot(aws_times[aws_idx],
             aws_longitudes[:,central_point+50],'x',color=sterni[2],label='AWS trajectory '+str(central_point)+'+50')
    plt.plot(earthcare_datatree.ScienceData.time[earth_idx],
             earthcare_datatree.ScienceData.longitude[earth_idx],'x',color=sterni[3],label='EarthCare trajectory')
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.title('Time series of Longitude vs Time', fontsize=16)
    plt.xlabel('Time',fontsize=14)
    plt.ylabel('Longitude',fontsize=14)






def haversine_km_array(lat1, lon1, lat2, lon2, R=6371.0088):
    """
    Vectorized/higher-robust haversine:
    - broadcasts inputs
    - normalizes lon-difference to [-180, 180] (per-pair)
    - returns np.nan where inputs are invalid
    """
    # convert and broadcast
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)
    lat1, lon1, lat2, lon2 = np.broadcast_arrays(lat1, lon1, lat2, lon2)

    # per-pair normalized longitude difference in degrees
    #dlon_deg = lon2 - lon1
    #this does not work for some reason: 
    dlon_deg = ((lon2 - lon1 + 180.0) % 360.0) - 180.0
    dlon = np.deg2rad(dlon_deg)

    # latitudes in radians
    lat1r = np.deg2rad(lat1)
    lat2r = np.deg2rad(lat2)
    dlat = lat2r - lat1r

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2

    # protect invalid values (e.g., NaNs) and clamp numeric issues
    with np.errstate(invalid='ignore'):
        d = 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

    d[~np.isfinite(d)] = np.nan
    return d

def calculate_distances_for_one_Earthcare_sample_faster(aws_lat_array,aws_lon_array,earthcare_lat,earthcare_lon):
    #calculate distances between one earthcare sample and all aws samples
    
    distances_array=haversine_km_array(aws_lat_array,aws_lon_array,earthcare_lat,earthcare_lon)
    return distances_array


def select_nearest_data_samples_faster_old(aws_datatree,earthcare_datatree,time,delay,aws_channel_group=3,colocate_to_sat_1_sat_2='EarthCare'):
     #ensure 'time' is a timestamp
    time_ts = pd.to_datetime(time)
    delay__=delay

    lower_bound = np.datetime64(time_ts - pd.Timedelta(minutes=5.5))
    upper_bound = np.datetime64(time_ts + pd.Timedelta(minutes=5.5))

    lower_bound_earthcare = np.datetime64(time_ts - pd.Timedelta(minutes=5.5)+ pd.Timedelta(minutes=delay__))
    upper_bound_earthcare = np.datetime64(time_ts + pd.Timedelta(minutes=5.5)+ pd.Timedelta(minutes=delay__))

    if isinstance(aws_datatree,xr.DataTree):
        aws_times = aws_datatree.data.navigation.time_startscan_utc_earthview.values

    if isinstance(aws_datatree,xr.Dataset):
        aws_times=aws_datatree.time.values
    earth_times = earthcare_datatree.ScienceData.time.values

#    print(earth_times.size)

    # use elementwise & (or np.logical_and) instead of Python 'and'
    aws_idx = np.where((aws_times >= lower_bound) & (aws_times <= upper_bound))[0]
    earth_idx = np.where((earth_times >= lower_bound_earthcare) & (earth_times <= upper_bound_earthcare))[0]

    ## calculate distances for each earthcare time sample to all aws swath samples and find the minimum distance index
    earth_latitudes = earthcare_datatree.ScienceData.latitude[earth_idx].values
    earth_longitudes = earthcare_datatree.ScienceData.longitude[earth_idx].values 
    if isinstance(aws_datatree,xr.DataTree):  
        aws_latitudes = aws_datatree.data.navigation.aws_lat[aws_idx,:,aws_channel_group].values
        aws_longitudes = aws_datatree.data.navigation.aws_lon[aws_idx,:,aws_channel_group].values
    if isinstance(aws_datatree,xr.Dataset):
        aws_latitudes = aws_datatree.aws_lat[aws_idx,:].values
        aws_longitudes = aws_datatree.aws_lon[aws_idx,:].values

    if colocate_to_sat_1_sat_2=='EarthCare':

        distances = np.empty((len(earth_latitudes), len(aws_latitudes)))
        aws_selected_swath_indices= np.zeros((aws_latitudes.size,aws_latitudes[:,1].size))

        sel_aws_time_array_along_aws_traj=np.zeros((earth_latitudes.size,1))
        sel_aws_swath_array_along_aws_traj=np.zeros((earth_latitudes.size,1))

    if colocate_to_sat_1_sat_2=='AWS':
        distances = np.empty((len(aws_latitudes), len(earth_latitudes)))

        aws_selected_swath_indices= np.zeros((aws_latitudes[:,1].size,earth_latitudes.size))

        sel_aws_time_array_along_aws_traj=np.zeros((aws_latitudes[:,1].size,1))
        sel_aws_swath_array_along_aws_traj=np.zeros((aws_latitudes[:,1].size,aws_latitudes[1,:].size))


    #take care that the temporal resolution of aws is unequal to earthcare,
    #so for each earthcare point in time find the nearest aws point that could be along aws_time axis or swath axis
    # 
    # return for an array of aws_time indices and aws_swath indices corresponding to the nearest points for each earthcare point

   
    

    if colocate_to_sat_1_sat_2=='EarthCare':
        # loop over earthcare points
        for i in tqdm.tqdm(np.arange(earth_latitudes.size)):
            # loop over aws points in time
            for j in range(len(aws_latitudes[:,0])):

                #old approch: 
                # compute distance between earthcare point and all aws swath points at time j
                dists =calculate_distances_for_one_Earthcare_sample_faster(aws_latitudes[j, :], aws_longitudes[j, :], earth_latitudes[i], earth_longitudes[i])
                # find minimum distance and store it
                distances[i, j] = np.nanmin(dists)
                aws_selected_swath_indices[i,j]=np.nanargmin(dists)

                #print(distances.shape)
        # find the indices of the minimum distances
            min_distance_indices = np.argmin(distances, axis=1)
                #print(min_distance_indices)

                #replaced with accurate approach:
                #dists = calculate_distances_for_one_Earthcare_sample_faster(aws_latitudes[j, :], aws_longitudes[j, :], earth_latitudes[i], earth_longitudes[i])

                #if np.all(~np.isfinite(dists)):
                #    distances[i, j] = np.nan
                #    aws_selected_swath_indices[i, j] = -1   # sentinel meaning "no valid swath"
                #else:
                #    distances[i, j] = np.nanmin(dists)
                #    aws_selected_swath_indices[i, j] = int(np.nanargmin(dists))
            # later when picking per-row minima:
            #min_distance_indices = np.full(distances.shape[0], -1, dtype=int)
            #for k in range(distances.shape[0]):
            #    if np.all(~np.isfinite(distances[k, :])):
            #        min_distance_indices[k] = -1  # no valid minimum
            #    else:
            #        min_distance_indices[k] = int(np.nanargmin(distances[k, :]))
    # else leave as -1 and skip indexing with it
            sel_aws_time_array_along_aws_traj[i]=aws_idx[min_distance_indices[i]]
            sel_aws_swath_array_along_aws_traj[i]=aws_selected_swath_indices[i,min_distance_indices[i]]


    
            

    
    return distances,sel_aws_swath_array_along_aws_traj,sel_aws_time_array_along_aws_traj

def select_nearest_data_samples_faster(aws_datatree, earthcare_datatree, time, delay, aws_channel_group=3, colocate_to_sat_1_sat_2='EarthCare'):
    # ensure 'time' is a timestamp
    time_ts = pd.to_datetime(time)
    delay__ = delay

    lower_bound = np.datetime64(time_ts - pd.Timedelta(minutes=5.5))
    upper_bound = np.datetime64(time_ts + pd.Timedelta(minutes=5.5))

    lower_bound_earthcare = np.datetime64(time_ts - pd.Timedelta(minutes=5.5) + pd.Timedelta(minutes=delay__))
    upper_bound_earthcare = np.datetime64(time_ts + pd.Timedelta(minutes=5.5) + pd.Timedelta(minutes=delay__))

    if isinstance(aws_datatree, xr.DataTree):
        aws_times = aws_datatree.data.navigation.time_startscan_utc_earthview.values
    if isinstance(aws_datatree, xr.Dataset):
        aws_times = aws_datatree.time.values
    earth_times = earthcare_datatree.ScienceData.time.values

    print(earth_times.size)

    aws_idx = np.where((aws_times >= lower_bound) & (aws_times <= upper_bound))[0]
    earth_idx = np.where((earth_times >= lower_bound_earthcare) & (earth_times <= upper_bound_earthcare))[0]

    earth_latitudes = earthcare_datatree.ScienceData.latitude[earth_idx].values
    earth_longitudes = earthcare_datatree.ScienceData.longitude[earth_idx].values

    if isinstance(aws_datatree, xr.DataTree):
        aws_latitudes = aws_datatree.data.navigation.aws_lat[aws_idx, :, aws_channel_group].values
        aws_longitudes = aws_datatree.data.navigation.aws_lon[aws_idx, :, aws_channel_group].values
    if isinstance(aws_datatree, xr.Dataset):
        aws_latitudes = aws_datatree.aws_lat[aws_idx, :].values
        aws_longitudes = aws_datatree.aws_lon[aws_idx, :].values

    # Dimensions:
    # nt = number of aws time samples (len(aws_latitudes[:, 0]))
    # ns = number of swath samples (aws_latitudes.shape[1])
    nt = aws_latitudes.shape[0]
    ns = aws_latitudes.shape[1]
    ne = earth_latitudes.size

    # distances[i, j] = minimal distance (km) between earthcare sample i and aws swath points at aws time j
    distances = np.full((ne, nt), np.nan)
    # aws_selected_swath_indices[i, j] = swath index (0..ns-1) of the nearest swath at that time j, or -1 if none valid
    aws_selected_swath_indices = np.full((ne, nt), -1, dtype=int)

    # Selectors for final chosen time & swath for each earthcare sample (sentinel -1 = none)
    sel_aws_time_array_along_aws_traj = np.full(ne, -1, dtype=int)
    sel_aws_swath_array_along_aws_traj = np.full(ne, -1, dtype=int)

    # For each EarthCare sample compute dmat (nt, ns) in one vectorized call and reduce across axis=1
    for i in tqdm.tqdm(np.arange(ne)):
        # dmat shape (nt, ns): vectorized haversine between all aws points and earthcare point i
        dmat = haversine_km_array(aws_latitudes, aws_longitudes, earth_latitudes[i], earth_longitudes[i])  # broadcasts to (nt, ns)

        # Replace non-finite entries with +inf to make argmin/min safe
        dmat_fill = np.where(np.isfinite(dmat), dmat, np.inf)

        # row-wise minima and swath indices (per aws time)
        row_min = np.min(dmat_fill, axis=1)           # shape (nt,)
        row_argmin = np.argmin(dmat_fill, axis=1)     # shape (nt,)

        # convert inf back to nan and mark invalid swath index as -1
        invalid_mask = np.isinf(row_min)
        row_min[invalid_mask] = np.nan
        row_argmin[invalid_mask] = -1

        distances[i, :] = row_min
        aws_selected_swath_indices[i, :] = row_argmin.astype(int)

    # Now pick the best aws time (and associated swath) per EarthCare sample (i.e., argmin across times)
    # Fill inf for rows that are entirely NaN to handle argmin safely
    distances_fill = np.where(np.isfinite(distances), distances, np.inf)
    min_time_indices = np.argmin(distances_fill, axis=1)     # gives a time index per EarthCare sample
    min_time_values = np.min(distances_fill, axis=1)        # gives distance value (inf if no valid)

    no_valid_mask = np.isinf(min_time_values)
    min_time_indices[no_valid_mask] = -1  # sentinel

    # Map local aws time indices to actual aws_idx values; set -1 for invalid
    for k in range(ne):
        if min_time_indices[k] == -1:
            sel_aws_time_array_along_aws_traj[k] = -1
            sel_aws_swath_array_along_aws_traj[k] = -1
        else:
            sel_aws_time_array_along_aws_traj[k] = int(aws_idx[int(min_time_indices[k])])
            sw = int(aws_selected_swath_indices[k, int(min_time_indices[k])])
            sel_aws_swath_array_along_aws_traj[k] = sw if sw >= 0 else -1

    return distances, sel_aws_swath_array_along_aws_traj, sel_aws_time_array_along_aws_traj


def get_along_AWS_path_closest_earthcare_points_faster(aws_datatree,earthcare_datatree,
                                                time,delay,aws_channel_group=3,
                                                number_of_nearest_EC_samples=5):

     # ensure 'time' is a timestamp
    time_ts = pd.to_datetime(time)
    delay__=delay

    lower_bound = np.datetime64(time_ts - pd.Timedelta(minutes=5.5))
    upper_bound = np.datetime64(time_ts + pd.Timedelta(minutes=5.5))

    lower_bound_earthcare = np.datetime64(time_ts - pd.Timedelta(minutes=5.5)+ pd.Timedelta(minutes=delay__))
    upper_bound_earthcare = np.datetime64(time_ts + pd.Timedelta(minutes=5.5)+ pd.Timedelta(minutes=delay__))

    if isinstance(aws_datatree,xr.DataTree):
        aws_times = aws_datatree.data.navigation.time_startscan_utc_earthview.values

    if isinstance(aws_datatree,xr.Dataset):
        aws_times=aws_datatree.time.values
    earth_times = earthcare_datatree.ScienceData.time.values


    # use elementwise & (or np.logical_and) instead of Python 'and'
    aws_idx = np.where((aws_times >= lower_bound) & (aws_times <= upper_bound))[0]
    earth_idx = np.where((earth_times >= lower_bound_earthcare) & (earth_times <= upper_bound_earthcare))[0]

    ## calculate distances for each earthcare time sample to all aws swath samples and find the minimum distance index
    earth_latitudes = earthcare_datatree.ScienceData.latitude[earth_idx].values
    earth_longitudes = earthcare_datatree.ScienceData.longitude[earth_idx].values 
    if isinstance(aws_datatree,xr.DataTree):  
        aws_latitudes = aws_datatree.data.navigation.aws_lat[aws_idx,:,aws_channel_group].values
        aws_longitudes = aws_datatree.data.navigation.aws_lon[aws_idx,:,aws_channel_group].values
    if isinstance(aws_datatree,xr.Dataset):
        aws_latitudes = aws_datatree.aws_lat[aws_idx,:].values
        aws_longitudes = aws_datatree.aws_lon[aws_idx,:].values

    # set up arrays to store EC neighbours
    #distances=np.nan*np.zeros((aws_idx.size,aws_latitudes[1,:].size,number_of_nearest_EC_samples))
    #earth_ids=np.nan*np.zeros((aws_idx.size,aws_latitudes[1,:].size,number_of_nearest_EC_samples))


    distances_2D=np.nan*np.zeros((aws_idx.size*aws_latitudes[1,:].size,number_of_nearest_EC_samples))
    earth_ids_2d=np.nan*np.zeros((aws_idx.size*aws_latitudes[1,:].size,number_of_nearest_EC_samples))

    aws_latitudes_2D=np.reshape(aws_latitudes,(aws_idx.size*aws_latitudes[1,:].size))
    aws_longitudes_2D=np.reshape(aws_longitudes,(aws_idx.size*aws_latitudes[1,:].size))

    

    for i in tqdm.tqdm(np.arange(aws_latitudes_2D[:].size)):

            
        # loop over aws points in time
        # compute distance between earthcare point and all aws swath points at time j
        dists =calculate_distances_for_one_AWS_sample_array(aws_latitudes_2D[i],aws_longitudes_2D[i], earth_latitudes, earth_longitudes)
        # find nearest n distances and store tehm plus IDs
        distances_2D[i] = np.sort(dists)[0:number_of_nearest_EC_samples]
        earth_ids_2d[i]=np.argsort(dists)[number_of_nearest_EC_samples]

    distances=np.reshape(distances_2D,(aws_idx.size,aws_latitudes[1,:].size,number_of_nearest_EC_samples))
    earth_ids=np.reshape(earth_ids_2d,(aws_idx.size,aws_latitudes[1,:].size,number_of_nearest_EC_samples))

    return distances,earth_ids,aws_idx,aws_times 


     

    
def calculate_distances_for_one_AWS_sample_array(AWS_lat,AWS_long,EartCare_lat_array,EarthCare_long_array):
    dist__=np.nan*np.zeros((EarthCare_long_array.size))
    
    dist__=haversine_km_array(AWS_lat,AWS_long,EartCare_lat_array,EarthCare_long_array)
    
    return dist__  


def colocate_cross_section_with_Earthcare(aws_datatree,aws_time_index,aws_swath_index,select_AWS_channel=3):
    # extract cross section from aws datatree at given time and swath index

    if isinstance(aws_datatree,xr.DataTree):
        aws_latitudes = aws_datatree.data.navigation.aws_lat
        aws_longitudes = aws_datatree.data.navigation.aws_lon

    if isinstance(aws_datatree,xr.Dataset):
        aws_latitudes = aws_datatree.aws_lat
        aws_longitudes = aws_datatree.aws_lon



    sel_aws_latitudes=np.zeros((aws_swath_index.size,1))
    sel_aws_longitudes=np.zeros((aws_swath_index.size,1))


    for i in range(aws_swath_index.size):
        #print(aws_latitudes[int(aws_time_index[i]),int(aws_swath_index[i]),select_AWS_channel].values.size)
        if isinstance(aws_datatree,xr.DataTree):

            sel_aws_latitudes[i]=aws_latitudes[int(aws_time_index[i]),int(aws_swath_index[i]),select_AWS_channel].values
            sel_aws_longitudes[i]=aws_longitudes[int(aws_time_index[i]),int(aws_swath_index[i]),select_AWS_channel].values

        if isinstance(aws_datatree,xr.Dataset):

            sel_aws_latitudes[i]=aws_latitudes[int(aws_time_index[i]),int(aws_swath_index[i])].values
            sel_aws_longitudes[i]=aws_longitudes[int(aws_time_index[i]),int(aws_swath_index[i])].values


    
    return sel_aws_latitudes,sel_aws_longitudes


def colocate_cross_section_AWS_icewater_with_Earthcare(aws_datatree,aws_time_index,aws_swath_index,select_AWS_channel=3):
    # extract cross section from aws datatree at given time and swath index

    if isinstance(aws_datatree,xr.DataTree):
        aws_icewater = aws_datatree.data.calibration.aws_toa_brightness_temperature.values
    if isinstance(aws_datatree,xr.Dataset):
        aws_icewater = aws_datatree.aws_toa_brightness_temperature.values


    sel_aws_icewater_=np.zeros((aws_swath_index.size,1))


    for i in tqdm.tqdm(np.arange(aws_swath_index.size)):
        #print(aws_latitudes[int(aws_time_index[i]),int(aws_swath_index[i]),select_AWS_channel].values.size)
        sel_aws_icewater_[i]=aws_icewater[int(aws_time_index[i]),int(aws_swath_index[i]),select_AWS_channel]

    
    return sel_aws_icewater_








def earthcare_load(time__, delay__, loc__,product_type='/L2b/ACM_CAP_2B/'):
    """Load nearest EarthCare file based on time_ea within ±1 hour window.
    
    Args:
        time__: colocation timestamp (str or datetime)
        delay__: delay in minutes (timedelta or numeric)
        loc__: location ('Antarctic' or 'Arctic')
    
    Returns:
        xr.DataTree with EarthCare data
    """
    # Convert time to pandas Timestamp and add delay
    time_ea = pd.to_datetime(time__) + pd.Timedelta(minutes=float(delay__))

    if time_ea>=pd.Timestamp("2025-12-01"):
            base_path = Path('/data/s13/EarthCare'+product_type+'BC/')
    if time_ea<pd.Timestamp("2025-12-01"):
            base_path = Path('/data/s13/EarthCare'+product_type+'BA/')


         


    
    # Define search window: -1 hour to +0 (time_ea)
    search_start = time_ea - pd.Timedelta(hours=1)
    search_end = time_ea + pd.Timedelta(hours=1)
    
    # Set base path and end letter based on location
    #base_path = Path('/data/s13/EarthCare/L2b/ACM_CAP_2B/')
    end_let = 'G' if loc__ == 'Antarctic' else ''
    
    # Search across potentially multiple days (if search spans midnight)
    search_dates = pd.date_range(start=search_start.date(), end=search_end.date(), freq='D')
    
    all_h5_files = []
    
    for search_date in search_dates:
        year_str = search_date.strftime('%Y')
        month_str = search_date.strftime('%m')
        day_str = search_date.strftime('%d')
        
        search_dir = base_path / year_str / month_str / day_str
        
        if search_dir.exists():
            # Find all h5 files ending with end_let
            h5_files = list(search_dir.glob(f'*{end_let}/*.h5'))
            all_h5_files.extend(h5_files)
    
    if not all_h5_files:
        print(f"No EarthCare files found for location {loc__} between {search_start} and {search_end}")
        return None
    
    # Extract timestamps from folder names and find closest to time_ea
    closest_file = None
    min_time_diff = None
    
    for h5_file in all_h5_files:
        folder_name = h5_file.parent.name
        try:
            # Parse start timestamp from folder name (format: ECA_EXBA_ACM_CAP_2B_20250707T102155Z_...)
            start_time_str = folder_name.split('_')[5]  # Extract timestamp
            start_time = pd.to_datetime(start_time_str, format='%Y%m%dT%H%M%SZ')
            
            # Check if file's start time falls within search window
            if search_start <= start_time <= search_end:
                time_diff = abs((time_ea - start_time).total_seconds())
                
                if min_time_diff is None or time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_file = h5_file
        except Exception as e:
            print(f"Error parsing {folder_name}: {e}")
            continue
    
    if closest_file is None:
        print(f"No suitable EarthCare file found within window {search_start} to {search_end}")
        return None
    
    #print(f"Loading: {closest_file}")
    #print(f"Time difference: {min_time_diff} seconds")
    ea_data = xr.open_datatree(str(closest_file), engine='h5netcdf')

    
    return ea_data, closest_file




def plot_distance_matrix_selected_cross_section(distances,aws_swath_indices,aws_time_indices,
                                                earthcare_datatree,aws_datatree,Antarctica_or_Arctic,
                                                time,delay,dist_thres=20,plot_figures=True):
    
    time_ts = pd.to_datetime(time)
    delay__=delay

    

    lower_bound_earthcare = np.datetime64(time_ts - pd.Timedelta(minutes=5.5)+ pd.Timedelta(minutes=delay__))
    upper_bound_earthcare = np.datetime64(time_ts + pd.Timedelta(minutes=5.5)+ pd.Timedelta(minutes=delay__))

    earth_times = earthcare_datatree.ScienceData.time.values

    print(earth_times.size)

    # use elementwise & (or np.logical_and) instead of Python 'and'
    earth_idx = np.where((earth_times >= lower_bound_earthcare) & (earth_times <= upper_bound_earthcare))[0]

    dist_thres_clip=np.where(np.nanmin(distances,1)<=dist_thres)[0]
    if plot_figures==True:
        plt.figure(7,(21,6))

        plt.subplot(1,3,1)
        plt.imshow(distances,aspect='auto',cmap='Blues',extent=[0,distances.shape[1],earthcare_datatree.ScienceData.latitude.size,0])
        plt.colorbar(label='Distance (km)')
        plt.grid(True)
        plt.xlabel('AWS Time Index',fontsize=14)
        plt.ylabel('EarthCare Time Index',fontsize=14)
        plt.title('Distance Matrix between EarthCare and AWS Samples\nwith Selected Nearest AWS Samples Overlayed',fontsize=16)

        plt.subplot(1,3,2)
        plt.plot(aws_time_indices,aws_swath_indices,'r-',label='Selected nearest AWS samples')
        plt.grid(True)

        plt.xlabel('AWS Time Index',fontsize=14)
        plt.ylabel('AWS Swath Index',fontsize=14)
        plt.title('Selected Nearest AWS Samples for Each EarthCare Sample',fontsize=16)

        plt.subplot(1,3,3)
        plt.plot(np.nanmin(distances,1),aws_swath_indices,'x')
        plt.grid(True)
        plt.ylabel('AWS Swath Index',fontsize=14)
        plt.xlabel('Distances to nearest AWS sample [km]',fontsize=14)
        plt.title('Distances vs. Swath IDs', fontsize=16)


    aws_traj_lat,aws_traj_lon=colocate_cross_section_with_Earthcare(aws_datatree,aws_time_indices,aws_swath_indices,3)
    
    if plot_figures==True:
        plt.figure(9,(12,18))

        plt.subplot(3,1,1)
        plt.plot(earthcare_datatree.ScienceData.longitude[earth_idx].values,earthcare_datatree.ScienceData.latitude[earth_idx].values, 'x')
        plt.plot(earthcare_datatree.ScienceData.longitude[earth_idx].values,aws_traj_lat, 'x')
        plt.ylabel('Latitude', fontsize=16)
        plt.grid(True)


        plt.subplot(3,1,2)
        plt.plot(earthcare_datatree.ScienceData.longitude[earth_idx].values,np.nanmin(distances,1), 'bx')
        plt.ylabel('Distances [km]', fontsize=16)
        plt.xlabel('Longitude',fontsize=16)
        plt.grid(True)

        plt.subplot(3,1,3)
        plt.plot(earth_times[earth_idx],np.nanmin(distances,1), 'bx')
        plt.xlabel('Time', fontsize=16)
        plt.ylabel('Distances [km]',fontsize=16)
        plt.grid(True)



        plt.figure(8,(10,10))
        if Antarctica_or_Arctic=='Arctic':
            ax=plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=150))
            ax.set_extent([np.min(earthcare_datatree.ScienceData.longitude[earth_idx])-1.5,
                        np.max(earthcare_datatree.ScienceData.longitude[earth_idx])+1.5,
                          np.min(earthcare_datatree.ScienceData.latitude[earth_idx])-1,
                           np.max(earthcare_datatree.ScienceData.latitude[earth_idx])+1], ccrs.PlateCarree())
        else:
            ax=plt.axes(projection=ccrs.SouthPolarStereo(central_longitude=150))
            ax.set_extent([np.min(earthcare_datatree.ScienceData.longitude[earth_idx])-5,
                        np.max(earthcare_datatree.ScienceData.longitude[earth_idx])+5,
                        np.min(earthcare_datatree.ScienceData.latitude[earth_idx])-1,
                          np.max(earthcare_datatree.ScienceData.latitude[earth_idx])+2], ccrs.PlateCarree())  
    
        ax.plot(earthcare_datatree.ScienceData.longitude[earth_idx].values,earthcare_datatree.ScienceData.latitude[earth_idx].values,'x', markersize=2,
            label='EarthCare Trajectory', transform=ccrs.PlateCarree())
        ax.plot(aws_traj_lon,aws_traj_lat,'x', markersize=2,label='Selected Nearest AWS Samples Trajectory', transform=ccrs.PlateCarree())
        ax.coastlines(color='black',linewidth=2)
        ax.gridlines()
        plt.legend(fontsize=14)
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.grid(True)

    return earth_idx,dist_thres_clip



def average_quantities_along_AWS(earth_datatre_var,eartidx_matr,eart_dist):
    
    avg_field=np.nan*np.zeros(eartidx_matr[:,:,1].shape)
    eart_var=earth_datatre_var

    for i in np.arange(eartidx_matr[:,1,1].size):
        for j in np.arange(eartidx_matr[1,:,1].size):

            if np.median(eart_dist[i,j,:])<15:
                avg_field[i,j]=np.mean(eart_var[eartidx_matr[i,j,:].astype(int)])

    return avg_field



def average_quantities_along_AWS_max(earth_datatre_var,eartidx_matr,eart_dist):
    
    avg_field=np.nan*np.zeros(eartidx_matr[:,:,1].shape)
    eart_var=earth_datatre_var

    for i in np.arange(eartidx_matr[:,1,1].size):
        for j in np.arange(eartidx_matr[1,:,1].size):

            if np.max(eart_dist[i,j,:])<15:
                avg_field[i,j]=np.mean(eart_var[eartidx_matr[i,j,:].astype(int)])

    return avg_field








def average_quantities_along_AWS_but_retain_vert_column(earth_datatre_var,eartidx_matr,eart_dist,prod='CAP'):
    if prod=='CAP':
        avg_field=np.nan*np.zeros((eartidx_matr[:,1,1].size,eartidx_matr[1,:,1].size,242))#earth_datatre_var.))
    if prod=='CPR':
        avg_field=np.nan*np.zeros((eartidx_matr[:,1,1].size,eartidx_matr[1,:,1].size,218))

    eart_var=earth_datatre_var

    for i in np.arange(eartidx_matr[:,1,1].size):
        for j in np.arange(eartidx_matr[1,:,1].size):

            if np.median(eart_dist[i,j,:])<15:
                avg_field[i,j]=np.mean(eart_var[eartidx_matr[i,j,:].astype(int)],0)

    return avg_field



def average_quantities_along_AWS_but_retain_vert_column_max(earth_datatre_var,eartidx_matr,eart_dist,prod='CAP'):
    if prod=='CAP':
        avg_field=np.nan*np.zeros((eartidx_matr[:,1,1].size,eartidx_matr[1,:,1].size,242))#earth_datatre_var.))
    if prod=='CPR':
        avg_field=np.nan*np.zeros((eartidx_matr[:,1,1].size,eartidx_matr[1,:,1].size,218))

    eart_var=earth_datatre_var

    for i in np.arange(eartidx_matr[:,1,1].size):
        for j in np.arange(eartidx_matr[1,:,1].size):

            if np.max(eart_dist[i,j,:])<15:
                avg_field[i,j]=np.mean(eart_var[eartidx_matr[i,j,:].astype(int)],0)

    return avg_field




def plot_AWS_along_Earthcare_trajectory(ea_datatree,ea_CPR_L2a_datatree,aws_dataset,aws_time_idx,aws_swath_idx,earthcare_time_idx):
    
    plt.figure(4,(25,25))

    plt.subplot(3,1,1)   
    
    
    
    plt.plot(ea_datatree.ScienceData.time[earthcare_time_idx],
         colocate_cross_section_AWS_icewater_with_Earthcare(aws_dataset,aws_time_idx,aws_swath_idx,
                                                            aws_dataset.n_channels=='AWS21'),label='AWS21',color='C2')

    plt.plot(ea_datatree.ScienceData.time[earthcare_time_idx],
         colocate_cross_section_AWS_icewater_with_Earthcare(aws_dataset,aws_time_idx,aws_swath_idx,
                                                            aws_dataset.n_channels=='AWS31'),label='AWS31',color='C3')
 
    plt.plot(ea_datatree.ScienceData.time[earthcare_time_idx],
         colocate_cross_section_AWS_icewater_with_Earthcare(aws_dataset,aws_time_idx,
                                                            aws_swath_idx,aws_dataset.n_channels=='AWS33'),label='AWS33',color='C1')
    plt.plot(ea_datatree.ScienceData.time[earthcare_time_idx],
         colocate_cross_section_AWS_icewater_with_Earthcare(aws_dataset,aws_time_idx,aws_swath_idx,
                                                            aws_dataset.n_channels=='AWS44'),label='AWS44',color='C0')
    
    

    plt.title('AWS brightness temperatures of AWS33 and AWS44 channels',fontsize=20)
    plt.ylabel('Brightness temperature [K]',fontsize=17)
    plt.xlabel('EarthCare time',fontsize=17)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.xlim(np.min(ea_datatree.ScienceData.time[earthcare_time_idx].values),
             np.max(ea_datatree.ScienceData.time[earthcare_time_idx].values))

    ax=plt.subplot(3,1,3)

    plt.pcolor(ea_datatree.ScienceData.time[earthcare_time_idx],ea_datatree.ScienceData.height[1,:],
           ea_datatree.ScienceData.CPR_reflectivity_factor[earthcare_time_idx].T,vmin=-50,vmax=10,cmap=plt.cm.gist_stern_r,)
    
    plt.fill_between(ea_datatree.ScienceData.time[earthcare_time_idx],
             ea_datatree.ScienceData.elevation[earthcare_time_idx],color='grey')


    
    cbar =plt.colorbar(location='bottom')
    cbar.ax.tick_params(labelsize=15)


    plt.title('Earthcare CPR reflectivity factor',fontsize=20)
    plt.ylabel('Height [m]',fontsize=17)
    plt.ylim(0,14000)
    plt.xlabel('EarthCare time',fontsize=17)
    plt.grid(True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)



    plt.subplot(3,1,2)

    plt.plot(ea_datatree.ScienceData.time[earthcare_time_idx],
             ea_CPR_L2a_datatree.ScienceData.brightness_temperature.values[earthcare_time_idx])

    plt.title('Earthcare CPR passive brightness temperature',fontsize=20)
    plt.ylabel('Brightness temp [K]',fontsize=17)
    plt.xlabel('EarthCare time',fontsize=17)
    plt.grid(True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(np.min(ea_datatree.ScienceData.time[earthcare_time_idx].values),
             np.max(ea_datatree.ScienceData.time[earthcare_time_idx].values))
    




def plot_AWS_along_Earthcare_trajectory_clip(ea_datatree,ea_CPR_L2a_datatree,aws_dataset,aws_time_idx,aws_swath_idx,earthcare_time_idx):
    
    plt.figure(4,(25,15))

    plt.subplot(2,1,1)   
    
    
    
    plt.plot(ea_datatree.ScienceData.time[earthcare_time_idx],
         colocate_cross_section_AWS_icewater_with_Earthcare(aws_dataset,aws_time_idx,aws_swath_idx,
                                                            aws_dataset.n_channels=='AWS21'),label='AWS21',color='C2')

    plt.plot(ea_datatree.ScienceData.time[earthcare_time_idx],
         colocate_cross_section_AWS_icewater_with_Earthcare(aws_dataset,aws_time_idx,aws_swath_idx,
                                                            aws_dataset.n_channels=='AWS31'),label='AWS31',color='C3')
 
    plt.plot(ea_datatree.ScienceData.time[earthcare_time_idx],
         colocate_cross_section_AWS_icewater_with_Earthcare(aws_dataset,aws_time_idx,
                                                            aws_swath_idx,aws_dataset.n_channels=='AWS33'),label='AWS33',color='C1')
    plt.plot(ea_datatree.ScienceData.time[earthcare_time_idx],
         colocate_cross_section_AWS_icewater_with_Earthcare(aws_dataset,aws_time_idx,aws_swath_idx,
                                                            aws_dataset.n_channels=='AWS44'),label='AWS44',color='C0')
    
    

    plt.title('AWS brightness temperatures of AWS33 and AWS44 channels',fontsize=20)
    plt.ylabel('Brightness temperature [K]',fontsize=17)
    plt.xlabel('EarthCare time',fontsize=17)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.xlim(np.min(ea_datatree.ScienceData.time[earthcare_time_idx].values),
             np.max(ea_datatree.ScienceData.time[earthcare_time_idx].values))

    ax=plt.subplot(2,1,2)

    plt.pcolor(ea_datatree.ScienceData.time[earthcare_time_idx],ea_datatree.ScienceData.height[1,:],
           ea_datatree.ScienceData.CPR_reflectivity_factor[earthcare_time_idx].T,vmin=-50,vmax=10,cmap=plt.cm.gist_stern_r,)
    
    plt.fill_between(ea_datatree.ScienceData.time[earthcare_time_idx],
             ea_datatree.ScienceData.elevation[earthcare_time_idx],color='grey')


    
    cbar =plt.colorbar(location='bottom')
    cbar.ax.tick_params(labelsize=15)


    plt.title('Earthcare CPR reflectivity factor',fontsize=20)
    plt.ylabel('Height [m]',fontsize=17)
    plt.ylim(0,14000)
    plt.xlabel('EarthCare time',fontsize=17)
    plt.grid(True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)



    



def plot_Eartcare_along_AWS_scan(ea_datatree,ea_CPR_L2a_datatree,aws_dataset,
                                 earthcare_indices_along_AWS,
                                 earthcare_distance_matrix, 
                                 aws_time_idxs,Antarctica_or_Arctic):
    
    a=aws_dataset.n_channels=='AWS33'
    b=aws_dataset.n_channels=='AWS44'
    c=aws_dataset.n_channels=='AWS31'
    d=aws_dataset.n_channels=='AWS21'



    rad_refle_max=average_quantities_along_AWS(ea_datatree.ScienceData.CPR_reflectivity_factor.max(dim='JSG_height').values,
                                           earthcare_indices_along_AWS,earthcare_distance_matrix)
    
    brightness_temp_mean=average_quantities_along_AWS(ea_CPR_L2a_datatree.ScienceData.brightness_temperature.values,
                                           earthcare_indices_along_AWS,earthcare_distance_matrix)
    

    #todo check for singularities in AWS longitudes

    aws_dataset_lon_=aws_dataset.aws_lon[aws_time_idxs,:].values
    aws_dataset_lon=np.nan_to_num(aws_dataset_lon_,nan=180)

    aws_dataset_lat_=aws_dataset.aws_lat[aws_time_idxs,:].values
    if Antarctica_or_Arctic=='Arctic':

      aws_dataset_lat=np.nan_to_num(aws_dataset_lat_,nan=np.nanmedian(np.nanmax(aws_dataset_lat_,0)))
    if Antarctica_or_Arctic=='Antarctic':
      aws_dataset_lat=np.nan_to_num(aws_dataset_lat_,nan=np.nanmedian(np.nanmax(aws_dataset_lat_,0)))



   

    if Antarctica_or_Arctic=='Arctic':
        fig,ax=plt.subplots(2,3, figsize=(24,14),
                     subplot_kw={'projection':ccrs.NorthPolarStereo(central_longitude=np.nanmedian(aws_dataset_lon))})
    
    if Antarctica_or_Arctic=='Antarctic':
        fig,ax=plt.subplots(2,3, figsize=(24,14),
                     subplot_kw={'projection':ccrs.SouthPolarStereo(central_longitude=np.nanmedian(aws_dataset_lon))})

    a1=ax[0][2].pcolormesh(aws_dataset_lon,aws_dataset_lat,
          np.squeeze(aws_dataset.aws_toa_brightness_temperature[aws_time_idxs,:,a].values),
          vmin=np.nanmin(np.squeeze(aws_dataset.aws_toa_brightness_temperature[aws_time_idxs,:,a].values)),
          vmax=np.nanmax(np.squeeze(aws_dataset.aws_toa_brightness_temperature[aws_time_idxs,:,a].values)),
          cmap=plt.cm.gist_stern,
          transform=ccrs.PlateCarree())


    ax[0][2].coastlines(color='black',linewidth=2)
    ax[0][2].set_title('AWS 33 brightness temp [K]',fontsize=16)
    fig.colorbar(a1)


    a2=ax[1][0].pcolormesh(aws_dataset_lon,aws_dataset_lat,
          np.squeeze(aws_dataset.aws_toa_brightness_temperature[aws_time_idxs,:,a].values)-np.squeeze(aws_dataset.aws_toa_brightness_temperature[aws_time_idxs,:,b].values),
          vmin=-30,
          vmax=30,
          cmap=plt.cm.RdBu_r,
          transform=ccrs.PlateCarree())


    ax[1][0].coastlines(color='black',linewidth=2)
    ax[1][0].set_title('AWS 33-44 brightness temp. diff [K]',fontsize=16)
    fig.colorbar(a2)


    a3=ax[1][1].pcolormesh(aws_dataset_lon,aws_dataset_lat,
          np.squeeze(rad_refle_max),
          vmin=-40,
          vmax=5,
          cmap=plt.cm.Blues,
          transform=ccrs.PlateCarree())


    ax[1][1].coastlines(color='black',linewidth=1)
    ax[1][1].set_title('EarthCare maximum radar reflectivity',fontsize=16)
    fig.colorbar(a3)




    a4=ax[0][1].pcolormesh(aws_dataset_lon,aws_dataset_lat,
          np.squeeze(aws_dataset.aws_toa_brightness_temperature[aws_time_idxs,:,c].values),
          vmin=np.nanmin(np.squeeze(aws_dataset.aws_toa_brightness_temperature[aws_time_idxs,:,c].values)),
          vmax=np.nanmax(np.squeeze(aws_dataset.aws_toa_brightness_temperature[aws_time_idxs,:,c].values)),
          cmap=plt.cm.gist_stern,
          transform=ccrs.PlateCarree())

    fig.colorbar(a4)
    ax[0][1].coastlines(color='black',linewidth=2)
    ax[0][1].set_title('AWS 31 brightness temp [K]',fontsize=16)



    a5=ax[0][0].pcolormesh(aws_dataset_lon,aws_dataset_lat,
          np.squeeze(aws_dataset.aws_toa_brightness_temperature[aws_time_idxs,:,d].values),
          vmin=np.nanmin(np.squeeze(aws_dataset.aws_toa_brightness_temperature[aws_time_idxs,:,d].values)),
          vmax=np.nanmax(np.squeeze(aws_dataset.aws_toa_brightness_temperature[aws_time_idxs,:,d].values)),
          cmap=plt.cm.gist_stern,
          transform=ccrs.PlateCarree())


    ax[0][0].coastlines(color='black',linewidth=2)
    ax[0][0].set_title('AWS 21 brightness temp [K]',fontsize=16)
    fig.colorbar(a5)


    

    a6=ax[1][2].pcolormesh(aws_dataset_lon,aws_dataset_lat,
          np.squeeze(brightness_temp_mean),
          vmin=np.nanmin(np.squeeze(brightness_temp_mean)),
          vmax=np.nanmax(np.squeeze(brightness_temp_mean)),
          cmap=plt.cm.gist_stern,
          transform=ccrs.PlateCarree())


    ax[1][2].coastlines(color='black',linewidth=1)
    ax[1][2].set_title('EarthCare radar brightness temp at 94 GHz',fontsize=16)
    fig.colorbar(a6)




def plot_Eartcare_along_AWS_scan_special(ea_datatree,ea_CPR_L2a_datatree,aws_dataset,
                                 earthcare_indices_along_AWS,
                                 earthcare_distance_matrix, 
                                 aws_time_idxs,Antarctica_or_Arctic):
    
    a=aws_dataset.n_channels=='AWS33'
    b=aws_dataset.n_channels=='AWS44'
    c=aws_dataset.n_channels=='AWS31'
    d=aws_dataset.n_channels=='AWS21'



    rad_refle_max=average_quantities_along_AWS(ea_datatree.ScienceData.CPR_reflectivity_factor.max(dim='JSG_height').values,
                                           earthcare_indices_along_AWS,earthcare_distance_matrix)
    
    brightness_temp_mean=average_quantities_along_AWS(ea_CPR_L2a_datatree.ScienceData.brightness_temperature.values,
                                           earthcare_indices_along_AWS,earthcare_distance_matrix)
    

    #todo check for singularities in AWS longitudes

    aws_dataset_lon_=aws_dataset.aws_lon[aws_time_idxs,:].values
    aws_dataset_lon=np.nan_to_num(aws_dataset_lon_,nan=180)

    aws_dataset_lat_=aws_dataset.aws_lat[aws_time_idxs,:].values
    if Antarctica_or_Arctic=='Arctic':

      aws_dataset_lat=np.nan_to_num(aws_dataset_lat_,nan=np.nanmedian(np.nanmax(aws_dataset_lat_,0)))
    if Antarctica_or_Arctic=='Antarctic':
      aws_dataset_lat=np.nan_to_num(aws_dataset_lat_,nan=np.nanmedian(np.nanmax(aws_dataset_lat_,0)))



   

    if Antarctica_or_Arctic=='Arctic':
        fig,ax=plt.subplots(2,3, figsize=(24,14),
                     subplot_kw={'projection':ccrs.NorthPolarStereo(central_longitude=np.nanmedian(aws_dataset_lon))})
    
    if Antarctica_or_Arctic=='Antarctic':
        fig,ax=plt.subplots(2,3, figsize=(24,14),
                     subplot_kw={'projection':ccrs.SouthPolarStereo(central_longitude=np.nanmedian(aws_dataset_lon))})

    a1=ax[0][2].pcolormesh(aws_dataset_lon,aws_dataset_lat,
          np.squeeze(aws_dataset.aws_toa_brightness_temperature[aws_time_idxs,:,a].values),
          vmin=np.nanmin(np.squeeze(aws_dataset.aws_toa_brightness_temperature[aws_time_idxs,:,a].values)),
          vmax=np.nanmax(np.squeeze(aws_dataset.aws_toa_brightness_temperature[aws_time_idxs,:,a].values)),
          cmap=plt.cm.gist_stern,
          transform=ccrs.PlateCarree())
    
    ax[0][2].pcolormesh(aws_dataset_lon,aws_dataset_lat,
          np.squeeze(rad_refle_max),
          vmin=-999,
          vmax=200,
          cmap=plt.cm.Greys,
          transform=ccrs.PlateCarree())


    ax[0][2].coastlines(color='black',linewidth=2)
    ax[0][2].set_title('AWS 33 brightness temp [K]',fontsize=16)
    fig.colorbar(a1)


    a2=ax[1][0].pcolormesh(aws_dataset_lon,aws_dataset_lat,
          np.squeeze(aws_dataset.aws_toa_brightness_temperature[aws_time_idxs,:,a].values)-np.squeeze(aws_dataset.aws_toa_brightness_temperature[aws_time_idxs,:,b].values),
          vmin=-20,
          vmax=20,
          cmap=plt.cm.RdBu_r,
          transform=ccrs.PlateCarree())
    
    ax[1][0].pcolormesh(aws_dataset_lon,aws_dataset_lat,
          np.squeeze(rad_refle_max),
          vmin=-999,
          vmax=200,
          cmap=plt.cm.Greys,
          transform=ccrs.PlateCarree())


    ax[1][0].coastlines(color='black',linewidth=2)
    ax[1][0].set_title('AWS 33-44 brightness temp. diff [K]',fontsize=16)
    fig.colorbar(a2)


    a3=ax[1][1].pcolormesh(aws_dataset_lon,aws_dataset_lat,
          np.squeeze(rad_refle_max),
          vmin=-40,
          vmax=5,
          cmap=plt.cm.Blues,
          transform=ccrs.PlateCarree())


    ax[1][1].coastlines(color='black',linewidth=1)
    ax[1][1].set_title('EarthCare maximum radar reflectivity',fontsize=16)
    fig.colorbar(a3)




    a4=ax[0][1].pcolormesh(aws_dataset_lon,aws_dataset_lat,
          np.squeeze(aws_dataset.aws_toa_brightness_temperature[aws_time_idxs,:,c].values),
          vmin=np.nanmin(np.squeeze(aws_dataset.aws_toa_brightness_temperature[aws_time_idxs,:,c].values)),
          vmax=np.nanmax(np.squeeze(aws_dataset.aws_toa_brightness_temperature[aws_time_idxs,:,c].values)),
          cmap=plt.cm.gist_stern,
          transform=ccrs.PlateCarree())
    
    ax[0][1].pcolormesh(aws_dataset_lon,aws_dataset_lat,
          np.squeeze(rad_refle_max),
          vmin=-999,
          vmax=200,
          cmap=plt.cm.Greys,
          transform=ccrs.PlateCarree())

    fig.colorbar(a4)
    ax[0][1].coastlines(color='black',linewidth=2)
    ax[0][1].set_title('AWS 31 brightness temp [K]',fontsize=16)



    a5=ax[0][0].pcolormesh(aws_dataset_lon,aws_dataset_lat,
          np.squeeze(aws_dataset.aws_toa_brightness_temperature[aws_time_idxs,:,d].values),
          vmin=np.nanmin(np.squeeze(aws_dataset.aws_toa_brightness_temperature[aws_time_idxs,:,d].values)),
          vmax=np.nanmax(np.squeeze(aws_dataset.aws_toa_brightness_temperature[aws_time_idxs,:,d].values)),
          cmap=plt.cm.gist_stern,
          transform=ccrs.PlateCarree())
    
    ax[0][0].pcolormesh(aws_dataset_lon,aws_dataset_lat,
          np.squeeze(rad_refle_max),
          vmin=-999,
          vmax=200,
          cmap=plt.cm.Greys,
          transform=ccrs.PlateCarree())


    ax[0][0].coastlines(color='black',linewidth=2)
    ax[0][0].set_title('AWS 21 brightness temp [K]',fontsize=16)
    fig.colorbar(a5)


    

    a6=ax[1][2].pcolormesh(aws_dataset_lon,aws_dataset_lat,
          np.squeeze(brightness_temp_mean),
          vmin=np.nanmin(np.squeeze(brightness_temp_mean)),
          vmax=np.nanmax(np.squeeze(brightness_temp_mean)),
          cmap=plt.cm.gist_stern,
          transform=ccrs.PlateCarree())


    ax[1][2].coastlines(color='black',linewidth=1)
    ax[1][2].set_title('EarthCare radar brightness temp at 94 GHz',fontsize=16)
    fig.colorbar(a6)



def plot_Eartcare_along_AWS_scan_tiny(ea_datatree,ea_CPR_L2a_datatree,aws_dataset,
                                 earthcare_indices_along_AWS,
                                 earthcare_distance_matrix, 
                                 aws_time_idxs,Antarctica_or_Arctic):
    
    a=aws_dataset.n_channels=='AWS33'
    b=aws_dataset.n_channels=='AWS44'
    c=aws_dataset.n_channels=='AWS31'
    d=aws_dataset.n_channels=='AWS21'



    rad_refle_max=average_quantities_along_AWS(ea_datatree.ScienceData.CPR_reflectivity_factor.max(dim='JSG_height').values,
                                           earthcare_indices_along_AWS,earthcare_distance_matrix)
    
   

    #todo check for singularities in AWS longitudes

    aws_dataset_lon_=aws_dataset.aws_lon[aws_time_idxs,:].values
    aws_dataset_lon=np.nan_to_num(aws_dataset_lon_,nan=180)

    aws_dataset_lat_=aws_dataset.aws_lat[aws_time_idxs,:].values
    if Antarctica_or_Arctic=='Arctic':

      aws_dataset_lat=np.nan_to_num(aws_dataset_lat_,nan=np.nanmedian(np.nanmax(aws_dataset_lat_,0)))
    if Antarctica_or_Arctic=='Antarctic':
      aws_dataset_lat=np.nan_to_num(aws_dataset_lat_,nan=np.nanmedian(np.nanmax(aws_dataset_lat_,0)))



   

    if Antarctica_or_Arctic=='Arctic':
        fig,ax=plt.subplots(1,3, figsize=(18,6),
                     subplot_kw={'projection':ccrs.NorthPolarStereo(central_longitude=np.nanmedian(aws_dataset_lon))})
    
    if Antarctica_or_Arctic=='Antarctic':
        fig,ax=plt.subplots(1,3, figsize=(18,6),
                     subplot_kw={'projection':ccrs.SouthPolarStereo(central_longitude=np.nanmedian(aws_dataset_lon))})

    a1=ax[0].pcolormesh(aws_dataset_lon,aws_dataset_lat,
          np.squeeze(aws_dataset.aws_toa_brightness_temperature[aws_time_idxs,:,a].values),
          vmin=np.nanmin(np.squeeze(aws_dataset.aws_toa_brightness_temperature[aws_time_idxs,:,a].values)),
          vmax=np.nanmax(np.squeeze(aws_dataset.aws_toa_brightness_temperature[aws_time_idxs,:,a].values)),
          cmap=plt.cm.gist_stern,
          transform=ccrs.PlateCarree())
    
    ax[0].pcolormesh(aws_dataset_lon,aws_dataset_lat,
          np.squeeze(rad_refle_max),
          vmin=-999,
          vmax=200,
          cmap=plt.cm.Greys, alpha=0.25,
          transform=ccrs.PlateCarree())


    ax[0].coastlines(color='black',linewidth=2)
    ax[0].set_title('AWS 33 brightness temp [K]',fontsize=16)
    fig.colorbar(a1, orientation='horizontal')


    a2=ax[1].pcolormesh(aws_dataset_lon,aws_dataset_lat,
          np.squeeze(aws_dataset.aws_toa_brightness_temperature[aws_time_idxs,:,a].values)-np.squeeze(aws_dataset.aws_toa_brightness_temperature[aws_time_idxs,:,b].values),
          vmin=-10,
          vmax=10,
          cmap=plt.cm.RdBu_r,
          transform=ccrs.PlateCarree())
    
    ax[1].pcolormesh(aws_dataset_lon,aws_dataset_lat,
          np.squeeze(rad_refle_max), alpha=0.25,
          vmin=-999,
          vmax=200,
          cmap=plt.cm.Greys,
          transform=ccrs.PlateCarree())


    ax[1].coastlines(color='black',linewidth=2)
    ax[1].set_title('AWS 33-44 brightness temp. diff [K]',fontsize=16)
    fig.colorbar(a2,orientation='horizontal')


    a3=ax[2].pcolormesh(aws_dataset_lon,aws_dataset_lat,
          np.squeeze(rad_refle_max),
          vmin=-40,
          vmax=5,
          cmap=plt.cm.Blues,
          transform=ccrs.PlateCarree())


    ax[2].coastlines(color='black',linewidth=1)
    ax[2].set_title('EarthCare maximum radar reflectivity',fontsize=16)
    fig.colorbar(a3,orientation='horizontal')






def vertical_interpolator(ea_data,aws_id_time,ea_aws_dists,ea_ids,vertical_axis_step,product='CAP'):

    vert_axis=np.arange(0,16001,vertical_axis_step)
    print(vert_axis)

    if product=='CAP':
        print('Captivate')
        ea_CPR_act=ea_data.ScienceData.CPR_reflectivity_factor.values

        ea_height=ea_data.ScienceData.height.values


    if product=='CPR':
        print('Cloud profiling radar')
        ea_CPR_act=ea_data.ScienceData.reflectivity_masked.values
        #ea_CPR_act=np.nan_to_num(ea_CPR_act,nan=-9999) 
        ea_height=ea_data.ScienceData.height.values
        #ea_height=np.nan_to_num(ea_height,nan=-9999)
        #reflectivity_masked

    vertical_box=np.nan*np.zeros((int(aws_id_time.size*145),vert_axis.size))

    data_highres_2D=np.reshape(average_quantities_along_AWS_but_retain_vert_column(ea_CPR_act,
                                                                                   ea_ids,ea_aws_dists,product),
                                                                                   (int(aws_id_time.size*145),ea_height[0].size))
    height_highres_2D=np.reshape(average_quantities_along_AWS_but_retain_vert_column(ea_height,ea_ids,
                                                                                     ea_aws_dists,product),
                                                                                     (int(aws_id_time.size*145),ea_height[0].size))
    


    for i in tqdm.tqdm(np.arange(height_highres_2D[:,1].size)):

        for j in np.arange(vert_axis.size-1):

            sel_height=np.where((height_highres_2D[i,:] >= vert_axis[j]) & (height_highres_2D[i,:] < vert_axis[j+1]))[0]

            if sel_height.size == 0:
                vertical_box[i,j]=np.nan

            if sel_height.size > 0: 
                #compute mean in linear space, not dB space, beforehand transform each element
                un_dezi= 10**(1/10*data_highres_2D[i,sel_height])
                #print(un_dezi.shape)
                vertical_box[i,j]=np.nanmean(un_dezi,axis=0)

    vertical_box_DBZ=10*np.log10(vertical_box)

    return vert_axis, np.reshape(vertical_box_DBZ, (aws_id_time.size,145, vert_axis.size)) 


def convection_finder(coarse_refl_field,coarse_height_field,
                      lower_conv_bin, middle_conv_bin,high_conv_bin,
                      conv_rad_thres):
    print(coarse_height_field)
    low_ids=np.where((0 <= coarse_height_field) & (coarse_height_field  <lower_conv_bin))[0]
    mid_ids=np.where((lower_conv_bin <= coarse_height_field) & (coarse_height_field  < middle_conv_bin))[0]
    high_ids=np.where((middle_conv_bin <= coarse_height_field) & (coarse_height_field  < high_conv_bin))[0]
    print(low_ids)
    print(mid_ids)
    print(high_ids)

    rad_low_flag=np.where((coarse_refl_field[:,:,low_ids].flatten() >conv_rad_thres))[0]
    rad_mid_flag=np.where((coarse_refl_field[:,:,mid_ids].flatten() >conv_rad_thres))[0]
    rad_high_flag=np.where((coarse_refl_field[:,:,high_ids].flatten() >conv_rad_thres))[0]

    print(rad_low_flag.size)
    print(rad_mid_flag.size)
    print(rad_high_flag.size)




    if rad_low_flag.size > 0:
        low_conv=True
    else: 
        low_conv=False
    if rad_mid_flag.size > 0:
        mid_conv=True
    else: 
        mid_conv=False 
    if rad_high_flag.size > 0:
        high_conv=True
    else: 
        high_conv=False

    conv_flag= low_conv or mid_conv or high_conv
    
    if conv_flag==True:
        print('Convective Case')

    else:
        print('Neglible convection below dbZ threshold')
    print('low convection below '+ str(lower_conv_bin)+'m: '+str(low_conv))
    print('middle convection '  + str(lower_conv_bin)+'m - ' +str(middle_conv_bin) +'m: '+str(mid_conv))
    print('high convection above '  +str(middle_conv_bin) +'m: '+str(high_conv))



    

    return conv_flag,low_conv,mid_conv,high_conv


def colocation_fact_sheet(ea_data,ea_1_CPR,AWS_data,aws_idx,distance_matrix,aws_ea_remap,ea_time_id,
                          ea_time_long,aws_time_idx_clip,aws_swath_idx_,pred_time):
    #ea_clipped_index=[ea_time_long][]

    from sklearn.metrics import r2_score

    #transform pred time from str into datetime
    pred_time=pd.to_datetime(pred_time,format='%Y-%m-%d %H:%M:%S')
    print('Predicted colocation time: '+ str(pred_time))

    print(distance_matrix.shape)
    closest_distance_id=np.argmin(np.nanmin(np.nanmin(distance_matrix,2),1),0)
    print(closest_distance_id)
    times=ea_data.ScienceData.time[ea_time_long][ea_time_id][closest_distance_id].values
    print(times)
    print(aws_idx.shape)
    print(distance_matrix.shape)
    aws_swath_68_max_dist=np.max(np.nanmin(distance_matrix[:,68,:],0))
    cent=np.nanargmin(np.nanmax(distance_matrix[:,68,:],1),0)
    aws_colocat_68_time=AWS_data.time.values[np.nanargmin(np.nanmin(distance_matrix[:,68,:],1),0)]
    aws_colocat_68_time_ts=pd.to_datetime(aws_colocat_68_time,format='%Y-%m-%d %H:%M:%S')
    time_diff_seconds_68 = (aws_colocat_68_time_ts - pred_time).total_seconds()
    aws_colocat_68_lat=AWS_data.aws_lat.values[cent,68]
    aws_colocat_68_lon=AWS_data.aws_lon.values[cent,68]
    #print number of samples used for average at minimum distance
    print('aws 68 time difference: '+str(time_diff_seconds_68))
    print(~np.isnan(distance_matrix[cent,68,:]))
    aws_swath_68_number_avg= np.sum(~np.isnan(distance_matrix[cent,68,:]),0)




    print(aws_swath_68_max_dist)
    print(aws_colocat_68_time)
    print(aws_colocat_68_lat)
    print(aws_colocat_68_lon)




    cent=np.nanargmin(np.nanmax(distance_matrix[:,10,:],1),0)

    aws_swath_10_max_dist=np.max(np.nanmin(distance_matrix[:,10,:],0))
    print(aws_swath_10_max_dist)
    #print number of samples used for average at minimum distance
    aws_swath_10_number_avg= np.sum(~np.isnan(distance_matrix[cent,10,:]),0)

    aws_colocat_10_time=AWS_data.time[aws_idx].values[np.nanargmin(np.nanmin(distance_matrix[:,10,:],1),0)]
    print(aws_colocat_10_time)

    cent=np.nanargmin(np.nanmax(distance_matrix[:,135,:],1),0)

    aws_swath_135_max_dist=np.max(np.nanmin(distance_matrix[:,135,:],0))
    print(aws_swath_135_max_dist)
    #print number of samples used for average at minimum distance
    aws_swath_135_number_avg= np.sum(~np.isnan(distance_matrix[cent,135,:]),0)



    aws_colocat_135_time=AWS_data.time[aws_idx].values[np.nanargmin(np.nanmin(distance_matrix[:,135,:],1),0)]
    print(aws_colocat_135_time)


    ## print corrcoef AWS 21 vs ea_1_CPR

    AWS_21_flat=np.squeeze(colocate_cross_section_AWS_icewater_with_Earthcare(AWS_data,
                                                                              aws_time_idx_clip,aws_swath_idx_,
                                                            AWS_data.n_channels=='AWS21'))
    
    AWS_31_flat=np.squeeze(colocate_cross_section_AWS_icewater_with_Earthcare(AWS_data,
                                                                              aws_time_idx_clip,aws_swath_idx_,
                                                            AWS_data.n_channels=='AWS31'))
    
    AWS_33_flat=np.squeeze(colocate_cross_section_AWS_icewater_with_Earthcare(AWS_data,
                                                                              aws_time_idx_clip,aws_swath_idx_,
                                                            AWS_data.n_channels=='AWS33'))
    

    AWS_44_flat=np.squeeze(colocate_cross_section_AWS_icewater_with_Earthcare(AWS_data,
                                                                              aws_time_idx_clip,aws_swath_idx_,
                                                            AWS_data.n_channels=='AWS44'))
    

    



    print(AWS_21_flat.shape)
    print(ea_1_CPR.ScienceData.brightness_temperature.values[ea_time_id].shape)

    AWS_21_norm=(AWS_21_flat-np.mean(AWS_21_flat))/(np.max(AWS_21_flat)-np.min(AWS_21_flat))
    AWS_31_norm=(AWS_31_flat-np.mean(AWS_31_flat))/(np.max(AWS_31_flat)-np.min(AWS_31_flat))
    AWS_33_norm=(AWS_33_flat-np.mean(AWS_33_flat))/(np.max(AWS_33_flat)-np.min(AWS_33_flat))
    AWS_44_norm=(AWS_44_flat-np.mean(AWS_44_flat))/(np.max(AWS_44_flat)-np.min(AWS_44_flat))




    ea_CPR__=ea_1_CPR.ScienceData.brightness_temperature.values[ea_time_id]
    ea_CPR_norm=(ea_CPR__-np.mean(ea_CPR__))/(np.max(ea_CPR__)-np.min(ea_CPR__))

    corr_CPR_AWS_21= np.corrcoef(ea_1_CPR.ScienceData.brightness_temperature.values[ea_time_id],AWS_21_flat)[0,1]

    corr_CPR_land_AWS_21= np.corrcoef(ea_1_CPR.ScienceData.land_flag.values[ea_time_id],AWS_21_flat)[0,1]
    corr_CPR_land_AWS_31= np.corrcoef(ea_1_CPR.ScienceData.land_flag.values[ea_time_id],AWS_31_flat)[0,1]
    corr_CPR_land_AWS_33= np.corrcoef(ea_1_CPR.ScienceData.land_flag.values[ea_time_id],AWS_33_flat)[0,1]
    corr_CPR_land_AWS_44= np.corrcoef(ea_1_CPR.ScienceData.land_flag.values[ea_time_id],AWS_44_flat)[0,1]

    
    land_perc_EarthCare=100*np.mean(ea_1_CPR.ScienceData.land_flag.values[ea_time_id])


    corr_AWS21_AWS31=np.corrcoef(AWS_21_flat,AWS_31_flat)[0,1]
    corr_AWS21_AWS33=np.corrcoef(AWS_21_flat,AWS_33_flat)[0,1]
    corr_AWS21_AWS44=np.corrcoef(AWS_21_flat,AWS_44_flat)[0,1]
    corr_AWS33_AWS44=np.corrcoef(AWS_33_flat,AWS_44_flat)[0,1]
    corr_AWS31_AWS33=np.corrcoef(AWS_31_flat,AWS_33_flat)[0,1]
    corr_AWS31_AWS44=np.corrcoef(AWS_31_flat,AWS_44_flat)[0,1]

    
    coeff_deter_CPR_AWS_1=1-(np.mean((ea_CPR_norm-AWS_21_norm)**2)/np.var(ea_CPR_norm))

    r2_sklearn_CPR_AWS_1=r2_score(np.nan_to_num(ea_CPR_norm,nan=0),np.nan_to_num(AWS_21_norm,nan=0))

    print(corr_CPR_AWS_21)
    print(coeff_deter_CPR_AWS_1)
    print(r2_sklearn_CPR_AWS_1)


    vert_axis_1_2km,field_1_2km=vertical_interpolator(ea_data,aws_idx,
                           distance_matrix,aws_ea_remap,2000)
    
    conv,low_conv,mid_conv,high_conv=convection_finder(field_1_2km,vert_axis_1_2km,2000,6000,160001,-10)



    ##create summary fact sheet as pandas dataframe
    fact_sheet_dict={'Earthcare time':[times],
                     'col time AWS swID 68':[aws_colocat_68_time],
                     'col time AWS swID 10':[aws_colocat_10_time],
                     'col time AWS swID 135':[aws_colocat_135_time],
                     'col lat AWS swID 68 [°N]':[aws_colocat_68_lat],
                     'col lon AWS swID 68 [°E]':[aws_colocat_68_lon],
                     'time difference to predicted colocation [s]':[time_diff_seconds_68],
                     'maximum col distance AWS swID 68 [km]':[aws_swath_68_max_dist],
                     'EarthCare samples used for avg at col AWS swID 68':[aws_swath_68_number_avg],
                     'maximum col distance AWS swID 10 [km]':[aws_swath_10_max_dist],
                     'EarthCare samples used for avg at col AWS swID 10':[aws_swath_10_number_avg],
                     'maximum col distance AWS swID 135 [km]':[aws_swath_135_max_dist],
                     'EarthCare samples used for avg at col AWS swID 135':[aws_swath_135_number_avg],
                     'corr coeff CPR vs. AWS21 bright. temp':[corr_CPR_AWS_21],
                     'coeff determination CPR vs. AWS 21 bright. temp':[coeff_deter_CPR_AWS_1],
                     'R2 sklearn CPR vs. AWS 21 bright. temp':[r2_sklearn_CPR_AWS_1],
                     'conv_case':[conv],
                     'shallow_conv < 2km':[low_conv],
                     'middle_conv 2km < 6km':[mid_conv],
                     'high_conv > 6km':[high_conv],
                     'land percentage EarthCare [%]':[land_perc_EarthCare],
                     'corr coeff landflag vs. AWS21 bright. temp':[corr_CPR_land_AWS_21],
                     'corr coeff landflag vs. AWS31 bright. temp':[corr_CPR_land_AWS_31],
                     'corr coeff landflag vs. AWS33 bright. temp':[corr_CPR_land_AWS_33],
                     'corr coeff landflag vs. AWS44 bright. temp':[corr_CPR_land_AWS_44],
                     'corr coeff AWS21 vs. AWS31 bright. temp':[corr_AWS21_AWS31],
                     'corr coeff AWS21 vs. AWS33 bright. temp':[corr_AWS21_AWS33],
                     'corr coeff AWS21 vs. AWS44 bright. temp':[corr_AWS21_AWS44],
                     'corr coeff AWS33 vs. AWS44 bright. temp':[corr_AWS33_AWS44],
                     'corr coeff AWS31 vs. AWS33 bright. temp':[corr_AWS31_AWS33],
                     'corr coeff AWS31 vs. AWS44 bright. temp':[corr_AWS31_AWS44]}

    fact_sheet_df=pd.DataFrame(fact_sheet_dict)
    return fact_sheet_df


def full_scale_database_intemizer(ID,colocation_simulation_data_tle):

    #grep colocation,time,delay,distance,aws_lat,aws_lon,earthcare_lat,earthcare_lon,earthcare_altitude
    coloc,time,delay=handy_colocation_list_selector(ID,colocation_simulation_data_tle)
    n = 5
    sterni = plt.cm.gist_stern(np.linspace(0,1,n))
    #load AWS and Earthcare and EarthcareL2a data     
    ds=AWS_time_select(time)
    ea=earthcare_load(time, delay, coloc)  
    ea_CPR_L2a=earthcare_load(time,delay,coloc,product_type='/L2a/CPR_FMR_2A/')

    #check if earthcare data is none, if so return dataframe with only ID and Nans for all other values, to keep track of missing data in the final database
    if ea is None or ea_CPR_L2a is None:
        print('EarthCare data not available for this colocation with ID'+ str(ID)+' , returning empty dataframe')
        
        return pd.DataFrame({'Earthcare time':[str(np.nan)],
                     'col time AWS swID 68':[str(np.nan)],
                     'col time AWS swID 10':[str(np.nan)],
                     'col time AWS swID 135':[str(np.nan)],
                     'col lat AWS swID 68 [°N]':[str(np.nan)],
                     'col lon AWS swID 68 [°E]':[str(np.nan)],
                     'time difference to predicted colocation [s]':[str(np.nan)],
                     'maximum col distance AWS swID 68 [km]':[str(np.nan)],
                     'EarthCare samples used for avg at col AWS swID 68':[str(np.nan)],
                     'maximum col distance AWS swID 10 [km]':[str(np.nan)],
                     'EarthCare samples used for avg at col AWS swID 10':[str(np.nan)],
                     'maximum col distance AWS swID 135 [km]':[str(np.nan)],
                     'EarthCare samples used for avg at col AWS swID 135':[str(np.nan)],
                     'corr coeff CPR vs. AWS21 bright. temp':[str(np.nan)],
                     'coeff determination CPR vs. AWS 21 bright. temp':[str(np.nan)],
                     'R2 sklearn CPR vs. AWS 21 bright. temp':[str(np.nan)],
                     'conv_case':[str(np.nan)],
                     'shallow_conv < 2km':[str(np.nan)],
                     'middle_conv 2km < 6km':[str(np.nan)],
                     'high_conv > 6km':[str(np.nan)],
                     'land percentage EarthCare [%]':[str(np.nan)],
                     'corr coeff landflag vs. AWS21 bright. temp':[str(np.nan)],
                     'corr coeff landflag vs. AWS31 bright. temp':[str(np.nan)],
                     'corr coeff landflag vs. AWS33 bright. temp':[str(np.nan)],
                     'corr coeff landflag vs. AWS44 bright. temp':[str(np.nan)],
                     'corr coeff AWS21 vs. AWS31 bright. temp':[str(np.nan)],
                     'corr coeff AWS21 vs. AWS33 bright. temp':[str(np.nan)],
                     'corr coeff AWS21 vs. AWS44 bright. temp':[str(np.nan)],
                     'corr coeff AWS33 vs. AWS44 bright. temp':[str(np.nan)],
                     'corr coeff AWS31 vs. AWS33 bright. temp':[str(np.nan)],
                     'corr coeff AWS31 vs. AWS44 bright. temp':[str(np.nan)]})


    #get Eartcare along AWS path
    dist_aws_earth_remap,aws_earth_indices_remap,aws_idx,aws_times=get_along_AWS_path_closest_earthcare_points_faster(ds,ea,time,delay,number_of_nearest_EC_samples=10)
    #get AWS nearest data for each Earthcare point
    dist_map,aws_swath_indices_map,aws_time_indices_map=select_nearest_data_samples_faster(ds,ea,time,delay)
    #get cross section indices
    ea_time_idx,ea_clip_20=plot_distance_matrix_selected_cross_section(dist_map,
                                                                 aws_swath_indices_map,
                                                                 aws_time_indices_map,
                                                                 ea,ds,coloc,time,delay,plot_figures=False)
    #create fact sheet
    coll_data_frame=colocation_fact_sheet(ea,ea_CPR_L2a,ds,
                                          aws_idx,dist_aws_earth_remap,
                                          aws_earth_indices_remap,
                                          ea_clip_20,ea_time_idx,
                                          aws_time_indices_map[ea_clip_20],
                                          aws_swath_indices_map[ea_clip_20],time)
    





    return coll_data_frame
