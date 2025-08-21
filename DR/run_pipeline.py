#!/usr/bin/env python3
"""
run_pipeline.py

Usage:
    python run_pipeline.py --infile data/Serial_ACC_GNSS_data.txt --outdir outputs

What it does (default mode):
 - Parses serial text for IMU lines (ACCL: ICM20600 ...) and GPS lines (GPS_SERVICE lat/long OR NMEA GNGGA/GNRMC)
 - Builds imu_df and gps_df (timestamps are taken from the line prefix)
 - Merges IMU -> GPS using `merge_asof` (IMU as master timeline)
 - Converts GPS lat/lon -> local meters (ENU, using pyproj if available, else equirectangular)
 - Runs a Constant-Velocity Kalman Filter (state x=[x,y,vx,vy]) that:
     * Predicts each sample with dt (variable)
     * Optionally accepts accel control if orientation is estimated (requires `ahrs`)
     * Updates when GPS measurement exists (predict-only when GPS missing)
 - Computes error metrics (only at timestamps where raw GPS existed)
 - Saves merged CSV and plots to outdir
"""

import re, os, argparse, sys, math
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# try optional libs
try:
    from ahrs.filters import Madgwick
    AHRS_AVAILABLE = True
except Exception:
    AHRS_AVAILABLE = False

try:
    from pyproj import Transformer
    PYPROJ_AVAILABLE = True
except Exception:
    PYPROJ_AVAILABLE = False

# -------------------------
# Regex patterns
# -------------------------
IMU_RE = re.compile(
    r'(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s+ACCL: ICM20600 '
    r'ACC_X=(?P<ax>-?\d+)\s+ACC_Y=(?P<ay>-?\d+)\s+ACC_Z=(?P<az>-?\d+)\s+'
    r'GYRO_X=(?P<gx>-?\d+)\s+GYRO_Y=(?P<gy>-?\d+)\s+GYRO_Z=(?P<gz>-?\d+)'
)

GPS_SERVICE_RE = re.compile(
    r'(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).*GPS_SERVICE:.*lat:(?P<lat>[-+]?\d+\.\d+)\s+long:(?P<lon>[-+]?\d+\.\d+)'
)

# NMEA lines detection (we'll parse GNGGA/GNRMC)
NMEA_RE = re.compile(r'(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+).*\$(?P<msg>[A-Z]{5}),(.+)')

# helper to convert NMEA ddmm.mmmm -> decimal degrees
def nmea_to_deg(nmea_str, hemi):
    # accept strings like 1258.96778 or 07732.92713
    if nmea_str == '' or nmea_str is None:
        return None
    v = float(nmea_str)
    deg = int(v // 100)
    minutes = v - deg*100
    dec = deg + minutes / 60.0
    if hemi.upper() in ('S', 'W'):
        dec = -dec
    return dec

# -------------------------
# Parsing function
# -------------------------
def parse_serial_file(path):
    imu_rows = []
    gps_rows = []
    with open(path, 'r', errors='ignore') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            # IMU match
            m = IMU_RE.search(line)
            if m:
                d = m.groupdict()
                imu_rows.append({
                    'ts': pd.to_datetime(d['ts']),
                    'ax_raw': int(d['ax']),
                    'ay_raw': int(d['ay']),
                    'az_raw': int(d['az']),
                    'gx_raw': int(d['gx']),
                    'gy_raw': int(d['gy']),
                    'gz_raw': int(d['gz']),
                    'raw_line': line
                })
                continue
            # GPS service lines with decimal lat/lon
            g = GPS_SERVICE_RE.search(line)
            if g:
                d = g.groupdict()
                gps_rows.append({
                    'ts': pd.to_datetime(d['ts']),
                    'lat': float(d['lat']),
                    'lon': float(d['lon']),
                    'source': 'GPS_SERVICE',
                    'raw_line': line
                })
                continue
            # NMEA catch (GNGGA / GNRMC)
            n = NMEA_RE.search(line)
            if n:
                ts = pd.to_datetime(n.group('ts'))
                fields = line.split(',')
                msg = n.group('msg')
                # handle GNGGA: fields[2]=lat, [3]=N/S, [4]=lon, [5]=E/W
                try:
                    if msg.endswith('GGA') and len(fields) > 5:
                        lat_n = fields[2]
                        ns = fields[3]
                        lon_n = fields[4]
                        ew = fields[5]
                        lat = nmea_to_deg(lat_n, ns)
                        lon = nmea_to_deg(lon_n, ew)
                        if lat is not None and lon is not None:
                            gps_rows.append({
                                'ts': ts,
                                'lat': lat,
                                'lon': lon,
                                'source': 'GNGGA',
                                'raw_line': line
                            })
                    # handle RMC: fields[3]=lat, [4]=N/S, [5]=lon, [6]=E/W (depending on format)
                    elif msg.endswith('RMC') and len(fields) > 6:
                        lat_n = fields[3]
                        ns = fields[4]
                        lon_n = fields[5]
                        ew = fields[6]
                        lat = nmea_to_deg(lat_n, ns)
                        lon = nmea_to_deg(lon_n, ew)
                        if lat is not None and lon is not None:
                            gps_rows.append({
                                'ts': ts,
                                'lat': lat,
                                'lon': lon,
                                'source': 'GNRMC',
                                'raw_line': line
                            })
                except Exception:
                    # ignore malformed NMEA but continue
                    pass
                continue
    imu_df = pd.DataFrame(imu_rows).sort_values('ts').reset_index(drop=True)
    gps_df = pd.DataFrame(gps_rows).sort_values('ts').reset_index(drop=True)
    return imu_df, gps_df

# -------------------------
# Merge and convert coords
# -------------------------
def merge_and_convert(imu_df, gps_df, use_pyproj=True):
    # merge_asof: imu as master timeline (we assume IMU high-rate)
    imu_df = imu_df.sort_values('ts').reset_index(drop=True)
    gps_df = gps_df.sort_values('ts').reset_index(drop=True)
    # if gps_df empty -> create empty columns only
    if gps_df.shape[0] == 0:
        imu_df['lat'] = np.nan
        imu_df['lon'] = np.nan
        merged = imu_df.copy()
        merged['e_gps'] = np.nan
        merged['n_gps'] = np.nan
        return merged
    
    merged = pd.merge_asof(imu_df, gps_df[['ts','lat','lon']], on='ts', direction='backward', tolerance=pd.Timedelta('500ms'))
    # Note: some IMU rows before first GPS will have NaN lat/lon; it's fine.
    # detect gps reference (first valid)
    valid_idx = merged['lat'].first_valid_index()
    if valid_idx is None:
        raise RuntimeError("No GPS lat/lon parsed from file.")
    lat0 = merged.loc[valid_idx, 'lat']
    lon0 = merged.loc[valid_idx, 'lon']
    
    # Use pyproj ECEF->ENU if available; else use simple equirectangular approx.
    if PYPROJ_AVAILABLE and use_pyproj:
        try:
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
            def ll_to_ecef(lon, lat, alt=0.0):
                return transformer.transform(lon, lat, alt)
            # compute reference ecef
            x0,y0,z0 = ll_to_ecef(lon0, lat0)
            # rotation matrix from ECEF->ENU
            phi = math.radians(lat0)
            lam = math.radians(lon0)
            R = np.array([
                [-math.sin(lam),  math.cos(lam), 0],
                [-math.sin(phi)*math.cos(lam), -math.sin(phi)*math.sin(lam), math.cos(phi)],
                [ math.cos(phi)*math.cos(lam),  math.cos(phi)*math.sin(lam), math.sin(phi)]
            ])
            es = []
            ns = []
            for (lat,lon) in zip(merged['lat'].values, merged['lon'].values):
                if np.isfinite(lat) and np.isfinite(lon):
                    x,y,z = ll_to_ecef(lon, lat, 0.0)
                    d = np.array([x-x0, y-y0, z-z0])
                    enu = R.dot(d)
                    es.append(float(enu[0])); ns.append(float(enu[1]))
                else:
                    es.append(np.nan); ns.append(np.nan)
            merged['e_gps'] = es
            merged['n_gps'] = ns
        except Exception as e:
            print("pyproj ENU failed, falling back to equirectangular:", e)
            PYPROJ_AVAILABLE = False
    
    if (not PYPROJ_AVAILABLE) or (not use_pyproj):
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * math.cos(math.radians(lat0))
        merged['e_gps'] = (merged['lon'] - lon0) * meters_per_deg_lon
        merged['n_gps'] = (merged['lat'] - lat0) * meters_per_deg_lat
    
    return merged

# -------------------------
# Optional: auto-calibrate accelerometer scale
# -------------------------
def estimate_accel_scale(ax_raw, ay_raw, az_raw):
    """
    If user doesn't provide scale, try to estimate scale such that
    median magnitude ~= 9.81 m/s^2 during the file.
    Works only if there are large stationary periods.
    """
    mag_raw = np.sqrt(ax_raw.astype(float)**2 + ay_raw.astype(float)**2 + az_raw.astype(float)**2)
    med = np.nanmedian(mag_raw)
    if med == 0:
        return None
    scale = 9.81 / med
    return scale

# -------------------------
# KF: CV model with optional accel control
# -------------------------
def run_cv_kf(times, e_gps, n_gps, ax_enu=None, ay_enu=None, gps_std=3.0, accel_noise_std=0.5):
    """
    times: pandas Series of timestamps
    e_gps, n_gps: arrays (meters) with NaN where GPS missing
    ax_enu, ay_enu: optional accelerations in ENU (m/s^2) to use as control (same length)
    Returns: states (N,4) columns [x,y,vx,vy] and covariances (N,4,4)
    """
    N = len(times)
    states = np.zeros((N,4))
    covs = np.zeros((N,4,4))
    # initial state at first valid GPS
    gps_mask = np.isfinite(e_gps) & np.isfinite(n_gps)
    first = np.where(gps_mask)[0]
    if len(first)==0:
        # No GPS at all -> start at zeros
        x = np.zeros(4)
    else:
        f = first[0]
        x = np.array([e_gps[f], n_gps[f], 0.0, 0.0])
    P = np.diag([100.0, 100.0, 10.0, 10.0])  # initial uncertainty (tune)
    R = np.diag([gps_std**2, gps_std**2])
    q = accel_noise_std**2
    
    for k in range(N):
        if k == 0:
            dt = 0.0
        else:
            dt = (times.iloc[k] - times.iloc[k-1]).total_seconds()
            if dt <= 0 or dt > 5.0:
                dt = 1.0 / 50.0  # fallback if weird time gaps
        
        # state transition
        F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        # control matrix B
        B = np.array([[0.5*dt*dt, 0],[0, 0.5*dt*dt],[dt,0],[0,dt]])
        # process noise Q from accel noise
        Q = np.array([
            [dt**4/4*q, 0, dt**3/2*q, 0],
            [0, dt**4/4*q, 0, dt**3/2*q],
            [dt**3/2*q, 0, dt**2*q, 0],
            [0, dt**3/2*q, 0, dt**2*q]
        ])
        # predict
        if ax_enu is not None and ay_enu is not None:
            u = np.array([ax_enu[k], ay_enu[k]])
            x = F.dot(x) + B.dot(u)
        else:
            x = F.dot(x)
        P = F.dot(P).dot(F.T) + Q
        # update if GPS available
        if np.isfinite(e_gps[k]) and np.isfinite(n_gps[k]):
            z = np.array([e_gps[k], n_gps[k]])
            H = np.array([[1,0,0,0],[0,1,0,0]])
            y = z - H.dot(x)
            S = H.dot(P).dot(H.T) + R
            K = P.dot(H.T).dot(np.linalg.inv(S))
            x = x + K.dot(y)
            P = (np.eye(4) - K.dot(H)).dot(P)
        states[k] = x
        covs[k] = P
    return states, covs

# -------------------------
# Error calculation and plotting
# -------------------------
def evaluate_and_plot(merged, states, outdir, prefix='results', lat0=None, lon0=None):
    os.makedirs(outdir, exist_ok=True)
    merged['est_e'] = states[:,0]
    merged['est_n'] = states[:,1]
    
    gps_mask = merged['e_gps'].notna() & merged['n_gps'].notna()
    dx = merged.loc[gps_mask, 'e_gps'].values - merged.loc[gps_mask, 'est_e'].values
    dy = merged.loc[gps_mask, 'n_gps'].values - merged.loc[gps_mask, 'est_n'].values
    errors = np.sqrt(dx*dx + dy*dy)
    
    rmse = np.sqrt(np.nanmean(errors**2))
    median = np.nanmedian(errors)
    max_err = np.nanmax(errors)
    pct1 = np.mean(errors<=1.0)*100.0
    pct3 = np.mean(errors<=3.0)*100.0
    pct5 = np.mean(errors<=5.0)*100.0
    
    print("KF Evaluation:")
    print(f"  samples: {len(merged)}, gps available: {gps_mask.sum()}, missing: {len(merged)-gps_mask.sum()}")
    print(f"  RMSE = {rmse:.3f} m, median = {median:.3f} m, max = {max_err:.3f} m")
    print(f"  % within 1m = {pct1:.2f}%, within 3m = {pct3:.2f}%, within 5m = {pct5:.2f}%")
    
    # save merged csv
    csv_path = os.path.join(outdir, f"{prefix}_merged_with_estimates.csv")
    merged.to_csv(csv_path, index=False)
    print("Saved merged CSV ->", csv_path)
    
    # plots
    plt.figure(figsize=(8,6))
    plt.plot(merged['e_gps'], merged['n_gps'], label='raw GPS track', linewidth=1)
    plt.plot(merged['est_e'], merged['est_n'], label='KF estimated track', linewidth=1)
    plt.scatter(merged.loc[~gps_mask,'est_e'], merged.loc[~gps_mask,'est_n'], c='k', s=10, label='est (no GPS)')
    plt.xlabel('East (m)'); plt.ylabel('North (m)')
    plt.axis('equal'); plt.grid(True); plt.legend()
    p1 = os.path.join(outdir, f"{prefix}_track.png"); plt.savefig(p1, dpi=200); plt.close()
    
    plt.figure(figsize=(10,3))
    plt.plot(merged.loc[gps_mask, 'ts'], errors)
    plt.xlabel('time'); plt.ylabel('position error (m)'); plt.grid(True)
    p2 = os.path.join(outdir, f"{prefix}_error_time.png"); plt.savefig(p2, dpi=150); plt.close()
    
    plt.figure(figsize=(5,4))
    plt.hist(errors, bins=50)
    plt.xlabel('error (m)'); plt.ylabel('count'); plt.grid(True)
    p3 = os.path.join(outdir, f"{prefix}_error_hist.png"); plt.savefig(p3, dpi=150); plt.close()
    
    print("Saved plots ->", p1, p2, p3)
    return {'rmse': rmse, 'median': median, 'max': max_err, 'pct1': pct1, 'pct3': pct3, 'pct5': pct5}

# -------------------------
# Main CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', required=True, help='path to serial log file')
    parser.add_argument('--outdir', default='outputs', help='folder to save results')
    parser.add_argument('--gps_std', type=float, default=3.0, help='GPS measurement std (m)')
    parser.add_argument('--accel_noise', type=float, default=0.5, help='process accel noise std (m/s^2)')
    parser.add_argument('--use_ahrs', action='store_true', help='use AHRS (Madgwick) to rotate acc into ENU (requires ahrs)')
    parser.add_argument('--acc_scale', type=float, default=None, help='accelerometer scale (counts -> m/s^2); if omitted script will attempt to estimate')
    args = parser.parse_args()
    
    infile = args.infile
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    
    print("Parsing serial file:", infile)
    imu_df, gps_df = parse_serial_file(infile)
    print("Parsed IMU rows:", len(imu_df), "Parsed GPS rows:", len(gps_df))
    
    # quick sanity print of first few GPS values
    if len(gps_df)>0:
        print("GPS examples (first 10):")
        print(gps_df[['ts','lat','lon','source']].head(10).to_string(index=False))
    
    merged = merge_and_convert(imu_df, gps_df)
    print("Merged timeline length:", len(merged))
    
    # scale accel if needed
    if args.acc_scale is None:
        try:
            scale = estimate_accel_scale(merged['ax_raw'].values, merged['ay_raw'].values, merged['az_raw'].values)
            print("Estimated accel scale:", scale)
        except Exception:
            scale = None
    else:
        scale = args.acc_scale
    if scale is None:
        print("No accel scale provided or estimated => accelerations will not be used as KF control.")
    
    # prepare accel/gyro arrays scaled if available
    if scale is not None:
        merged['ax'] = merged['ax_raw'].astype(float) * scale
        merged['ay'] = merged['ay_raw'].astype(float) * scale
        merged['az'] = merged['az_raw'].astype(float) * scale
    else:
        merged['ax'] = np.nan; merged['ay'] = np.nan; merged['az'] = np.nan
    
    # if user asked for AHRS and library present, estimate orientation and rotate acc to ENU
    ax_enu = None; ay_enu = None
    if args.use_ahrs:
        if not AHRS_AVAILABLE:
            print("AHRS (madgwick) requested but 'ahrs' lib not installed. Skipping orientation.")
        elif scale is None:
            print("AHRS requested but no accel scale. Skipping orientation.")
        else:
            print("Running Madgwick AHRS to get orientation and rotate body->ENU (this may take time)...")
            fs_est = 1.0 / np.nanmedian(np.diff(merged['ts'].astype('int64'))/1e9)
            sampleperiod = 1.0/fs_est
            madg = Madgwick(sampleperiod=sampleperiod)
            # prepare arrays
            gx = merged['gx_raw'].astype(float).values  # NOTE: raw gyro likely in counts -> requires scale to rad/s. We omit gyro scaling here.
            gy = merged['gy_raw'].astype(float).values
            gz = merged['gz_raw'].astype(float).values
            ax = merged['ax'].astype(float).values
            ay = merged['ay'].astype(float).values
            az = merged['az'].astype(float).values
            # NOTE: full correct use requires gyro scale (deg/s->rad/s). Here we'll run Madgwick with raw gyros which is approximate.
            # If you know gyro_scale, multiply gx,gy,gz by it first (in rad/s) for correctness.
            q = np.zeros((len(merged),4))
            q_prev = np.array([1.0,0.0,0.0,0.0])
            from ahrs.common import quat2mat
            for i in range(len(merged)):
                try:
                    q_prev = madg.updateIMU(q_prev, gyr=np.array([gx[i],gy[i],gz[i]]), acc=np.array([ax[i],ay[i],az[i]]))
                except Exception:
                    # fallback - keep previous
                    pass
                q[i,:] = q_prev
            # rotate body accel to global using quat2mat
            acc_enu = np.zeros((len(merged),3))
            for i in range(len(merged)):
                C = quat2mat(q[i,:])  # rotation matrix body->nav (if using ahrs implementation)
                acc_body = np.array([ax[i], ay[i], az[i]])
                acc_nav = C.dot(acc_body)
                # subtract gravity (z axis ~ +9.81 in nav frame depending on conventions) -- adjust sign if needed
                acc_nav[2] = acc_nav[2] - 9.81
                acc_enu[i,:] = acc_nav
            ax_enu = acc_enu[:,0]; ay_enu = acc_enu[:,1]
            print("AHRS produced ENU accelerations (ax_enu,ay_enu).")
    
    # run KF
    print("Running CV-KF (gps_std=", args.gps_std, ", accel_noise=", args.accel_noise, ") ...")
    states, covs = run_cv_kf(merged['ts'], merged['e_gps'].values, merged['n_gps'].values,
                             ax_enu if ax_enu is not None else None,
                             ay_enu if ay_enu is not None else None,
                             gps_std=args.gps_std, accel_noise_std=args.accel_noise)
    
    # evaluate and save
    metrics = evaluate_and_plot(merged, states, args.outdir, prefix='kf_results')
    print("Done. Outputs in:", args.outdir)

if __name__ == '__main__':
    main()
