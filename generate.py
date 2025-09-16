"""
Produces 2-week synthetic multi-sensor timeseries per device with labeled anomalies.
"""

import numpy as np
import pandas as pd
import os
from datetime import timedelta
from helper import make_daily_profile, inject_forced_removal, inject_fever, inject_water_submersion, inject_stillness
from config import OUTPUT_DIR, NUM_DEVICES, START, DAYS, FORCED_REMOVALS_PER_CHILD, FEVER_EVENTS_PER_CHILD, STILLNESS_EVENTS_PER_CHILD, NORMAL_REMOVALS_OUTSIDE_WINDOW, SUBMERSION_EVENTS_TOTAL


np.random.seed(42)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate per-device
events = []
for dev_i in range(NUM_DEVICES):
    device_id = f"device_{dev_i+1}"
    start_ts = START
    end_ts = START + timedelta(days=DAYS)
    # build main timeline at 1Hz resolution
    total_seconds = int((end_ts - start_ts).total_seconds())
    n_samples = total_seconds
    # create timestamps list as DatetimeIndex
    timestamps = pd.date_range(
        start_ts, periods=n_samples, freq=f"1000ms")  # ms freq
    df = pd.DataFrame({"timestamp": timestamps})
    # daily activity profile
    profile = make_daily_profile(df['timestamp'])
    # base accel noise (device frame)
    accel_base = np.random.normal(0, 0.05, size=(n_samples, 3))
    # scale accel by activity profile (simulate steps/jerks)
    # per-sample multiplier
    movement_mag = profile * np.random.uniform(0.2, 1.6)
    df['accel_x'] = accel_base[:, 0] + movement_mag * \
        np.random.normal(0.0, 0.8, size=n_samples)
    df['accel_y'] = accel_base[:, 1] + movement_mag * \
        np.random.normal(0.0, 0.8, size=n_samples)
    df['accel_z'] = accel_base[:, 2] + movement_mag * \
        np.random.normal(0.0, 0.8, size=n_samples)
    # step estimate: aggregate per second - simple proxy from movement magnitude
    mag = np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2)
    # step: probabilistic based on profile and momentary mag
    step_prob = np.clip(profile * (0.01 + mag/5.0), 0.0, 1.0)
    df['step'] = (np.random.rand(n_samples) < step_prob).astype(int)
    # heart rate baseline per child (age-dependent)
    age = np.random.randint(6, 13)
    # younger = higher baseline
    hr_rest = int(np.random.normal(75 - (age-6)*1.5, 5))
    # diurnal HR modulation
    hours = np.array([ts.hour + ts.minute/60.0 for ts in df['timestamp']])
    hr_mod = 5 * np.sin((hours - 6)/24 * 2*np.pi)  # slight circadian
    df['heart_rate'] = hr_rest + hr_mod + \
        (profile * np.random.uniform(5, 20))*1.2
    # temperature: ambient + body
    ambient_base = 20 + 3 * np.sin((hours - 6)/24 * 2*np.pi)  # 17..23 typical
    # small circadian shift
    body_base = 36.4 + 0.3 * np.sin((hours - 2)/24 * 2*np.pi)
    # body temp noise and activity bump
    df['ambient_temp_c'] = ambient_base + \
        np.random.normal(0, 0.2, size=n_samples)
    df['body_temp_c'] = body_base + \
        (profile * np.random.uniform(0.0, 0.4)) + \
        np.random.normal(0, 0.08, size=n_samples)
    # contact index = body - ambient (positive when on-wrist)
    df['contact_index'] = np.clip(
        df['body_temp_c'] - df['ambient_temp_c'], 0.0, 5.0)
    # water sensor default 0
    df['water_binary'] = 0
    # charging flag rarely true (simulate nightly charging)
    df['charging'] = 0
    # Add some device dropouts: random short windows where readings missing
    for _ in range(3):
        drop_start = np.random.randint(0, n_samples - 60*5)
        drop_len = np.random.randint(5, 60)
        df.loc[drop_start:drop_start+drop_len, ['heart_rate']
               ] = df.loc[drop_start:drop_start+drop_len, ['heart_rate']].ffill().bfill()

    # inject anomalies per frequencies
    # forced removals
    for _ in range(int(np.random.poisson(FORCED_REMOVALS_PER_CHILD))):
        idx = np.random.randint(int(0.1*n_samples), int(0.9*n_samples))
        s, e = inject_forced_removal(
            df, idx, duration_seconds=np.random.randint(10, 60))
        events.append({"device_id": device_id, "event": "forced_removal",
                      "start_idx": s, "end_idx": e, "start_ts": df.at[s, 'timestamp']})
    # fevers
    if np.random.rand() < FEVER_EVENTS_PER_CHILD:
        idx = np.random.randint(int(0.1*n_samples), int(0.9*n_samples))
        s, e = inject_fever(df, idx, duration_seconds=np.random.randint(
            10*60, 60*60))  # 10min-60min
        events.append({"device_id": device_id, "event": "fever",
                      "start_idx": s, "end_idx": e, "start_ts": df.at[s, 'timestamp']})
    # stillness events
    for _ in range(int(np.random.poisson(STILLNESS_EVENTS_PER_CHILD))):
        idx = np.random.randint(int(0.05*n_samples), int(0.95*n_samples))
        s, e = inject_stillness(
            df, idx, duration_seconds=np.random.randint(120, 400))
        events.append({"device_id": device_id, "event": "stillness",
                      "start_idx": s, "end_idx": e, "start_ts": df.at[s, 'timestamp']})
    # normal removals outside window (info-level)
    for _ in range(NORMAL_REMOVALS_OUTSIDE_WINDOW):
        idx = np.random.randint(int(0.05*n_samples), int(0.95*n_samples))
        s, e = inject_forced_removal(
            df, idx, duration_seconds=np.random.randint(5, 20))
        events.append({"device_id": device_id, "event": "normal_removal",
                      "start_idx": s, "end_idx": e, "start_ts": df.at[s, 'timestamp']})
    
    # write per-device CSV (downsample heavy columns to manageable size if needed)
    out_file = os.path.join(OUTPUT_DIR, f"{device_id}.parquet")
    df.to_parquet(out_file, index=False)
    print("Wrote", out_file)

# inject dataset-wide submersion rare event
if SUBMERSION_EVENTS_TOTAL > 0:
    # pick random device and time
    dev_pick = f"device_{np.random.randint(1, NUM_DEVICES+1)}"
    # load device file, inject, rewrite
    fpath = os.path.join(OUTPUT_DIR, f"{dev_pick}.parquet")
    df = pd.read_parquet(fpath)
    idx = np.random.randint(int(0.2*len(df)), int(0.8*len(df)))
    s, e = inject_water_submersion(df, idx, duration_seconds=90)
    events.append({"device_id": dev_pick, "event": "submersion",
                  "start_idx": s, "end_idx": e, "start_ts": df.at[s, 'timestamp']})
    df.to_parquet(fpath, index=False)
    print("Injected submersion into", fpath)

# write events.csv
events_df = pd.DataFrame(events)
events_df.to_csv(os.path.join(OUTPUT_DIR, "events.csv"), index=False)
print("Wrote events.csv with", len(events_df), "events")
