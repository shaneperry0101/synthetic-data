"""
Helper functions for synthetic data generation
"""

import numpy as np


def make_daily_profile(dt_index, wake_hour=7, school_start=8, school_end=15, sleep_hour=21):
    # produces an activity factor [0..1] by hour
    hours = np.array([ts.hour + ts.minute/60.0 for ts in dt_index])
    profile = np.zeros_like(hours)
    # awake baseline
    profile += ((hours >= wake_hour) & (hours < sleep_hour)).astype(float) * 0.5
    # active periods
    profile += ((hours >= wake_hour+0.5) & (hours < school_start)).astype(float) * 0.5  # morning
    profile += ((hours >= school_end) & (hours < school_end+2)).astype(float) * 0.7     # after-school play
    profile += ((hours >= 18) & (hours < 20)).astype(float) * 0.6                       # evening play
    # sleep suppression
    profile *= (1.0 - ((hours >= sleep_hour) | (hours < wake_hour)).astype(float) * 0.9)
    # slight randomness
    profile += np.random.normal(0, 0.05, size=profile.shape)
    return np.clip(profile, 0.0, 1.0)


def inject_forced_removal(df, start_idx, duration_seconds=30):
    # sudden temp drop, high jerk spike, then off-wrist (contact_index=0) for a bit
    end_idx = min(start_idx + int(duration_seconds), len(df)-1)
    # jerk spike: multiply accel by factor for 1s window
    spike_len = int(1)
    idx0 = max(start_idx, 0)
    idx_spike_end = min(idx0 + spike_len, len(df)-1)
    df.loc[idx0:idx_spike_end, ['accel_x', 'accel_y', 'accel_z']] *= np.random.uniform(4.0, 8.0)
    # temp drop: subtract 2-3 deg over 10s
    temp_drop = np.linspace(0, np.random.uniform(2.0, 3.0), 10)
    for i, d in enumerate(temp_drop):
        ii = idx0 + i
        if ii < len(df):
            df.at[ii, 'body_temp_c'] -= d
    # set contact_index low/off
    df.loc[idx0:end_idx, 'contact_index'] = 0.0
    # reduce motion after removal
    df.loc[idx0:end_idx, ['accel_x', 'accel_y', 'accel_z']] *= 0.1
    return idx0, end_idx


def inject_fever(df, start_idx, duration_seconds=20*60):
    # gradual body temp rise over time window
    n = min(len(df)-start_idx, int(duration_seconds))
    slope = np.linspace(0, np.random.uniform(
        1.0, 2.0), n)  # degrees distributed
    for i, s in enumerate(slope):
        df.at[start_idx+i, 'body_temp_c'] += s
    # heart rate elevated
    df.loc[start_idx:start_idx+n-1, 'heart_rate'] += np.random.uniform(8, 20)
    return start_idx, start_idx+n-1


def inject_water_submersion(df, start_idx, duration_seconds=80):
    # water toggles on, temp drop over 60s, reduced motion (submersion)
    n = min(len(df)-start_idx, int(duration_seconds))
    # set water_binary = 1 at 1Hz resolution - we have 10Hz main timeline, so set every 10 samples
    for i in range(n):
        df.at[start_idx+i, 'water_binary'] = 1
    # temp drop
    for i in range(10):
        ii = start_idx + i
        if ii < len(df):
            # small per-sample drop accumulating
            df.at[ii, 'body_temp_c'] -= np.random.uniform(0.05, 0.12)
    # motion suppressed
    df.loc[start_idx:start_idx+n-1, ['accel_x', 'accel_y', 'accel_z']] *= 0.2
    return start_idx, start_idx+n-1


def inject_stillness(df, start_idx, duration_seconds=200):
    # stillness: low step rate and low accel for minutes
    n = min(len(df)-start_idx, int(duration_seconds))
    df.loc[start_idx:start_idx+n-1, ['accel_x', 'accel_y', 'accel_z']] *= 0.05
    df.loc[start_idx:start_idx+n-1, 'step'] = 0
    # heart rate drop slightly
    df.loc[start_idx:start_idx+n-1, 'heart_rate'] -= np.random.uniform(2, 6)
    return start_idx, start_idx+n-1
