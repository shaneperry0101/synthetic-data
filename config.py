"""
Configurations for synthetic data generation
"""

from datetime import datetime

OUTPUT_DIR = "synthetic_dataset"

NUM_DEVICES = 6                       # 5-8 devices recommended
START = datetime(2025, 1, 1, 0, 0, 0)  # arbitrary start
DAYS = 14
ACCEL_HZ = 10
HR_HZ = 1
TEMP_HZ = 0.2
WATER_HZ = 1

# Event frequencies (per doc)
FORCED_REMOVALS_PER_CHILD = 1  # between 1-2 over 2 weeks
FEVER_EVENTS_PER_CHILD = 0.5   # 0-1
SUBMERSION_EVENTS_TOTAL = 1    # dataset-wide rare
STILLNESS_EVENTS_PER_CHILD = 2
NORMAL_REMOVALS_OUTSIDE_WINDOW = 3
