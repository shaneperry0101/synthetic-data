# Synthetic Bracelet Dataset Generator

**Disclaimer**

This dataset is entirely synthetic and generated for research and testing purposes only. While the code attempts to mimic realistic patterns of wearable device signals, **there is no guarantee that the statistical distribution, variability, or anomalies reflect real-world device data**. Any conclusions drawn from this dataset should not be assumed to generalize to actual populations or hardware.


Creates a **two-week synthetic multi-sensor time-series dataset** simulating wearable bracelet devices for children. It is useful for testing algorithms in anomaly detection, health monitoring, and wearable signal analysis.

## What It Does

* Simulates data for **multiple devices** (5-8).
* Produces a **continuous timeline** at 1Hz resolution, covering 14 days.
* Generates realistic physiological and environmental signals, including:

  * **Accelerometer (x, y, z)** — movement intensity tied to daily activity routines.
  * **Step counts** — derived from motion magnitude.
  * **Heart rate (bpm)** — age-dependent baseline with circadian and activity-driven modulation.
  * **Body temperature (°C)** — with diurnal variation and activity influence.
  * **Ambient temperature (°C)** — environment-dependent baseline.
  * **Contact index** — indicates whether the device is being worn (based on body vs. ambient temperature).
  * **Water exposure flag** — simulates wet/submersion events.

## Injected Anomalies & Events

To simulate real-world scenarios, the script injects labeled anomalies into the dataset:

* **Forced removals**: abrupt removal of the device (sudden motion spike + temperature drop + contact lost).
* **Normal removals**: short, routine removals outside anomalous windows.
* **Fever events**: elevated body temperature and heart rate over tens of minutes.
* **Stillness events**: periods of unusually low activity, steps, and reduced heart rate.
* **Water submersion**: rare, dataset-wide event with water flag active, suppressed motion, and cooling effect.
* **Device dropouts**: short spans with missing or flatlined sensor readings.

All injected anomalies are recorded in a companion `events.csv` file with device ID, type, timestamp, and index ranges.

## Output

The script writes results to a `synthetic_dataset/` folder:

* **Per-device time series** (`device_X.parquet`) with full sensor data.
* **Events log** (`events.csv`) summarizing all injected anomalies.

## Example Use Cases

* Benchmarking anomaly detection methods.
* Testing wearable health monitoring pipelines.
* Simulating longitudinal child activity data when real datasets are unavailable.
* Generating synthetic ground-truth datasets for research.
