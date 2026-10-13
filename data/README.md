# Dataset

The dataset is **not included** in this repository. The study uses the Kaggle single-label version of the AI4I 2020
predictive maintenance dataset:

* **File:** `predictive_maintenance.csv` (10,000 rows; columns `UDI, Product ID, Type, Air temperature [K],
  Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min], Target, Failure Type`)
* **Source:** Bansal, S. (2021). *Machine predictive maintenance classification* [Dataset]. Kaggle.
  https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification
* **Original data:** AI4I 2020 predictive maintenance dataset, UCI Machine Learning Repository (CC BY 4.0),
  https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset ; Matzka (2020).

The UCI release records the five mechanisms as separate binary flags; the Kaggle version assigns one `Failure Type`
per record. The two are **not interchangeable** for this code: use the Kaggle file.

## Steps

1. Sign in to Kaggle, open the dataset page above, and download it (the download is a zip containing
   `predictive_maintenance.csv`). Alternatively, with the Kaggle API:
   `kaggle datasets download -d shivamb/machine-predictive-maintenance-classification --unzip -p data/`
2. Put `predictive_maintenance.csv` in this folder (`data/`). It is ignored by git.
3. Check it:
   ```bash
   python -m ftaa.check_data data/predictive_maintenance.csv
   ```
   The check must end with `OK: matches the dataset used in the study`. It verifies the row count, the columns,
   and the complete label audit:

   | Failure Type | Target = 0 | Target = 1 |
   |---|---|---|
   | Heat Dissipation Failure | 0 | 112 |
   | No Failure | 9,643 | 9 |
   | Overstrain Failure | 0 | 78 |
   | Power Failure | 0 | 95 |
   | Random Failures | 18 | 0 |
   | Tool Wear Failure | 0 | 45 |

   It also prints the file's SHA-256 checksum, which you can record for your own copy.
