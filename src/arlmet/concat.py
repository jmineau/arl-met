import glob
import os
import subprocess

# This script takes 6 hourly HRRR files and concatenates these
# into daily chunks so avoid file limit issues within HYSPLIT,
# which will not allow more than 12 met files to be included as
# inputs.     DVM 6/18/2026

# What is our input and output directories?
input_dir = "./hrrr"
output_dir = "./HRRR_tmp"

# What months and year are we processing?
year = "2024"
months = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12]

# Make the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Loop through each month
for month in months:
    # Find all the unique days for the period of time we are looking at
    days = sorted(
        {
            os.path.basename(f).split("_")[0]
            for f in glob.glob(f"{input_dir}/{year}{month:02d}*_hrrr")
        }
    )

    # Loop through each day, and then find the HRRR files for the
    # corresponding day. Then concatenate the files together
    for day in days:
        files = sorted(glob.glob(f"{input_dir}/{day}_*_hrrr"))
        if len(files) == 0:
            continue
        outfile = f"{output_dir}/{day}_hrrr"
        cmd = ["cat"] + files
        with open(outfile, "wb") as fout:
            subprocess.run(cmd, stdout=fout, check=True)
        print(f"Created {outfile}")
