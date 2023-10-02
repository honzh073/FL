import os
import csv
import pandas as pd


# image path
image_folder = "/local/data1/honzh073/Data/named_data"

# get image list
image_files = [i for i in os.listdir(image_folder) if i.endswith(".png")]

# create csv file
with open("labels.csv", "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["image_name,","category"])

    for image_file in image_files:
        if "NFF" in image_file:
            category = 0
        elif "AFF" in image_file:
            category = 1
        else:
            continue
        csv_writer.writerow([image_file, category])

print("CSV file 'labels.csv' created.")


# statistcs by pandas
df = pd.read_csv(r'/local/data1/honzh073/model/localnn/labels.csv')
num_image = df.shape[0]
sum_AFF = df['category'].sum()
sum_NFF = df.shape[0] - sum_AFF


print(f'Number of images: {num_image}')
print(f'Number of AFF images: {sum_AFF}')
print(f'Number of NFF images: {sum_NFF}')

