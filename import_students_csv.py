"""
From instructions.md:

- Run python import_students_csv.py ./student.csv (you have to download the
    list of students xlsx beforehand and convert it to csv from a website or
    something) here the students will be added to the user folder
"""

import pandas as pd
import json
import sys

# path list of students ( xlsx converted to csv )
path = str(sys.argv[1])

# Read and clean up
df = pd.read_csv(path, delimiter=",", na_values=" ")
df = df.replace(float("NaN"), "")

# Columns used
uvanet_id = df["UvAnetID"]
student_id = df["StudentID"]
last_name = df["LastName"]
middle_name = df["MiddleName"]
first_name = df["FirstName"]
mail = df["Email"]

# Check that student ids match, and all ids are unique
assert uvanet_id.equals(student_id)
assert uvanet_id.is_unique

# Dump only the ids to a json list
with open("automark_server/users/user_list.json", "w") as f:
    f.write(json.dumps(uvanet_id.astype(str).tolist()))

# Dump names and email for each uvanet_id
for id_, first_, middle_, second_, mail_ in zip(
    uvanet_id, first_name, middle_name, last_name, mail
):
    name = f"{first_} {middle_} {second_}" if middle_ else f"{first_} {second_}"
    file_path = "automark_server/users/user_info/{}.json".format(id_)
    data = {"name": name, "mail": mail_}
    with open(file_path, "w") as f:
        f.write(json.dumps(data))
