import pandas as pd
import json
import sys

# path to new list of students ( xlsx converted to csv )
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

# Read currently registered ids
with open("automark_server/users/user_list.json", "r") as f:
    old_uvanet_id = json.loads(f.read())

# Dump names and email for each uvanet_id
uvanet_ids_to_add = []
for id_, first_, middle_, second_, mail_ in zip(
    uvanet_id, first_name, middle_name, last_name, mail
):
    # Skip those that already exist
    # TODO: What about modifications? should we not check all cols equal?
    if str(id_) in old_uvanet_id:
        continue

    name = f"{first_} {middle_} {second_}" if middle_ else f"{first_} {second_}"
    file_path = "automark_server/users/user_info/{}.json".format(id_)
    data = {"name": name, "mail": mail_}
    with open(file_path, "w") as f:
        f.write(json.dumps(data))

    uvanet_ids_to_add.append(str(id_))

# Dump old and new ids to a json list
print("New ids: " + str(uvanet_ids_to_add))
with open("automark_server/users/user_list.json", "w") as f:
    f.write(json.dumps(old_uvanet_id + uvanet_ids_to_add))
