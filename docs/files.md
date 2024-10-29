# files

## root

### automark.py

This script runs the client script for AutoMark
There are 2 main functions the end-user should use:

- get_progress(username) --- just to get the current progress to the stdout
- test_student_function(username, function, arg_keys) --- to test the provided function
  and to print the result / error to the stdout

  This scripts automatically downloads local tests into the `local_tests` folder
  Compatible with Python 2/3

  ```{aerial symbols}
    - 󰠱 ServerError
    - 󰠱 Config

    # Public API
    - 󰊕 get_progress
    - 󰊕 test_student_function

    # Actual meat
    - 󰊕 _remove_local_tests
    - 󰊕 _load_local_tests
    - 󰊕 _local_tests_are_valid
    - 󰊕 _passed_local_tests
    - 󰊕 _passed_remote_test
  ```

Config that we need to set properly:

```python
class Config:
    host = 'http://178.62.224.167:1234/'
    cwd = os.path.dirname(os.path.realpath(__file__))
    test_folder = os.path.join(cwd, 'local_tests')
    test_path = os.path.join(test_folder, 'tests.pickle')
```

Main functions:

- get_progress(username)
- test_student_function(username, function, arg_keys)

### others

- import_students_csv.py
  - input:
    - xl file from datanose, converted to csv
  - output:
    - `automark_server/users/user_list.json`
      - only ids
    - `automark_server/users/user_info/{id}.json` for each id
      - name, email
- update_students_csv.py
  - input:
    - xl file from datanose, converted to csv
    - `automark_server/users/user_list.json`
  - output:
    - `automark_server/users/user_list.json`
    - `automark_server/users/user_info/{id}.json` for each id

## automark_server

- main.py
- admin.py
- utils.py
- wrappers.py
- assignments
  - local_tests.pickle
  - remote_tests.pickle
- templates
  - index_assignment_1.html
  - index_assignment_2.html
- users
  - user_list.json
  - user_info
  - user_progress

## local_tests

- tests.pickle

## example

- uses `nnpn` library to generate tests
