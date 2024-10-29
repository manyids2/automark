# Automark

Same problem for the notion, but here is the relevant information:
Setting up the automatic grading system:

- download the list of students from datanose in xlsx and change it to csv
    file: https://datanose.nl/#course[99215]
- Digitalocean droplet
- Login with my google account
- uva_Server_8021_avu
- check if python is installed on the droplet
- Install pip
- pip install Flask
- pip install -U Werkzeug
- pip install numpy
- Git clone the project: git clone
    https://github.com/Melika-Ayoughi/automark.git
- Go to automark.py and change the host ip in the config class to the machine
    ip.(the server ip they have to connect to)
- Cd in example, run the generate_assignment.py file to generate test files and
    the users
- Run python import_students_csv.py ./student.csv (you have to download the
    list of students xlsx beforehand and convert it to csv from a website or
    something) here the students will be added to the user folder
- Type screen to open a new terminal screen
- Run python main.py
- Type screen to open a new terminal screen
- Run python admin.py (always running, it's the one that should be running for
    us to check everyone's result)
- in automark.py the file that is sent to students next to their lab
    assignment, put the machine IP in config.host
- https://cloud.digitalocean.com/droplets/271233437/access?i=2fa2c5
- http://178.62.224.167:9999/_admin/progress/done?assignment=2
- http://178.62.224.167:9999/_admin/progress/done

Email address and password:
appliedml.uva@gmail.com
rHJWgcyB68Xaayd

@0Appliedmachinelearning

rsync SRC/ root@161.35.93.12:/root/
