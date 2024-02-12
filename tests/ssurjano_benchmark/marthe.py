import pandas as pd
import os

dirPath = os.path.dirname(os.path.realpath(__file__))


# Load the dataset
marthedata = pd.read_csv(
    dirPath + "/marthedata.txt", delim_whitespace=True, header=0
)

# Extract input variables
per1 = marthedata["per1"]
per2 = marthedata["per2"]
per3 = marthedata["per3"]
perz1 = marthedata["perz1"]
perz2 = marthedata["perz2"]
perz3 = marthedata["perz3"]
perz4 = marthedata["perz4"]
d1 = marthedata["d1"]
d2 = marthedata["d2"]
d3 = marthedata["d3"]
dt1 = marthedata["dt1"]
dt2 = marthedata["dt2"]
dt3 = marthedata["dt3"]
kd1 = marthedata["kd1"]
kd2 = marthedata["kd2"]
kd3 = marthedata["kd3"]
poros = marthedata["poros"]
i1 = marthedata["i1"]
i2 = marthedata["i2"]
i3 = marthedata["i3"]

# Extract output variables
p102K = marthedata["p102K"]
p104 = marthedata["p104"]
p106 = marthedata["p106"]
p276 = marthedata["p2.76"]
p29K = marthedata["p29K"]
p31K = marthedata["p31K"]
p35K = marthedata["p35K"]
p37K = marthedata["p37K"]
p38 = marthedata["p38"]
p4b = marthedata["p4b"]
