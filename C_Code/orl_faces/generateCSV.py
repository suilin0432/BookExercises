import os

rootDIR = "/Users/mac/Desktop/book/ExerciseCode_Python/C_Code/orl_faces"
file = open("dataMessage.csv", "w")
fileLabelList = []
d = os.walk(rootDIR)
next(d)
for root, dirs, files in d:
    number = int(root.split("orl_faces/s")[1])-1
    for f in files:
        fileLabelList.append((os.path.join(rootDIR, root, f), number))
for fileSrc, index in fileLabelList:
    file.write("{0},{1}\n".format(fileSrc, index));
file.close();

