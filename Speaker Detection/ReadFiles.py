import os
def get_Files(s):
    F_Name = []
    for path  , dirs , files in os.walk(s):
        for f in files:
            fileName = path+'/'+f
            F_Name.append(fileName)


    return F_Name


# r= []



# for i in r:
#     print(i)


