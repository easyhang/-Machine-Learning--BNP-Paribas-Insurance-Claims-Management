# Write a data into a .txt file which save some data we need to use in future

def writer(filename, data):
    with open(filename,"r+") as file:
        file.read()
        # print data
        for i,x in enumerate(data):
            file.write(str(x))
            if i != len(data) -1:
                file.write(" ")
        file.write("\n")