def get_class_name():
    with open("./ntu_rgbd_class_name.txt","r") as f:
        s = f.readlines()
    class_name = []
    for i in range(len(s)):
        class_name.append(s[i].split('.')[1].strip())
        #A1. drink water.
    return class_name 


CLASS_NAME = get_class_name()

classdiff =[[3], [6], [7], [8], [12], [13], [14], [17, 19], [20], [21], [22, 48], [23], [24], [25], [26], [30, 31], [32, 38, 39], [9, 33], [5, 15, 16, 34], [35], [36], [37], [41], [42], [0, 1, 2, 4, 10, 11, 18, 27, 28, 29, 40, 43, 44, 46, 47], [45], [51], [49, 50, 52, 53], [54], [55], [56], [57], [58], [59]]
for i in classdiff:
    if(len(i)==1):
        continue
    else:
        for j in i:
            print("{:<30}".format(CLASS_NAME[j]),end=" ")
        print()
