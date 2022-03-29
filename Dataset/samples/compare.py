import re


with open("./acc1.txt","r") as f:
    s1 = f.read()
with open("./acc2.txt","r") as f:
    s2 = f.read()

pattern = re.compile("(?<='class_accuracy': )\d\.\d+")
acc1 = re.findall(pattern,s1)
acc2 = re.findall(pattern,s2)
acc_diff = [float(acc1[i])-float(acc2[i]) for i in range(60)]
# print(acc_diff)
with open("./acc_diff.txt","w") as f:
    for i in acc_diff:
        f.write(str(i))
        f.write('\n')