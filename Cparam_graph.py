import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

#default is 1.0
C = [0.001, 0.002, 0.01, 0.02, 0.1, 0.2, 1.0, 2.0]
Accuracy = [98.83, 99.12, 99.65, 99.77, 99.93, 99.96, 99.98, 99.99]

C_new = [0,2,4,6,8,10,12,14]

C_string = ['0.001', '0.002', '0.01', '0.02', '0.1', '0.2', '1.0', '2.0']
plt.xticks(C_new, C_string, rotation=45)


plt.plot(C_new, Accuracy)
plt.xlabel("C parameter")
plt.ylabel("Accuracy")
plt.title("C parameter - Accuracy")
plt.axis([0, 14, 98.5, 100.5])
plt.show()