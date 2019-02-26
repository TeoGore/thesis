import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

#default is 1.0
C = [0.001, 0.002, 0.01, 0.02, 0.1, 0.2, 1.0, 2.0, 10.0, 20.0]
#C_string = ['0.001', '0.002', '0.01', '0.02', '0.1', '0.2', '1.0', '2.0', '10.0', '20.0']
#C_new = [1,2,3,4,5,6,7,8,9,10]
C_string = ['0.001', '0.002', '0.01', '0.02', '0.1', '0.2', '1.0', '2.0']
C_new = [1,2,3,4,5,6,7,8]
plt.xticks(C_new, C_string, rotation=45)

Accuracy = [85.38, 92.26, 98.32, 98.97, 99.42, 99.49, 99.74, 99.79]



plt.plot(C_new, Accuracy)
plt.xlabel("C parameter")
plt.ylabel("Accuracy")
plt.title("C parameter - Accuracy")
plt.axis([min(C_new), max(C_new), min(Accuracy)-0.1, 100])
plt.show()

#per salvare click dx sulla icona della immagine a dx del grafico