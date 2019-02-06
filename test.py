temps = [2.5, 100,5, 10, 15, 22, 23, 44, 66, 88, 1, 0,800]


list = [temps.index(w) for w in sorted(temps)[-5:]][::-1]
for i in list:
    print(i)

