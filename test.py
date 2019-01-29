class Test:

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

        self.d = self.a + self.b / self.c

    def getD(self):
        return self.d

test1 = Test(1,6,3)
print(test1.getD())
