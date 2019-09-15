from math import sqrt, log2, log

def KLCal(a, b):
    assert len(a) == len(b)
    result = 0
    for i in range(len(a)):
        result += a[i] * log2(a[i]/b[i])
    return result

def equal(a, b):
    return abs(a-b) < 1e-8

A = [1/2, 1/2]
B = [1/4, 3/4]
C = [1/8, 7/8]

# calculate the 9 distance
dAA = KLCal(A, A)
dAB = KLCal(A, B)
dAC = KLCal(A, C)
dBA = KLCal(B, A)
dBB = KLCal(B, B)
dBC = KLCal(B, C)
dCA = KLCal(C, A)
dCB = KLCal(C, B)
dCC = KLCal(C, C)

# state things about the first property of distance --- d(x,y) >= 0
print ("(1)non-negative property: {0}".format(dAA>=0 and dAB>=0 and dAC>=0 and dBA>=0
                                              and dBB>=0 and dBC>=0 and dCA>=0 and dCB>=0 and dCC>=0))

# state things about the second property of distance --- d(x,y) == d(y,x)
# print("(2)symmetric property: {}".format((dAB == dBA) and (dAC == dCA) and (dBC == dCB)))
# the line below has some issues... because the error of the float data
print("(2)symmetric property: {}".format(equal(dAB, dBA) and equal(dAC, dCA) and equal(dBC, dCB)))

# state things about the third property of distance --- d(x, y) == 0 if and only if x == y
print("(3)identity of indiscernibles: {}".format(equal(dAA, 0) and equal(dBB, 0) and equal(dCC, 0) and not equal(dAB, 0)
                                          and not equal(dAC, 0) and not equal(dBA, 0) and not equal(dBC, 0) and not equal(dCA, 0)
                                          and not equal(dCB, 0)))
# state things about the forth property of distance --- d(x, z) <= d(x, y) + d(y, z)
print("(4)triangle inequality: {}".format((dAC <= dAB + dBC) and (dAB <= dAC + dCB) and (dBA <= dBC + dCA) and (dBC <= dBA + dAC)
                                          and (dCA <= dCB + dBA) and (dCB <= dCA + dAB)))
