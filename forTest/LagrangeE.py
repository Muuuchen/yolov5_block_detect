from sympy import *

def LagrangeE(x0,y0,xe,ye,A,B,C,f):
    x = symbols("x")
    y = symbols("y")
    l = symbols("l")

    L = (x - x0)**2 + (y - y0)**2  - l *(A*(x-xe)**2+B*(x-xe)*(y-ye)+C*(y-ye)**2+f)
    difyL_x = diff(L, x)
    difyL_y = diff(L, y)
    difyL_l = diff(L, l)
    print(difyL_x)
    print(difyL_y)
    print(difyL_l)

    aa = solve([difyL_x, difyL_y, difyL_l],[x,y,l])
    print(aa)


print("adas")
LagrangeE(0,0,50,100,10,3,20,-200)
