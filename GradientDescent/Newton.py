# 在函数Newton_3中实现利用牛顿迭代法求解方程 𝑓(𝑡)=𝑡3−2=0 在 𝑡0=1 附近的根（求导数功能可以自建函数实现）:

def func_f(x):
    return x**3-2

def func_df(x):
    return 3*(x**2)

def Newton(c, t):
    while abs(t * t * t - c) > 1e-6:
        t = t - func_f(t)/func_df(t)
    return t

if __name__ == '__main__':
    t = Newton(2, 1)
    print(func_f(t))