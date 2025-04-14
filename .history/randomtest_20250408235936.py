from numpy.random import default_rng

def main():
    rng = default_rng(42)

    # 生成随机变量 A
    # A = rng.random(3)
    # print("A:", A)

    # 生成随机变量 B（传入 rng）
    B = generateB(rng)
    print("B:", B)

if __name__ == "__main__":
    main()
