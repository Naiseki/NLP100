def main():
    s1 = "パトカー"
    s2 = "タクシー"
    result = "".join([a + b for a, b in zip(s1, s2)])
    print(result)


if __name__ == "__main__":
    main()
