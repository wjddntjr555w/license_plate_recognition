def solution(m, n, puddles):
    answer = 0
    arr = [[0 for col in range(n + 2)] for row in range(m + 2)]
    num = [[0 for col in range(n + 2)] for row in range(m + 2)]
    for i in range(0, len(puddles)):
        arr[puddles[i][1]][puddles[i][0]] = 1
    arr[n][m] = 2

    num[0][1] = 1
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if arr[i][j] != 1:
                num[i][j] += num[i - 1][j] % 1000000007
                num[i][j] += num[i][j - 1] % 1000000007
    answer = num[n][m] % 1000000007
    print(answer)
    return answer

solution(100,100,[[2,2],[3,4],[5,7],[7,5]])
1000000007
1039980675