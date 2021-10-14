def solution(info):
    answer = []

    cnt = [0 for i in range(0, 86400)]

    for i in range(0, len(info)):
        # for j in range(info[i][0], info[i][1]+1):
        start = info[i][0]
        end = info[i][1]
        cnt[start:end] += 1

    _max = max(cnt)
    for i in range(0, 86400):
        if _max == cnt[i]:
            answer.append(i)

    return answer


solution([[1,5],[3,5],[7,8]])