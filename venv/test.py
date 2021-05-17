from random import randint

numbers = []

#정답 숫자 세 개 뽑을 때까지 반복
while len(numbers) < 3:
    new_number = randint(0,9)
    while new_number in numbers:
        new_number = randint(0,9)
    numbers.append(new_number)

print("0과 9 사이의 서로 다른 세 숫자를 랜덤한 순서로 뽑았습니다.\n\n세 수를 하나씩 차례대로 입력하세요.")

strike = 0
ball = 0
guesses = []    #사용자가 제시한 숫자 리스트
tries = 0    #정답 시도 횟수

#도전 숫자 세 개 뽑을 때까지 반복
while len(guesses) < 3:
    guess = int(input("%d번째 수를 입력하세요:" % (len(guesses)+1)))
    if guess in guesses:
        print("중복되는 수 입니다. 다시 입력해주세요.")
    elif guess > 9 or guess < 0:
        print("범위를 벗어나는 수입니다. 다시 입력해주세요.")
    else :
        guesses.append(guess)
print("%dS, %dB" % (strike, ball))
tries += 1

#스트라이크가 세 개 일때까지 반복
i = 0
while strike < 3:
    if guesses[i] == numbers[i]:
        strike += 1
    elif guesses[i] in numbers:
        ball += 1
    else :
        print("축하합니다. %d번만에 세 숫자의 값과 위치를 모두 맞추셨습니다." % (tries))
