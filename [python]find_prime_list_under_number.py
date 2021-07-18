input = 20


def find_prime_list_under_number(number):
    prime_list = []

    for i in range(1, number+1):
        for j in range(2, i+1):
            if i == 2: #2의 경우
                prime_list.append(i)
            elif i%j == 0: #소수일 경우
                break
            elif j==i-1:
                prime_list.append(i)
    return prime_list


result = find_prime_list_under_number(input)
print(result)
