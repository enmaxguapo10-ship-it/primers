import math

def es_primo(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

numeros = [
985973, 985979, 985981, 985991, 985993, 985997,
986023, 986047, 986053, 986071, 986101
# añade todos los que quieras
]

for n in numeros:
    if not es_primo(n):
        print("NO es primo:", n)