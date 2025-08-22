def kaigenhao(x):
    if x < 0:
        return False
    if x == 1:
        return 1
    guess = x / 2
    while True:
        newguess = (guess + x / guess) / 2
        if abs(newguess - guess) < 1e-9:
            return newguess
        guess = newguess