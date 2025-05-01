from decimal import Decimal, getcontext
import argparse

# Set precision for Decimal calculations (e.g., 100 decimal places)
getcontext().prec = 100

def compute_stirling(m):
    """
    Compute Stirling numbers of the second kind S(m, x) for x = 0 to m using DP.
    Returns a list S where S[x] = S(m, x).
    """
    # S[i][j] represents S(i, j)
    S = [[0] * (m + 1) for _ in range(m + 1)]
    S[0][0] = 1  # Base case: S(0, 0) = 1
    for i in range(1, m + 1):
        for j in range(1, i + 1):
            S[i][j] = j * S[i - 1][j] + S[i - 1][j - 1]
    return S[m]

def compute_factorials(m):
    """
    Compute factorials up to m!.
    Returns a list fact where fact[x] = x!.
    """
    fact = [1] * (m + 1)
    for x in range(1, m + 1):
        fact[x] = fact[x - 1] * x
    return fact

def binomial(n, x):
    """
    Compute binomial coefficient n choose x using big integers.
    """
    if x == 0:
        return 1
    # Compute iteratively to avoid large factorials
    result = 1
    for j in range(x):
        result = result * (n - j) // (j + 1)
    return result

def compute_probability(n, m, y):
    """
    Compute Pr{X >= y} for n bins, m balls, threshold y.
    """
    # Edge cases
    if m == 0:
        return Decimal('1') if y == 0 else Decimal('0')
    if y > m or y > n:
        return Decimal('0')
    if y <= 0:
        return Decimal('1')

    # Precompute Stirling numbers and factorials
    S = compute_stirling(m)
    fact = compute_factorials(m)

    # Compute n^m as a big integer
    nm = n ** m

    # Sum probabilities from x = y to min(n, m)
    total_prob = Decimal('0')
    upper = min(n, m)
    for x in range(y, upper + 1):
        # Compute f = binom(n, x) * x! * S(m, x) using big integers
        binom = binomial(n, x)
        f = binom * fact[x] * S[x]
        # Compute Pr{X = x} with high precision
        prob_x = Decimal(f) / Decimal(nm)
        total_prob += prob_x

    return total_prob

# Example usage
# n = 8000  # bins
# m = 900   # balls
# y = 1    # threshold
# result = compute_probability(n, m, y)
# print(f"Pr{{X >= {y}}} = {result}")

parser = argparse.ArgumentParser(description='Compute probability based on n,m,y.')
parser.add_argument('n', type=int, help='Number of bins')
parser.add_argument('m', type=int, help='Number of balls')
parser.add_argument('y', type=int, help='Threshold')
args = parser.parse_args()

result = compute_probability(args.n, args.m, args.y)
print(f"Pr{{X >= {args.y}}} = {result}")