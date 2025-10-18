from matplotlib import pyplot as plt

starting_amount = 160000
years = 10
interest_rate = 0.09

amounts = [starting_amount]

for _ in range(years):
    amounts.append(amounts[-1] + amounts[-1] * interest_rate)

plt.plot(amounts)



plt.show()