import csv
import random
name_to_stats = {}
pts = []

NUM_GAMES_AVG = 10

with open("../modified-datasets/combinedCSVs_date_sorted.csv", "r") as f:
	csv_reader = csv.reader(f)
	val_map = {0: .5, 1: .25, 2: .125, 3: .075, 4: .05} 
	for row in csv_reader:
		pts += [(row[1], row[2])]
		if row[1] not in name_to_stats:
			name_to_stats[row[1]] = []
		better_row = []
		for elem in row:
			try:
				better_row += [float(elem)]
			except ValueError:
				pass
		for i in range(len(name_to_stats[row[1]]) - 1, max(len(name_to_stats[row[1]]) - NUM_GAMES_AVG, -1), -1):
			for j in range(min(len(name_to_stats[row[1]][i]), len(better_row))):
				name_to_stats[row[1]][i][j] += better_row[j] * val_map[i % 5]
		name_to_stats[row[1]] += [[0 for i in range(len(better_row))]]
		# for i in range(len(name_to_stats[row[1]][max(len(name_to_stats[row[1]]) - NUM_GAMES_AVG, 0)])):
		# 	name_to_stats[row[1]][max(len(name_to_stats[row[1]]) - NUM_GAMES_AVG, 0)][i] /= max(min(len(name_to_stats[row[1]]) - NUM_GAMES_AVG + 1, NUM_GAMES_AVG), 1)

# print(pts)
with open('weighted_average_output.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    for key in name_to_stats:
    	for game in name_to_stats[key]:
    		inp = game
    		if len(inp) < 51:
    			inp += [random.randrange(0, 5) for i in range(51 - len(inp))]
    		spamwriter.writerow([0] + inp[1:])
# print(name_to_stats["Curry, Stephen^"])
