import csv, json
import numpy as np
from collections import defaultdict

reader = csv.reader(open("analize_result/result.txt"), delimiter="\t")
q = []
indexes = ['20', '21', '22', '23', '30', '31', '32', '33',
           '40', '41', '43', '50', '51', '53', '60', '61', '63']
r_dist = {i:[] for i in indexes}
r = {i:[] for i in indexes}
for l in reader:
    data = json.loads("{\"" + l[1][1:-1])
    q.append([data["q"][i] for i in ["0", "1", "2", "3"]])
    for i in indexes:
        r_dist[i].append(int(data["r"][i][0]))
        r[i].append((int(data["r"][i][0]), int(data["r"][i][-1][-1])))
    # print(data["r"]["20"])
# for i in indexes:
#     print(i, np.mean(r_dist[i]))
# d = r["63"]
# check_dict = defaultdict(int)
# for j in d:
#     check_dict[j] += 1
# print(check_dict)
# exit()
# print(r_dist["23"])
# exit()

q = np.array(q).astype(np.int)
# print(q)
check = 3
print(np.mean(q[:, :, check], 0))
