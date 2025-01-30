# Quick script to manipulate inputs

int_vals = []
with open("input.txt", 'r') as file:
    content = file.readlines()
    if content[-1].endswith('\n'):
        content[-1] = content[-1].rstrip('\n')
    for line in content:
        x1, y1, x2, y2 = map(float, line.strip().split())
        int_vals.append((int(x1/10), int(y1/10), int(x2/10), int(y2/10)))
with open("input_int_small.txt", 'w') as file:
    for int_val in int_vals:
        file.write(str(int_val[0]) + ' ' + str(int_val[1]) + ' ' + str(int_val[2]) + ' ' + str(int_val[3]) + '\n')