def get_p_at_n_in_m(data, n, m, ind):
    pos_score = data[ind][0];
    curr = data[ind:ind + m];
    curr = sorted(curr, key=lambda x: x[0], reverse=True)

    if curr[n - 1][0] <= pos_score:
        return 1;
    return 0;


def evaluate(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip();
            tokens = line.split("\t")

            if len(tokens) != 2:
                continue

            data.append((float(tokens[0]), int(tokens[1])));

    # assert len(data) % 10 == 0

    p_at_1_in_2 = 0.0
    p_at_1_in_10 = 0.0
    p_at_2_in_10 = 0.0
    p_at_5_in_10 = 0.0
    length = len(data) / 10  # pos:neg=1:9 10条组成一个数据
    print(length)
    length = int(length)

    for i in range(0, length):
        ind = i * 10
        assert data[ind][1] == 1

        p_at_1_in_2 += get_p_at_n_in_m(data, 1, 2, ind)
        p_at_1_in_10 += get_p_at_n_in_m(data, 1, 10, ind)
        p_at_2_in_10 += get_p_at_n_in_m(data, 2, 10, ind)
        p_at_5_in_10 += get_p_at_n_in_m(data, 5, 10, ind)

    return (p_at_1_in_2 / length, p_at_1_in_10 / length, p_at_2_in_10 / length, p_at_5_in_10 / length)


def ComputeR10_1(scores, labels, count=10):
    total = 0
    correct = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            total = total + 1
            sublist = scores[i:i + count]
            if max(sublist) == scores[i]:
                correct = correct + 1
    print(f'10@1:{float(correct) / total}')


def ComputeR2_1(scores, labels, count=2):
    total = 0
    correct = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            total = total + 1
            sublist = scores[i:i + count]
            if max(sublist) == scores[i]:
                correct = correct + 1
    print(f'2@1:{float(correct) / total}')


def ComputeR10_5(scores, labels, count=10):
    total = 0
    correct = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            total += 1
            sublist = scores[i:i + count]
            sublist = sorted(sublist, reverse=True)
            #print(sublist)
            if scores[i] >= sublist[4]:
                correct += 1
    print(f"10@5:{float(correct) / total}")


def ComputeR10_2(scores, labels, count=10):
    total = 0
    correct = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            total += 1
            sublist = scores[i:i + count]
            sublist = sorted(sublist, reverse=True)
            #print(sublist)
            if scores[i] >= sublist[1]:
                correct += 1
    print(f"10@2:{float(correct) / total}")