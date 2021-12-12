import numpy as np

def hit_rate(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(bought_list, recommended_list)

    hit_rate = (flags.sum() > 0) * 1

    return hit_rate


def hit_rate_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    bought_list = bought_list  # Тут нет [:k] !!
    recommended_list = recommended_list[:k]

    flags = np.isin(bought_list, recommended_list)

    hit_rate = (flags.sum() > 0) * 1

    return hit_rate


def precision(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(bought_list, recommended_list)

    precision = flags.sum() / len(recommended_list)

    return precision


def precision_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    bought_list = bought_list  # Тут нет [:k] !!
    recommended_list = recommended_list[:k]

    flags = np.isin(bought_list, recommended_list)

    precision = flags.sum() / len(recommended_list)

    return precision


def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    bought_list = bought_list  # Тут нет [:k] !!
    recommended_list = recommended_list[:k]

    flags = np.isin(bought_list, recommended_list)

    bought_sum = [prices_recommended[product] for product in bought_list]
    recommended_sum = sum([prices_recommended[product] for product in recommended_list])

    precision = np.dot(flags, bought_sum) / recommended_sum

    return precision


def recall(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(bought_list, recommended_list)

    recall = flags.sum() / len(bought_list)

    return recall


def recall_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    bought_list = bought_list  # Тут нет [:k] !!
    recommended_list = recommended_list[:k]

    flags = np.isin(bought_list, recommended_list)

    recall = flags.sum() / len(bought_list)

    return recall


def money_recall_at_k(recommended_list, bought_list, prices_recommended, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    bought_list = bought_list  # Тут нет [:k] !!
    recommended_list = recommended_list[:k]

    flags = np.isin(bought_list, recommended_list)

    bought_sum = [prices_recommended[product] for product in bought_list]

    recall = np.dot(flags, bought_sum) / sum(bought_sum)

    return recall


def ap_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)[:k]

    relevant_indexes = np.nonzero(np.isin(recommended_list, bought_list))[0]

    if len(relevant_indexes) == 0:
        return 0

    amount_relevant = len(relevant_indexes)

    sum_ = sum(
        [precision_at_k(recommended_list, bought_list, k=index_relevant + 1) for index_relevant in relevant_indexes])
    return sum_ / amount_relevant

def map_k(recommended_list, bought_list, k=5):
    recommended_list = recommended_list[:k]
    sum_bought = 0
    count_users = len(recommended_list_3_users)

    for num in range(count_users):
        sum_bought += ap_k(recommended_list[num], bought_list[num], k=5)

    result = sum_bought / count_users

    return result


# по желанию
def ndcg_at_k(recommended_list, bought_list, k=5):
    recommended_list = recommended_list[:k]
    flags = np.isin(recommended_list, bought_list)

    discount = []
    for num in range(1, len(recommended_list) + 1):
        if num <= 2:
            discount.append(num)
        else:
            discount.append(math.log2(num))
    #     discount = [math.log2(i) for i in range(1, len(recommended_list) + 1)]

    DCG = np.dot(flags, discount) / len(recommended_list)
    NDCG = DCG / np.dot(np.ones(len(recommended_list)), discount)

    return NDCG


def reciprocal_rank(recommended_list, bought_list, k=1):
    rank = []
    recommended_list = recommended_list[:k]
    for num in range(len(recommended_list)):
        recommend = recommended_list[num]
        bought = bought_list[num]
        flags = np.isin(bought, recommend)
        for elem in flags:
            if elem == True:
                rank.append(list(flags).index(True) + 1)
                continue
    result = np.mean(rank)
    return result