import os
import pandas as pd
import json
from collections import defaultdict
from itertools import chain, combinations
from tqdm import tqdm
import time


def powerset(iterable):
    """生成集合的所有非空子集"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def get_support(itemset, transactions):
    """计算项集的支持度"""
    count = 0
    for transaction in transactions:
        if itemset.issubset(transaction):
            count += 1
    return count / len(transactions)


def apriori(transactions, min_support=0.02):
    """实现Apriori算法挖掘频繁项集，带进度条"""
    items = sorted(set(chain(*transactions)))

    # 生成1-项集
    frequent_itemsets = []
    k = 1
    C_k = [{item} for item in items]

    print("\nApriori算法执行进度：")
    while C_k:
        print(f"\n处理{k}-项集...")
        # 计算支持度并筛选频繁项集
        L_k = []
        for itemset in tqdm(C_k, desc=f"计算{k}-项集支持度"):
            support = get_support(itemset, transactions)
            if support >= min_support:
                L_k.append((itemset, support))

        # 保存当前k的频繁项集
        frequent_itemsets.extend(L_k)

        # 生成候选项集C_{k+1}
        k += 1
        C_k = []
        print(f"生成{k}-项集候选项...")
        time.sleep(0.1)  # 为了显示进度条
        for i in tqdm(range(len(L_k)), desc=f"生成{k}-项集"):
            for j in range(i + 1, len(L_k)):
                itemset1 = L_k[i][0]
                itemset2 = L_k[j][0]

                # 如果前k-2个元素相同，则合并
                if len(itemset1.union(itemset2)) == k:
                    new_itemset = itemset1.union(itemset2)
                    # 检查所有k-1子集是否频繁
                    valid = True
                    for subset in combinations(new_itemset, k - 1):
                        subset = frozenset(subset)
                        if not any(fs[0] == subset for fs in L_k):
                            valid = False
                            break
                    if valid and new_itemset not in [c[0] for c in C_k]:
                        C_k.append(new_itemset)

    return frequent_itemsets


def generate_rules(frequent_itemsets, transactions, min_confidence=0.5):
    """从频繁项集中生成关联规则，带进度条"""
    rules = []
    # 只考虑项集大小≥2的情况
    large_itemsets = [itemset for itemset, _ in frequent_itemsets if len(itemset[0]) >= 2]

    print("\n生成关联规则...")
    for itemset, support in tqdm(large_itemsets, desc="生成规则"):
        # 生成所有非空真子集
        subsets = list(powerset(itemset))
        for antecedent in subsets:
            antecedent = frozenset(antecedent)
            if antecedent != itemset:  # 确保是真子集
                consequent = itemset - antecedent
                # 计算置信度
                antecedent_support = next(
                    (s for s, sup in frequent_itemsets if s == antecedent), 0
                )
                if antecedent_support > 0:
                    confidence = support / antecedent_support
                    if confidence >= min_confidence:
                        rules.append({
                            'antecedent': antecedent,
                            'consequent': consequent,
                            'support': support,
                            'confidence': confidence,
                            'lift': confidence / next(
                                (sup for s, sup in frequent_itemsets if s == consequent), 0
                            ) if next((sup for s, sup in frequent_itemsets if s == consequent), 0) > 0 else float('inf')
                        })

    return rules


folder_path = './data/30G_data_new/'
parquet_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.parquet')]

purchase_history = []

for idx, file in enumerate(parquet_files[0:1]):
    df = pd.read_parquet(file)
    purchase_history.append(df['purchase_history'])

purchase_history = purchase_history[0]
data = []
for line in purchase_history:
    line = json.loads(line)
    data.append(line['items'])

purchase_history = data
transactions = [[item['id'] for item in transaction] for transaction in purchase_history]

# 挖掘频繁项集
frequent_itemsets = apriori(transactions, min_support=0.02)

# 生成关联规则
rules = generate_rules(frequent_itemsets, transactions, min_confidence=0.5)
print("频繁项集（支持度≥0.02）：")
for itemset, support in frequent_itemsets:
    print(f"项集: {itemset}, 支持度: {support:.4f}")

print("\n关联规则（置信度≥0.5）：")
for rule in rules:
    print(f"{rule['antecedent']} → {rule['consequent']}")
    print(f"  支持度: {rule['support']:.4f}, 置信度: {rule['confidence']:.4f}, 提升度: {rule['lift']:.4f}")
