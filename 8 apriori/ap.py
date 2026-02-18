import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv('../dataset/market_basket.csv', header=None)
transactions = []
for i in range(len(df)):
    transactions.append([str(df.values[i, j]) for j in range(len(df.columns)) if str(df.values[i, j]) != 'nan'])
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
sorted_rules = rules.sort_values(by='confidence', ascending=False)
result = sorted_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10)
print(result)