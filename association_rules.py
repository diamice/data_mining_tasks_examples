import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from tabulate import tabulate

# Данные о транзакциях
dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

# Преобразование данных транзакций в DataFrame
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Поиск частых наборов
frequent_itemsets = fpgrowth(df, min_support=0.6, use_colnames=True)

# Генерация ассоциативных правил
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)

# Преобразование столбцов antecedents и consequents в более читаемый формат
rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

# Форматирование числовых столбцов для двух знаков после запятой
numeric_cols = ['antecedent support', 'consequent support', 'support',
                'confidence', 'lift', 'leverage', 'conviction', 'zhangs_metric']
rules[numeric_cols] = rules[numeric_cols].applymap(lambda x: f"{x:.2f}" if isinstance(x, float) else x)

# Вывод таблицы
print(tabulate(rules, headers='keys', tablefmt='pretty', showindex=True))
