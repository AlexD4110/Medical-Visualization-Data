import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1

df = pd.read_csv('medical_examination.csv')
# 2
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)).apply(lambda x: 1 if x > 25 else 0)

# 3
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])


    # 6. Group and reformat the data to split it by cardio and show counts of each feature. Rename the 'size' column.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size().rename(columns={'size': 'total'})
    

    # 7. Draw the catplot with sns.catplot
    fig = sns.catplot(x="variable", y="total", hue="value", col="cardio", data=df_cat, kind="bar")


    # 8
    fig = fig.figure


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &  # Keep correct blood pressure data
        (df['height'] >= df['height'].quantile(0.025)) &  # Filter height >= 2.5th percentile
        (df['height'] <= df['height'].quantile(0.975)) &  # Filter height <= 97.5th percentile
        (df['weight'] >= df['weight'].quantile(0.025)) &  # Filter weight >= 2.5th percentile
        (df['weight'] <= df['weight'].quantile(0.975))    # Filter weight <= 97.5th percentile
    ]

    # 12
    corr = df_heat.corr()

    # 13. Generate a mask for the upper triangle (to hide the redundant half of the correlation matrix)
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # 14
    fig, ax = plt.subplots(figsize=(12, 8))

    # 15
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', cmap='coolwarm', square=True, linewidths=0.5, ax=ax)


    # 16
    fig.savefig('heatmap.png')
    return fig
