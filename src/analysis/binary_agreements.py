import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa

# Load the data
df = pd.read_csv('culture_annotation_datasets_human_ita_selection.csv')

# Convert scores to binary (0 for 1-2, 1 for 3-4)
df['annotator_1_binary'] = df['annotator_1'].apply(lambda x: 1 if x >= 3 else 0)
df['annotator_2_binary'] = df['annotator_2'].apply(lambda x: 1 if x >= 3 else 0)
df['annotator_3_binary'] = df['annotator_3'].apply(lambda x: 1 if x >= 3 else 0)

# Calculate pairwise Cohen's kappa
kappa_1_2 = cohen_kappa_score(df['annotator_1_binary'], df['annotator_2_binary'])
kappa_1_3 = cohen_kappa_score(df['annotator_1_binary'], df['annotator_3_binary'])
kappa_2_3 = cohen_kappa_score(df['annotator_2_binary'], df['annotator_3_binary'])

# Calculate average pairwise kappa
avg_kappa = (kappa_1_2 + kappa_1_3 + kappa_2_3) / 3

# Prepare data for Fleiss' kappa
# Create a matrix where each row represents an item and each column represents a category (0 or 1)
# Values are counts of annotators assigning that category
ratings = []
for _, row in df.iterrows():
    binary_scores = [row['annotator_1_binary'], row['annotator_2_binary'], row['annotator_3_binary']]
    count_0 = binary_scores.count(0)
    count_1 = binary_scores.count(1)
    ratings.append([count_0, count_1])

# Calculate Fleiss' kappa
fleiss = fleiss_kappa(np.array(ratings))

# Calculate raw agreement percentage
agreement_all = sum(1 for _, row in df.iterrows() 
                   if row['annotator_1_binary'] == row['annotator_2_binary'] == row['annotator_3_binary']) / len(df)

# Calculate percentage where majority agrees (at least 2 annotators agree)
majority_agreement = sum(1 for _, row in df.iterrows() 
                        if (row['annotator_1_binary'] == row['annotator_2_binary']) or 
                           (row['annotator_1_binary'] == row['annotator_3_binary']) or 
                           (row['annotator_2_binary'] == row['annotator_3_binary'])) / len(df)

# Print results
print(f"Pairwise Cohen's kappa scores:")
print(f"Annotators 1-2: {kappa_1_2:.4f}")
print(f"Annotators 1-3: {kappa_1_3:.4f}")
print(f"Annotators 2-3: {kappa_2_3:.4f}")
print(f"Average pairwise kappa: {avg_kappa:.4f}")
print(f"Fleiss' kappa: {fleiss:.4f}")
print(f"Raw agreement (all 3 annotators): {agreement_all:.4f}")
print(f"Majority agreement (at least 2 annotators): {majority_agreement:.4f}")

# Distribution of binary scores
binary_counts = {
    'Annotator 1': df['annotator_1_binary'].value_counts().to_dict(),
    'Annotator 2': df['annotator_2_binary'].value_counts().to_dict(),
    'Annotator 3': df['annotator_3_binary'].value_counts().to_dict()
}
print("\nDistribution of binary scores:")
print(pd.DataFrame(binary_counts))