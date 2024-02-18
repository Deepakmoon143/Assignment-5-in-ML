import pandas as pd

url = "https://raw.githubusercontent.com/husnainfareed/Simple-Naive-Bayes-Weather-Prediction/master/new_dataset.csv"
data = pd.read_csv(url)


frequency_tables = {}
likelihood_table = {}
features = data.columns[:-1] 
target_variable = "Play"
total_samples = len(data)


data['Play'] = data['Play'].str.lower()


prior_probabilities = data['Play'].value_counts(normalize=True)

if 'yes' not in prior_probabilities:
    prior_probabilities['yes'] = 0
if 'no' not in prior_probabilities:
    prior_probabilities['no'] = 0

for feature in features:
    
    frequency_table = data.groupby([feature, target_variable]).size().unstack(fill_value=0)
    frequency_tables[feature] = frequency_table
    
    
    feature_likelihoods = {}
    for feature_value in data[feature].unique():
        feature_count = len(data[data[feature] == feature_value])
        play_counts = data[data[feature] == feature_value][target_variable].value_counts()
        if 'yes' in play_counts:
            play_count_given_feature = play_counts['yes']
        else:
            play_count_given_feature = 0
        feature_likelihoods[feature_value] = play_count_given_feature / feature_count
    likelihood_table[feature] = feature_likelihoods


for feature, table in frequency_tables.items():
    print(f"Frequency table for {feature}:")
    print(table)
    print()


print("Likelihood table:")
for feature, likelihoods in likelihood_table.items():
    print(f"Feature: {feature}")
    for value, likelihood in likelihoods.items():
        print(f"    P({value} | Play = Yes) = {likelihood:.2f}")
    print()


posterior_probabilities = {}

for index, row in data.iterrows():
    posterior_probability_yes = prior_probabilities['yes']
    posterior_probability_no = prior_probabilities['no']
    for feature in features:
        value = row[feature]
        posterior_probability_yes *= likelihood_table[feature][value]
        posterior_probability_no *= likelihood_table[feature][value]
    posterior_probabilities[index] = {'Yes': posterior_probability_yes, 'No': posterior_probability_no}


print("Posterior probabilities:")
for index, probabilities in posterior_probabilities.items():
    print(f"Index: {index}, Post probabilities: {probabilities}")
