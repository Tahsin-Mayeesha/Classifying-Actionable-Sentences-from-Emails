import numpy as np
import pandas as pd
np.random.seed(42)


action_items_given = pd.read_csv("actions.csv",names=['text']) # given small dataset labelled
email_labelled = pd.read_csv("email_labelled.csv") # first 5k samples from enron sentences labelled with rule based model
action_items_given["target"] = 1
full_dataset = pd.concat([email_labelled[email_labelled['target']==0],action_items_given]) # concatanate the data
full_dataset = full_dataset.sample(frac=1).reset_index(drop=True) # shuffle the data
full_dataset.to_csv("email_labelled.csv",index=False)





