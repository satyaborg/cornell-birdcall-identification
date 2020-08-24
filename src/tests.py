#%%
import torch


model = torch.hub.load('harritaylor/torchvggish', 'vggish')
model.eval()

# Download an example audio file
# import urllib
# url, filename = ("http://soundbible.com/grab.php?id=1698&type=wav", "bus_chatter.wav")
# try: urllib.URLopener().retrieve(url, filename)
# except: urllib.request.urlretrieve(url, filename)

x = model.forward(filename)
# print(x.shape)

#%%
import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("../data/train_combined.csv")
y = df.ebird_code
X = df.loc[:, df.columns != "ebird_code"]
X_train, X_test, _, _ = train_test_split(X, y,
                                    random_state=42, 
                                    test_size=0.2, 
                                    stratify=y
                                    )
y_train.value_counts()/y_train.shape*100, y_test.value_counts()/y_test.shape*100

#%%


# %%

# %%
(y.value_counts()/y.shape)*100
# %%
y_train.index.tolist()[:5]
# %%
X_train.index.tolist()[:5]
X_test.index.tolist()[:5]