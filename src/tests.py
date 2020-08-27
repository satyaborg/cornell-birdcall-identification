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
# y_train.value_counts()/y_train.shape*100, y_test.value_counts()/y_test.shape*100

#%%


# %%

# %%
(y.value_counts()/y.shape)*100
# %%import pandas as pd
import seaborn as sns

X_test.index.tolist()[:5]

#%%
# multilabel case expects binary label indicators with shape (n_samples, n_classes).
import numpy as np
from sklearn.metrics import roc_auc_score, multilabel_confusion_matrix, classification_report, average_precision_score, f1_score
threshold = 0.5
y_true = np.array([[0,0,0,1,1], [0,0,1,0,1]])
y_pred = np.array([[0.6,0,0,0,0.7], [0,0,0.3,0,1]])
codes = ["x", "y", "z", "a", "b"] # df.ebird_code.unique().tolist()
y_scores = np.where(y_pred > threshold, 1, 0)
print("True:\n ", y_true)
print("Predicted:\n ", y_scores)
cm = multilabel_confusion_matrix(y_true, y_scores) #, labels=codes)
print(cm)
# plot_confusion_matrix(cm, codes)
micro_avg_f1 = f1_score(y_true, y_scores, average='samples') 
print(classification_report(y_true,y_scores))
print(average_precision_score(y_true, y_scores, average="samples")) 
print(micro_avg_f1)
roc_auc_score(y_true, y_scores, average="samples")

# df = pd.DataFrame([y_true, y_pred], columns=['y_Actual','y_Predicted'])
# confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

# sns.heatmap(confusion_matrix, annot=True)
# plt.show()


# from sklearn.metrics import multilabel_confusion_matrix

# mul_c = multilabel_confusion_matrix(
#     test_Y,
#     pred_k,
#     labels=["benign", "dos","probe","r2l","u2r"])
# mul_c








# np.random.randint(0,2)