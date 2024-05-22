import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

penguins_df = pd.read_csv("data/penguins.csv")
penguins_clean=penguins_df.dropna()


penguins_clean[penguins_clean['flipper_length_mm']>4000]
penguins_clean[penguins_clean['flipper_length_mm']<0]
penguins_clean = penguins_clean.drop([9,14])

df = pd.get_dummies(penguins_clean).drop("sex_.", axis=1)
scaler=StandardScaler()
scaled=scaler.fit_transform(df)
penguins_preprocessed = pd.DataFrame(data=scaled, columns=df.columns)

pca=PCA()
pca.fit(penguins_preprocessed)
n_components= sum(pca.explained_variance_ratio_ > 0.1)
pca=PCA(n_components=n_components)
penguins_PCA = pca.fit_transform(penguins_preprocessed)

inertia=[]
for i in range(1,10):
    kmeans= KMeans(n_clusters=i, random_state=42).fit(penguins_PCA)
    inertia.append(kmeans.inertia_)
    
n_clusters=4
kmeans=KMeans(n_clusters=n_clusters).fit(penguins_PCA)

penguins_clean['label']=kmeans.labels_

ncolumns=penguins_clean.select_dtypes(include='number')
numeric_columns=[]
for col in ncolumns.columns:
    numeric_columns.append(col)
    
stat_penguins=penguins_clean.groupby('label')[numeric_columns].mean()
