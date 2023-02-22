import numpy as np
import matplotlib.pyplot as plt
from os import path
from tp2_aux import images_as_matrix
from tp2_aux import report_clusters
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import f_classif
from sklearn.manifold import Isomap
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering, AffinityPropagation
from sklearn import metrics, preprocessing
from sklearn.metrics import precision_recall_fscore_support
import warnings

warnings.filterwarnings("ignore")

mat = np.loadtxt('labels.txt', delimiter=',')
y = mat[:, -1]
colors = ["b", "r", "g"]


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


if not path.isfile('best_features.npz'):
    data = images_as_matrix()
    data = (data - np.mean(data)) / np.std(data)
    pca = PCA(n_components=6)
    pca_data = pca.fit_transform(data)

    kPCA = KernelPCA(n_components=6)
    kPCA_data = kPCA.fit_transform(data)

    isomap = Isomap(n_components=6)
    isomap_data = isomap.fit_transform(data)

    np.savez('best_features.npz', pca_data=pca_data, kPCA_data=kPCA_data, isomap_data=isomap_data)
else:
    best_features = np.load('best_features.npz')
    lst = best_features.files
    pca_data = best_features[lst[0]]
    kPCA_data = best_features[lst[1]]
    isomap_data = best_features[lst[2]]

result = np.hstack((pca_data, kPCA_data, isomap_data))

result = preprocessing.normalize(result)
y_red = y[y != 0]
X = result[y != 0]

f, prob = f_classif(X, y_red)


def train_k_means(model, X, Y, ix):
    model_fit = model.fit(X)
    loss = model_fit.inertia_
    model_pred = model.predict(X)
    purity = purity_score(model_pred[ix], Y)
    precision, recall, f_measure, support = precision_recall_fscore_support(model_pred[ix], Y)
    rand_index = adjusted_rand_score(model_pred[ix], Y)
    precision = (precision * support).sum() / support.sum()
    recall = (recall * support).sum() / support.sum()
    f_measure = (f_measure * support).sum() / support.sum()

    return purity, precision, recall, f_measure, support, rand_index, loss


def train_model(model, X, Y, ix):
    model_pred = model.fit_predict(X)
    purity = purity_score(model_pred[ix], Y)
    precision, recall, f_measure, support = precision_recall_fscore_support(model_pred[ix], Y)
    rand_index = adjusted_rand_score(model_pred[ix], Y)
    precision = (precision * support).sum() / support.sum()
    recall = (recall * support).sum() / support.sum()
    f_measure = (f_measure * support).sum() / support.sum()

    return purity, precision, recall, f_measure, support, rand_index


n_feats_max_k_means, n_feats_max_spect, n_feats_max_agl = 0, 0, 0
max_k_means, max_spect, max_agl = 0, 0, 0
# testing for each algorithm the number of features that it prefers
for k_feats in range(2, 19):
    k_means_rand, spect_rand, agl_rand = [], [], []
    k_best = SelectKBest(f_classif, k=k_feats).fit(X, y_red)
    X_new = k_best.transform(result)
    for n_clusters in range(2, 12):
        purity, precision, recall, f_measure, support, rand_index = train_model(
            KMeans(n_clusters=n_clusters, n_init='auto', random_state=0),
            X_new, y_red, y != 0)
        k_means_rand.append(rand_index)
        purity, precision, recall, f_measure, support, rand_index = train_model(
            AgglomerativeClustering(n_clusters=n_clusters),
            X_new, y_red, y != 0)
        agl_rand.append(rand_index)
        purity, precision, recall, f_measure, support, rand_index = train_model(
            SpectralClustering(n_clusters=n_clusters, assign_labels='cluster_qr', n_neighbors=10),
            X_new, y_red, y != 0)
        spect_rand.append(rand_index)

    if max(k_means_rand) > max_k_means:
        max_k_means = max(k_means_rand)
        n_feats_max_k_means = k_feats

    if max(spect_rand) > max_spect:
        max_spect = max(spect_rand)
        n_feats_max_spect = k_feats
    if max(agl_rand) > max_agl:
        max_agl = max(agl_rand)
        n_feats_max_agl = k_feats
    plt.close()
print(f'AGL: {max_agl}, {n_feats_max_agl} features ')
print(f'Spect: {max_spect}, {n_feats_max_spect} features ')
print(f'K means: {max_k_means}, {n_feats_max_k_means} features ')

k_best_14 = SelectKBest(f_classif, k=14).fit(X, y_red)
k_best_8 = SelectKBest(f_classif, k=8).fit(X, y_red)

# testing for each algorithm the number of clusters that it prefers


k_means_purities, k_means_precisions, k_means_recalls, k_means_f_measures, k_means_rand_indexes, k_means_losses = [], [], [], [], [], []
agl_purities, agl_precisions, agl_recalls, agl_f_measures, agl_rand_indexes = [], [], [], [], []
spect_purities, spect_precisions, spect_recalls, spect_f_measures, spect_rand_indexes = [], [], [], [], []

for n_clusters in range(2, 18):
    k_means_purity, k_means_precision, k_means_recall, k_means_f_measure, k_means_support, k_means_rand_index, k_means_loss = train_k_means(
        KMeans(n_clusters=n_clusters, n_init='auto', random_state=0),
        k_best_14.transform(result),
        y_red, y != 0)
    k_means_purities.append(k_means_purity)
    k_means_precisions.append(k_means_precision)
    k_means_recalls.append(k_means_recall)
    k_means_f_measures.append(k_means_f_measure)
    k_means_rand_indexes.append(k_means_rand_index)
    k_means_losses.append(k_means_loss)

    agl_purity, agl_precision, agl_recall, agl_f_measure, agl_support, agl_rand_index = train_model(
        AgglomerativeClustering(n_clusters=n_clusters),
        k_best_8.transform(result),
        y_red, y != 0)

    agl_purities.append(agl_purity)
    agl_precisions.append(agl_precision)
    agl_recalls.append(agl_recall)
    agl_f_measures.append(agl_f_measure)
    agl_rand_indexes.append(agl_rand_index)

    spect_purity, spect_precision, spect_recall, spect_f_measure, spect_support, spect_rand_index = train_model(
        SpectralClustering(n_clusters=n_clusters, assign_labels='cluster_qr'),
        k_best_8.transform(result),
        y_red, y != 0)

    spect_purities.append(spect_purity)
    spect_precisions.append(spect_precision)
    spect_recalls.append(spect_recall)
    spect_f_measures.append(spect_f_measure)
    spect_rand_indexes.append(spect_rand_index)

# plotting the data for every single model

plt.figure()
plt.title('K Means')
plt.xlabel('Number of clusters')
plt.ylim(0, 1)
plt.plot(range(2, 18), k_means_purities, label='purity', color='blue')
plt.plot(range(2, 18), k_means_precisions, label='precision', color='green')
plt.plot(range(2, 18), k_means_recalls, label='recall', color='red')
plt.plot(range(2, 18), k_means_f_measures, label='f_measure', color='orange')
plt.plot(range(2, 18), k_means_rand_indexes, label='rand_index', color='purple')
plt.legend()
plt.savefig('pictures/k_means')
plt.close()

plt.figure()
plt.title('K Means loss')
plt.xlabel('Number of clusters')
plt.plot(range(2, 18), k_means_losses, label='loss', color='blue')
plt.legend()
plt.savefig('pictures/k_means_loss')
plt.close()

plt.figure()
plt.title('Aglomerative Clustering')
plt.xlabel('Number of clusters')
plt.ylim(0, 1)
plt.plot(range(2, 18), agl_purities, label='purity', color='blue')
plt.plot(range(2, 18), agl_precisions, label='precision', color='green')
plt.plot(range(2, 18), agl_recalls, label='recall', color='red')
plt.plot(range(2, 18), agl_f_measures, label='f_measure', color='orange')
plt.plot(range(2, 18), agl_rand_indexes, label='rand_index', color='purple')
plt.legend()
plt.savefig('pictures/aglomerative_clustering')
plt.close()

plt.figure()
plt.title('Spectral Clustering')
plt.xlabel('Number of clusters')
plt.ylim(0, 1)
plt.plot(range(2, 18), spect_purities, label='purity', color='blue')
plt.plot(range(2, 18), spect_precisions, label='precision', color='green')
plt.plot(range(2, 18), spect_recalls, label='recall', color='red')
plt.plot(range(2, 18), spect_f_measures, label='f_measure', color='orange')
plt.plot(range(2, 18), spect_rand_indexes, label='rand_index', color='purple')
plt.legend()
plt.savefig('pictures/spectral_clustering')
plt.close()

# evaluating spectral clustering with different neighbourhood values
purities, precisions, recalls, f_measures, rand_indexes = [], [], [], [], []
for neighbourhood in range(1, 200):
    purity, precision, recall, f_measure, support, rand_index = train_model(
        SpectralClustering(n_clusters=3, assign_labels='cluster_qr', n_neighbors=neighbourhood),
        k_best_8.transform(result),
        y_red, y != 0)
    purities.append(purity)
    precisions.append(precision)
    recalls.append(recall)
    f_measures.append(f_measure)
    rand_indexes.append(rand_index)

plt.figure()
plt.title('Spectral Clustering with different Neighbourhoods')
plt.xlabel('Neighbourhood')
plt.ylim(0, 1)
plt.plot(range(1, 200), purities, label='purity', color='blue')
plt.plot(range(1, 200), precisions, label='precision', color='green')
plt.plot(range(1, 200), recalls, label='recall', color='red')
plt.plot(range(1, 200), f_measures, label='f_measure', color='orange')
plt.plot(range(1, 200), rand_indexes, label='rand_index', color='purple')
plt.legend()
plt.savefig('pictures/spectral_clustering_neighbourhoods')
plt.close()

labels = AgglomerativeClustering(n_clusters=4).fit_predict(k_best_14.transform(result))
report_clusters(mat[:, 0], labels, 'aglomerative_clustering.html')

labels = KMeans(n_clusters=3, n_init='auto', random_state=0).fit_predict(k_best_8.transform(result))
report_clusters(mat[:, 0], labels, 'k_means_clustering.html')

labels = SpectralClustering(n_clusters=3, assign_labels='cluster_qr', n_neighbors=10).fit_predict(
    k_best_8.transform(result))
report_clusters(mat[:, 0], labels, 'spectral_clustering.html')

# testing for affinity propagation the damp it prefers

best_damp, best_features_af = 0, 0
for damp in np.arange(0.5, 1.0, 0.05):
    afin = AffinityPropagation(damping=damp, random_state=None)
    af_labels = afin.fit_predict(k_best_8.transform(result))
    af_labels_test = af_labels[y != 0]
    adj = adjusted_rand_score(y_red, af_labels_test)
    if (adj > best_damp):
        best_damp = damp
        best_features_af = adj

predict_final_af = AffinityPropagation(damping=best_damp, random_state=None).fit_predict(k_best_8.transform(result))
report_clusters(mat[:, 0], predict_final_af, 'affinity_propagation.html')
