import pandas as pd
import numpy as np
import glob, time, os
from sklearn import cluster
from scipy.misc import imread
from skimage import transform as sk_transform
import skimage.measure as sm
# import progressbar
import multiprocessing
import random

img_shape=(40,40)

def _load_img(file, img_shape=(224, 224)):
    shape = list(img_shape) + [3]
    img = imread(file)
    img = sk_transform.resize(img, shape, preserve_range=True)
    img = np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)
    # Convert to BGR from RGB
    img = img[::-1, :, :].astype(np.float32)
    # deduct mean value for VGG
    MEAN_VALUE = np.array([103.939, 116.779, 123.68])   # BGR
    for c in range(3):
        img[c, :, :] -= MEAN_VALUE[c]
    return img

def load_train_data(img_shape=(224,224), rotate=False, display=False):
    X = []
    X_id = []
    y = []
    start_time = time.time()

    print('Loading training images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
        index = folders.index(fld)
        print('Load folder {} (Index: {})'.format(fld, index))
        path = os.path.join('..', 'input', 'train', fld, '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = _load_img(fl, img_shape)
            X.append(img)
            X_id.append(flbase)
            y.append(fld)

    X = np.array(X).reshape(len(X), -1, img_shape[0], img_shape[1])
    X = X.astype(np.float32)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X, y, X_id

def load_test_data(img_start_ix=0, max_img=100, img_shape=(224,224)):
    path = os.path.join('..', 'input', 'test_stg1', '*.jpg')
    files = sorted(glob.glob(path))

    files = files[img_start_ix:img_start_ix+max_img]

    X = []
    X_id = []
    for fl in files:
        flbase = os.path.basename(fl)
        img = _load_img(fl, img_shape)
        X.append(img)
        X_id.append(flbase)

    X = np.array(X).reshape(len(X), -1, img_shape[0], img_shape[1])
    X = X.astype(np.float32)

    return X, X_id

# Function for computing distance between images
def compare(args):
    img, img2 = args
    img = (img - img.mean()) / img.std()
    img2 = (img2 - img2.mean()) / img2.std()
    return np.mean(np.abs(img - img2))

train, folder_ids, train_ids = load_train_data(img_shape=img_shape)
#train, folder_ids, train_ids = train[:500], folder_ids[:500], train_ids[:500]

print('Sizes of train:')
print(train.shape)

# Create the distance matrix in a multithreaded fashion
pool = multiprocessing.Pool(12)

distances = np.zeros((len(train), len(train)))
for i, img in enumerate(train):
    if i%100 == 0:
        print(i)
    all_imgs = [(img, f) for f in train]
    dists = pool.map(compare, all_imgs)
    distances[i, :] = dists

cls = cluster.DBSCAN(metric='precomputed', min_samples=20, eps=0.6)
clust_ids = cls.fit_predict(distances)
print('Cluster sizes:')
print(pd.Series(clust_ids).value_counts())

train_clusters = pd.DataFrame({"id": train_ids, "cluster": clust_ids, "folder": folder_ids})
train_clusters = train_clusters[["id","cluster","folder"]].sort(["id"])
print(train_clusters)
train_clusters.to_csv('../output/train_clusters.csv', index=False)

print("Reading test data...")
test, test_ids = load_test_data(0, 1000, img_shape=img_shape)
print('Sizes of test:')
print(test.shape)

cluster_means = []
for i in range(-1, np.max(train_clusters['cluster'])+1):
    cluster_imgs = np.array(train)[clust_ids == i]
    print(cluster_imgs.shape)
    cluster_means.append(np.mean(cluster_imgs, axis=0))

cluster_means = np.array(cluster_means).reshape(len(cluster_means), -1, img_shape[0], img_shape[1])
print("cluster means shape:")
print(cluster_means.shape)

print("Calculating distances from test images to cluster means...")
distances = np.zeros((len(test), len(cluster_means)))
for i, img in enumerate(test):
    if i%100 == 0:
        print(i)
    all_imgs = [(img, clus_mean) for clus_mean in cluster_means]
    dists = pool.map(compare, all_imgs)
    distances[i, :] = dists

clust_ids = np.argmin(distances, axis=1)-1
print('Test cluster sizes:')
print(pd.Series(clust_ids).value_counts())
test_clusters = pd.DataFrame({"id": test_ids, "cluster": clust_ids})
test_clusters = test_clusters[["id","cluster"]].sort(["id"])
test_clusters.to_csv('../output/test_clusters.csv', index=False)
