# Unsupervised Learning and CNN on Tous Jewelry Store README

### Authors
Murtadha


### Date

6th-May-2019

### Website

n/a

### License

GNU General Public License (version 3)-- GPLv3

---


### Problem Statement


In this project we try to analyze the a sample of products sold by Tous jewelry store. Only best selling items are considered here. We want to investigate how best-selling items on the Tous store are clustered. What features make sense of the clustering? How can we interpret the clusters meaningfully. We want to analyze the dataset using unsupervised learning algorithms. Performing Exploratory Data Analysis is a prerequisite for dimension reduction and model selection. The sample consists of nearly 300 items along with some of their details such wearing type, stone type, and price. Some items are labeled as best seller items, whole most of them are not.

The extra part of this project aims to predict the class of the items, whether it's best-selling or not, using the item's image. Moreover, we want to predict the price of the items using its price tag.

For more information:

- [Tous Jewelry Store - Saudi Arabai](https://www.tous.com/sa-en/)

- [Kaggle pagge of the dataset](https://www.kaggle.com/arimaha/what-makes-a-best-seller-item)

---


### Executive Summary

The first step is to import the data and clean it. In general, the data is clean except for one column values. We performed the required iterations to make sure all columns are assigned to the right data type. Next we counted the number of rows having a positive value in the best_seller column, and based on this we created a new data frame to have only those rows.

The second step is to perform some EDA. We run the regular EDA functions on the whole data frame, investigating the the values of the central tendency measures for each column. A heatmap graph of all features is given. At the end of this section we decided to take off some irrelevant features that have zero correlation with data, and a bar plot is given.

The third step is dimension reduction. As the number of features of the dataset is large and the data is spread, we wanted ti investigate how the data features vary and what is the best number of variance features to use. For this step we used PCA method.

The fourth step is to use an unsupervised learning algorithm. For this step we decided to use k-means algorithm. Prior to applying the algorithm on the selected PCA features, we plotted the distortion values compared to different number of clusters; the elbow method was used to decide on the best number of clusters.

Finally, we generated all possible cross tables to study how each single feature could be used to interpret the resulted clustering. We concluded that only the price feature could be used to interpret the clustering.

As for the extra part of the project, we started by retrieving images of the items from the column that has images link. We divided the images into its respected class folders, with the goal of applying Keras flow_from_directory; however, we found that applying Keras flow_from_dataframe is better fitting our needs. After processing the images using OpenCV and manually using Gimp, we feed the data to the CNN model. We provide two CNN models, one for classification and one for regression. Though the results seem to be mediocre, it was an enriching for learning purposes.

---

### Data Source


This data was scrapped and published on Kaggle on 2019-04-18. The last version was updated on 2019-05-03, of which this code is based. [Link to the dataset Kaggle page.](https://www.kaggle.com/arimaha/what-makes-a-best-seller-item)

---

### Data Dictionary



| Feature     | Type    | Dataset | Description                                |
|-------------|---------|---------|--------------------------------------------|
| best_seller | float64 | data    | 1 if this item is best-seller, otherwise 0 |
| price       |  object | data    | price of the item                          |
| image       |  object | data    | link to the image of the item              |
| bracelet    | float64 | data    | 1 if this item bracelet, otherwise 0       |
| ring        | float64 | data    | 1 if this item ring, otherwise 0           |
| necklace    | float64 | data    | 1 if this item necklace, otherwise 0       |
| earring     | float64 | data    | 1 if this item earring, otherwise 0        |
| choker      | float64 | data    | 1 if this item choker, otherwise 0         |
| pendant     | float64 | data    | 1 if this item pendant, otherwise 0        |
| mesh        | float64 | data    | 1 if this item mesh, otherwise 0           |
| gold        | float64 | data    | 1 if this item gold, otherwise 0           |
| silver      | float64 | data    | 1 if this item silver, otherwise 0         |
| white       | float64 | data    | 1 if this item white, otherwise 0          |
| titanium    | float64 | data    | 1 if this item titanium, otherwise 0       |
| ip          | float64 | data    | 1 if this item ip, otherwise 0             |
| rose        | float64 | data    | 1 if this item rose, otherwise 0           |
| steel       | float64 | data    | 1 if this item steel, otherwise 0          |
| oxidized    | float64 | data    | 1 if this item oxidized, otherwise 0       |
| mat_sum     | float64 | data    | 1 if this item mat_sum, otherwise 0        |
| pearl       | float64 | data    | 1 if this item pearl, otherwise 0          |
| chalcedony  | float64 | data    | 1 if this item chalcedony, otherwise 0     |
| rhodonite   | float64 | data    | 1 if this item rhodonite, otherwise 0      |
| opal        | float64 | data    | 1 if this item opal, otherwise 0           |
| beryllium   | float64 | data    | 1 if this item beryllium, otherwise 0      |
| morganite   | float64 | data    | 1 if this item morganite, otherwise 0      |
| citrine     | float64 | data    | 1 if this item citrine, otherwise 0        |
| topaz       | float64 | data    | 1 if this item topaz, otherwise 0          |
| praseolite  | float64 | data    | 1 if this item praseolite, otherwise 0     |
| carnelian   | float64 | data    | 1 if this item carnelian, otherwise 0      |
| amethyst    | float64 | data    | 1 if this item amethyst, otherwise 0       |
| malachite   | float64 | data    | 1 if this item malachite, otherwise 0      |
| amazonite   | float64 | data    | 1 if this item amazonite, otherwise 0      |
| labradorite | float64 | data    | 1 if this item labradorite, otherwise 0    |
| sapphires   | float64 | data    | 1 if this item sapphires, otherwise 0      |
| agate       | float64 | data    | 1 if this item agate, otherwise 0          |
| quartzite   | float64 | data    | 1 if this item quartzite, otherwise 0      |
| gemstones   | float64 | data    | 1 if this item gemstones, otherwise 0      |
| spinel      | float64 | data    | 1 if this item spinel, otherwise 0         |
| onyx        | float64 | data    | 1 if this item onyx, otherwise 0           |
| gem_sum     | float64 | data    | 1 if this item gem_sum, otherwise 0        |
| diamonds    | float64 | data    | 1 if this item diamonds, otherwise 0       |
| ruby        | float64 | data    | 1 if this item ruby, otherwise 0           |
| cry_sum     | float64 | data    | 1 if this item cry_sum, otherwise 0        |


---

### Observations/Conclusions

Based on the analysis performed on the dataset, specifically on the best-selling items, we found that only the price feature makes sense of the clustering. Each cluster has items in a distinct price range. However, the other features' values are spread across the clusters.
