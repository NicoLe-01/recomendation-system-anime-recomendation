# Laporan Proyek Machine Learning - Nico Siahaan

## Project Overview
Proyek ini merupakan proyek Rekomendasi Anime. Latar belakang proyek ini adalah pengalaman penulis saat mencari anime yang sesuai dengan selera. Dimana penulis cukup kesulitan dalam mencari anime yang sesuai dengan genre yang disukai. Maka dari itu penulis memiliki ide untuk membuat sistem rekomendasi anime. Agar penulis atau orang lain dapat mencari anime yang sesuai dengan genre yang disukai. Dari sudut pandang penyedia anime/distributor anime juga memiliki manfaat seperti dapat meningkatkan jumlah pengunjung/click rate pada penyedia anime tersebut.
Hal ini juga pernah diteliti oleh [Zartesya, M. A., & Komalasari, D. (2021)](https://conference.upnvj.ac.id/index.php/senamika/article/view/1343)[1], yang mana Zartesya menggunakan metode Collaborative Filtering, PCA dan K-Means proyek tersebut.
[SA Pratama - 2019](https://repository.upnvj.ac.id/2016/1/AWAL.pdf)[2], juga meneliti tentang hal ini, SA Pratama menggunakan metode Decision Tree pada proyeknya.

## Business Understanding
Bayangkan terdapat seorang yang baru menonton suatu judul anime, lalu dia tertarik untuk menonton anime yang sejenis dengan anime yang ditonton sebelumnya. Lalu dia mencoba mencari aplikasi yang penyedia jasa streaming anime. Nah, dengan adanya sistem rekomendasi anime ini, pengguna baru tersebut dapat melihat anime yang populer atau pengguna dapat membuat akun, lalu membuat daftar anime yang ditonton. 
Sehingga dari sudut pandang pengguna, pengguna merasa terbantu dengan rekomedasi yang diberikan sistem. Sedangkan dari sudut pandang penyedia, penyedia mendapat click rate/trafic yang meningkat.

### Problem Statements

Menjelaskan pernyataan masalah:
- Data apa yang terpenting dalam membuat sistem rekomendasi anime ?
- Bagaimana cara membuat sistem rekomendasi anime yang cocok untuk pengguna ?
- Metode apa yang paling efektif untuk sistem rekomendasi anime ?

### Goals
- Untuk mengetahui data yang terpenting dalam membuat sistem rekomendasi anime.
- Membuat model machine learning untuk sistem rekomendasi.
- Untuk mengetahui metode yang paling efektif dalam sistem rekomendasi anime.

### Solution statements
 - Metode machine learning yang dibuat akan menggunakan Content-Based Filtering dan Collaborative Filtering dengan pendekatan deep learning.


## Data Understanding
Dataset proyek ini merupakan dataset yang diambil dari [Kaggle](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)

Dataset ini berisi 2 file csv, Anime.csv dan Rating.csv. Pada Anime.csv berisi 12.294 judul anime. Sedangkan Rating.csv berisi 73,516 rating pengguna.
Dataset ini masih memiliki cukup banyak *missing value*

Variabel-variabel pada Anime.csv adalah sebagai berikut:
- anime_id  : nilai *unique* suatu dari judul anime
- name      : nama anime
- genre     : genre anime
- type      : jenis anime (movie, TV, OVA, lainnya)
- episodes  : jumlah episode
- rating    : rating rata-rata pengguna
- members   : komunitas pada suatu judul anime

Variabel-variabel pada Rating.csv adalah sebagai berikut:
- user_id   : nilai *unique* seorang pengguna
- anime_id  : anime yang di nilai/rating oleh pengguna
- rating    : rating yang diberikan pengguna. -1 artinya belum memberikan rating

Untuk visualisasi data pada dataset ini penulis menggunakan library wordCloud, yang digunakan untuk mengetahui genre apa yang paling banyak pada dataset anime.

## Data Preparation
### Content-Based Filtering
Tahapan : <br> 

Melihat kolom/fitur pada anime.csv
```
animes.head()
```
|   	| anime_id 	|               name               	|                       genre                       	|  type 	| episodes 	| rating 	| members 	|
|---	|:--------:	|:--------------------------------:	|:-------------------------------------------------:	|:-----:	|:--------:	|:------:	|:-------:	|
| 1 	|    32281 	|                   Kimi no Na wa. 	|              Drama, Romance, School, Supernatural 	| Movie 	|        1 	|   9.37 	|  200630 	|
| 2 	|     5114 	| Fullmetal Alchemist: Brotherhood 	| Action, Adventure, Drama, Fantasy, Magic, Mili... 	|    TV 	|       64 	|   9.26 	|  793665 	|
| 3 	|    28977 	|                         Gintama° 	| Action, Comedy, Historical, Parody, Samurai, S... 	|    TV 	|       51 	|   9.25 	|  114262 	|
| 4 	|     9253 	|                      Steins;Gate 	|                                  Sci-Fi, Thriller 	|    TV 	|       24 	|   9.17 	|  673572 	|
| 5 	|     9969 	|                    Gintama&#039; 	| Action, Comedy, Historical, Parody, Samurai, S... 	|    TV 	|       51 	|   9.16 	|  151266 	|

Terdapat 7 kolom/fitur pada anime.csv

Melihat apakah terdapat nilai null pada dataset.
```
animes.isnull().sum()
```
| anime_id 	| 0   	|
|----------	|-----	|
| name     	| 0   	|
| genre    	| 62  	|
| type     	| 25  	|
| episodes 	| 0   	|
| rating   	| 230 	|
| members  	| 0   	|

Terdapat cukup banyak nilai null pada dataset. 62 pada kolom genre, 25 pada type, 230 pada rating.

Hapus nilai null.
```
animes.dropna(inplace=True)
```
| anime_id 	| 0 	|
|----------	|---	|
| name     	| 0 	|
| genre    	| 0 	|
| type     	| 0 	|
| episodes 	| 0 	|
| rating   	| 0 	|
| members  	| 0 	|

Dapat dilihat dataset sudah bersih dari nilai null.

Melihat pesebaran genre yang ada pada dataset.
```
from collections import defaultdict
all_genres = defaultdict(int)
for genres in animes['genre']:
    for genre in genres.split(','):
        all_genres[genre.strip()] += 1

def wordCloud(words):
    wordCloud = WordCloud(width=1000, height=800, background_color='white').generate_from_frequencies(words)

    plt.imshow(wordCloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
wordCloud(all_genres)
```
![image](https://user-images.githubusercontent.com/64530694/189825520-3f3022e0-22e9-4f53-a5cf-8fdcd5488ded.png)

Dapat dilihat pesebaran genre anime didominasi oleh genre Comedy, Action, dan Adventure.

Pada bagian ini penulis membuat visualisasi untuk melihat pesebaran genre yang ada pada dataset, yaitu dengan menggunakan library wordCloud

### Collaborative Filtering
Tahapan :\
Cek data rating
```
ratings.head()
```
|   	| user_id 	| anime_id 	| rating 	|
|--:	|--------:	|---------:	|-------:	|
| 0 	|       1 	|       20 	|     -1 	|
| 1 	|       1 	|       24 	|     -1 	|
| 2 	|       1 	|       79 	|     -1 	|
| 3 	|       1 	|      226 	|     -1 	|
| 4 	|       1 	|      241 	|     -1 	|

Terdapat data rating -1 pada dataset. -1 artinya adalah nilai null.

Hilangkan nilai -1 (null) dari kolom rating.
```
mask = (ratings['rating'] == -1)
ratings = ratings.loc[~mask]
```

Cek nilai lagi.
|  user_id 	| 0.0 	|
|---------:	|-----	|
| anime_id 	| 0.0 	|
|  rating  	| 0.0 	|

Ambil 1000 data rating.
```
ratings = ratings[ratings['user_id'] < 1000]
```

Ambil nilai unique user_id dan anime_id.
```
userid_unique = ratings['user_id'].nunique()
anime_unique = ratings['anime_id'].nunique()
```
|  user unique : 	| 940  	|
|---------------:	|------	|
| anime unique : 	| 4510 	|

Bagi data menjadi train dan validation. Train set 80% dan Validation set 20%
```
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=22)
```

## Modeling
### Content-Based Filtering
Untuk metode ini, penulis menggunakan TfidfVectorizer dan cosine similarity.\
Pertama import library TfidVectorizer dan cosine_similarity dari library sklearn

Lalu hitung idf dengan TfidVectorizer.fit untuk kolom genre

Ubah matriks tfid menjadi vektor tf-idf dengan TfidVectorizer.fit_transform lalu ubah dengan funsi todense()

Dengan menggunakan library cosine_similarity, hitung cosine similarity dari matrix tf-idf

Lalu buat dataframe dengan baris dan kolom nama anime

Buat fungsi rekomendasi
```
def anime_recommendation(nama_anime, similarity_data=cosine_sim_df, items=animes[['name', 'genre']], k=5):

# Parameter 
# nama_anime :  nama anime yang ingin dicari
# similarity_data : matriks kesamaan anime
# items : fitur kemiripan
# k : banyak rekomendasi yang diinginkan
```

Hasil Prediksi
```
anime_recommendation('Sword Art Online', k=5)
```
|   	|               name               	|                   genre                   	|
|--:	|:--------------------------------:	|:-----------------------------------------:	|
| 0 	| Sword Art Online II              	| Action, Adventure, Fantasy, Game, Romance 	|
| 1 	| Sword Art Online: Extra Edition  	| Action, Adventure, Fantasy, Game, Romance 	|
| 2 	| Sword Art Online II: Debriefing  	| Action, Adventure, Fantasy, Game          	|
| 3 	| Bakugan Battle Brawlers          	| Action, Fantasy, Game                     	|
| 4 	| Monster Strike: Mermaid Rhapsody 	| Action, Fantasy, Game                     	|

Kelebihan :
- Tidak memerlukan data riwayat penguna
- Proses yang cepat
- Implementasinya mudah

Kekurangan :
- Rekomendasi yang ditampikan cukup terbatas. Hanya menampilkan genre yang sesuai dengan pengguna
- Rekomendasi yang ditampilkan tidak menggunakan rating. Hanya mencocokan genre yang diminta.

### Collaborative Filtering
Untuk metode ini, penulis menggunakan pendekatan deep learning.

Buat model
```
def RecommenderAnime(n_users, n_movies, n_dim):
# Parameter 
# n_user = banyaknya jumlah dimensi untuk layer user
# n_moview = banyaknya jumlah dimensi untuk layer anime
# n_dim = output dimension
```

Inisiasi model dengan nilai unique dari user_id dan anime. Lalu traning traning model denga model.fit


Fungsi prediksi
```
def buat_prediksi(user_id, anime_id, model):
# Parameter 
# user_id : id user yang ingin diprediksi
# anime_id : id anime yang ingin diprediksi
# model : model yang sudah di traning
```

Fungsi untuk menampilkan rekomendasi
```
def prediksi_teratas(user_id, model, k):
  """
  Parameter user_id : input user ke n dari data set
  Parameter model : input model yang telah di traning
  Parameter k : input berapa banyak prediksi yang akan tampilkan
  """
```

Hasil prediksi
7 genre yang banyak ditonton oleh pengguna : 

['Comedy', 'Adventure', 'Sci-Fi', 'Action', 'Fantasy', 'Shounen', 'Drama']
|          name         	|  type 	| rating_predict 	|                       genre                       	|
|:---------------------:	|:-----:	|:--------------:	|:-------------------------------------------------:	|
| Little Nemo           	| Movie 	| 9.98           	| Adventure, Fantasy                                	|
| TO-Y                  	| OVA   	| 9.86           	| Drama, Music, Shounen                             	|
| Yami no Matsuei       	| TV    	| 9.82           	| Comedy, Drama, Fantasy, Horror, Magic, Shoujo,... 	|
| Zettai Karen Children 	| TV    	| 9.80           	| Action, Comedy, Shounen, Supernatural             	|
| Tattoon Master        	| OVA   	| 9.78           	| Adventure, Comedy, Slice of Life, Supernatural    	|


Kelebihan : 
- Rekomendasi yang diberikan berdasarkan rating yang paling tinggi dan sesuai dengan genre.

Kekurangan :
- Proses traning yang cukup lama
- Implementasi cukup sulit

## Evaluation
### Conten-Based Filterring
Metrik evaluasi yang digunakan adalah menggunakan Jaccard Similarity yang formula nya :
    
    
Dengan :
A = Genre/nilai sebenarnya
B = Genre/nilai prediksi

Dengan prediksi :
|   	|               name               	|                   genre                   	|
|--:	|:--------------------------------:	|:-----------------------------------------:	|
| 0 	| Sword Art Online II              	| Action, Adventure, Fantasy, Game, Romance 	|
| 1 	| Sword Art Online: Extra Edition  	| Action, Adventure, Fantasy, Game, Romance 	|
| 2 	| Sword Art Online II: Debriefing  	| Action, Adventure, Fantasy, Game          	|
| 3 	| Bakugan Battle Brawlers          	| Action, Fantasy, Game                     	|
| 4 	| Monster Strike: Mermaid Rhapsody 	| Action, Fantasy, Game                     	|

Jaccard Similarity menghitung kesamaan antara dua set. 
Jadi, misalnya terdapat set A {1, 2, 3} dan set B {2, 3, 4}. Maka jaccard similarity akan menghitung apakah ada kesamaan antara dua set diatas. Dapat dilihat terdapat 1 kesamaan nilai pada set, yaitu nilai 2. Maka perhitungannya adalah :\
J = |{2}| / |{1, 2, 3, 4}| \
= 1 / 4 \
= 0.25\
Maka nilai Jaccard Similarity untuk set A, B adalah 0.25

Untuk nilai Jaccard Similarity prediksi didapatkan melalui perhitungan genre anime dengan genre anime yang disarankan.\
genre_anime = {'Action', 'Adventure', 'Fantasy', 'Game', 'Romance'}\
genre_prediksi = {'Action', 'Adventure', 'Fantasy', 'Game', 'Romance'}\
J[1] = |{'Action', 'Adventure', 'Fantasy', 'Game', 'Romance'}| / |{'Action', 'Adventure', 'Fantasy', 'Game', 'Romance'}|\
= 5 / 5 \
= 1 \
Perhitungan akan dilakukan terus sampai prediksi ke-n lalu di rata-ratakan.
Sehingga didapatkan score similarity sebesar 0.8 untuk prediksi anime "Sword Art Online", yang artinya sistem rekomendasi bekerja cukup baik.


### Collaborative Filtering
Metrik evaluasi yang digunakan adalah menggunakan Mean Squared Error, yang formulanya :\
<br>
![image](https://user-images.githubusercontent.com/64530694/189574922-e5cc96b1-30fe-4db7-b508-13aaf8c85c46.png)

Dengan :
- At = Nilai sebenarnya
- Ft = Nilai prediksi
- n = Banyaknya data

Nilai MSE ini adalah Mean Squared Error, yang mana merupakan perbandingan nilai sebenarnya dengan nilai prediksi lalu dirata-ratakan. 

Dari hasil Traning model, model mendapatkan nilai MSE 0.89 pada training sedangkan pada validation mendapatkan nilai MSE 1.53.\
<br>
![image](https://user-images.githubusercontent.com/64530694/189904751-7c605f87-1bd9-4494-9fed-1b9d5974d8fb.png)\

Artinya, model kurang baik bekerja pada data validation. Akan tetapi model masih dapat memberikan rekomendasi yang cukup relevean kepada pengguna.


Hasil prediksi nya adalah :
```
prediksi_teratas(200, model, 5)
```
7 genre yang banyak ditonton oleh pengguna : 

['Comedy', 'Adventure', 'Sci-Fi', 'Action', 'Fantasy', 'Shounen', 'Drama']
|          name         	|  type 	| rating_predict 	|                       genre                       	|
|:---------------------:	|:-----:	|:--------------:	|:-------------------------------------------------:	|
| Little Nemo           	| Movie 	| 9.98           	| Adventure, Fantasy                                	|
| TO-Y                  	| OVA   	| 9.86           	| Drama, Music, Shounen                             	|
| Yami no Matsuei       	| TV    	| 9.82           	| Comedy, Drama, Fantasy, Horror, Magic, Shoujo,... 	|
| Zettai Karen Children 	| TV    	| 9.80           	| Action, Comedy, Shounen, Supernatural             	|
| Tattoon Master        	| OVA   	| 9.78           	| Adventure, Comedy, Slice of Life, Supernatural    	|


```
prediksi_teratas(34, model, 5)
```

7 genre yang banyak ditonton oleh pengguna : 

'Action', 'Comedy', 'Adventure', 'Sci-Fi', 'Fantasy', 'Shounen', 'Romance'
|   	|                      name                      	|  type 	| rating_predict 	|                  genre                  	|
|--:	|:----------------------------------------------:	|:-----:	|:--------------:	|:---------------------------------------:	|
| 0 	| PetoPeto-san                                   	| TV    	| 10.24          	| Comedy, Fantasy, Romance, School        	|
| 1 	| Little Nemo                                    	| Movie 	| 10.11          	| Adventure, Fantasy                      	|
| 2 	| Aria The Animation                             	| TV    	| 10.10          	| Fantasy, Sci-Fi, Shounen, Slice of Life 	|
| 3 	| TO-Y                                           	| OVA   	| 10.10          	| Drama, Music, Shounen                   	|
| 4 	| Wellber no Monogatari: Sisters of Wellber Zwei 	| TV    	| 9.99           	| Adventure, Fantasy, Historical, Romance 	|


```
prediksi_teratas(45, model, 5)
```
7 genre yang banyak ditonton oleh pengguna : 

'Comedy', 'Adventure', 'Fantasy', 'Action', 'Shounen', 'Sci-Fi', 'Drama'
|   	|          name         	| type 	| rating_predict 	|                    genre                   	|
|:-:	|:---------------------:	|:----:	|:--------------:	|:------------------------------------------:	|
| 0 	| Penguin Musume♥Heart  	| ONA  	| 9.88           	| Comedy, Ecchi, School, Slice of Life       	|
| 1 	| PetoPeto-san          	| TV   	| 9.83           	| Comedy, Fantasy, Romance, School           	|
| 2 	| The Urotsuki          	| OVA  	| 9.79           	| Adventure, Demons, Fantasy, Hentai, Horror 	|
| 3 	| Batman: Gotham Knight 	| OVA  	| 9.61           	| Action, Adventure, Martial Arts            	|
| 4 	| Prince of Tennis      	| TV   	| 9.49           	| Action, Comedy, School, Shounen, Sports    	|

<br>


## Kesimpulan 
Jadi kesimpulan yang didapat dari proyek ini adalah : 
- Data yang terpenting dalam membuat sistem rekomendasi anime adalah Genre dan Rating dari suatu anime.
- Dengan membuat sistem rekomendasi menggukanan machine learning, baik dengan Content-Based Filtering atau Collaborative Filtering.
- Kedua metode memiliki kelebihan dan kekurangan masing-masing. Seperti yang dijelaskan pada bagian Modelling, dengan pendekatan Content-Based Filtering bagus untuk pengguna baru yang belum memiliki riwayat menonton anime. Sedangkan Collaborative Filtering bagus untuk pengguna yang sudah memiliki riwayat menonton anime.


## Referensi
[1] Zartesya, M. A., & Komalasari, D. (2021). Penerapan Collaborative Filtering, PCA dan K-Means dalam Pembangunan Sistem Rekomendasi Ongoing dan Upcoming Film Animasi Jepang. Senamika, 2(1), 606-615.

[2] Pratama, S. A. (2019). PERANCANGAN SISTEM REKOMENDASI ANIME MENGGUNAKAN METODE DECISION TREE PADA INDUSTRI KREATIF (Doctoral dissertation, Universitas Pembangunan Nasional Veteran Jakarta).

