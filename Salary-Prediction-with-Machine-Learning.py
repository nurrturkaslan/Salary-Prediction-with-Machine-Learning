# 1. İş Problemi
# Maaş bilgileri ve 1986 yılına ait kariyer istatistikleri paylaşılan beyzbol oyuncularının maaş tahminleri için bir makine öğrenmesi projesi gerçekleştirilebilir mi?

# 2. Veri Seti Hikayesi
#Bu veri seti orijinal olarak Carnegie Mellon Üniversitesi'nde bulunan StatLib kütüphanesinden alınmıştır.
# Veri seti 1988 ASA Grafik Bölümü Poster Oturumu'nda kullanılan verilerin bir parçasıdır.
# Maaş verileri orijinal olarak Sports Illustrated, 20 Nisan 1987'den alınmıştır.
# 1986 ve kariyer istatistikleri, Collier Books, Macmillan Publishing Company edilmiştir.

# 3. Değişkenler
# AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
# Hits: 1986-1987 sezonundaki isabet sayısı
# HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı
# Runs: 1986-1987 sezonunda takımına kazandırdığı sayı
# RBI: Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı
# Walks: Karşı oyuncuya yaptırılan hata sayısı
# Years: Oyuncunun major liginde oynama süresi (sene)
# CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı
# CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
# CHmRun: Oyucunun kariyeri boyunca yaptığı en değerli sayısı
# CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
# CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
# CWalks: Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
# League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
# Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
# PutOuts: Oyun icinde takım arkadaşınla yardımlaşma
# Assits: 1986-1987 sezonunda oyuncunun yaptığı asist sayısı
# Errors: 1986-1987 sezonundaki oyuncunun hata sayısı
# Salary: Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
# NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör

from Helpers.eda import *
from Helpers.data_prep import *
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score, confusion_matrix, classification_report, plot_roc_curve

df = pd.read_csv("Weeks/WEEK_07/hitters.csv")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    print(col, check_outlier(df, col))

# aykırı değerleri baskıla
for col in num_cols:
    replace_with_thresholds(df, col)

# baskıladıktan sonra kontrol
for col in num_cols:
    print(col, check_outlier(df, col))

# eksik değer kontrolü
df.isnull().values.any()
df.dropna(inplace = True)
df.head()

df.isnull().values.any() #oh miss :)

# feature engineering

df["NEW_HITS"] = df["Hits"] / df["CHits"]
df["NEW_RBI"] = df["RBI"] / df["CRBI"]
df["NEW_WALKS"] = df["Walks"] / df["CWalks"]
df["NEW_CAT_BAT"] = df["CAtBat"] / df["Years"]
df["NEW_CRUNS"] = df["CRuns"] / df["Years"]
df["NEW_CHITS"] = df["CHits"] / df["Years"]
df["NEW_CHMRUN"] = df["CHmRun"] / df["Years"]
df["NEW_CRBI"] = df["CRBI"] / df["Years"]
df["NEW_CWALKS"] = df["CWalks"] / df["Years"]

# kariyeri boyunca topa vurma sayısı ile isabetli vuruş arasındaki ilişki
df["NEW_SUCCESS"] = df["NEW_HITS"] * df["NEW_CAT_BAT"]

#OYUNCUNUN TAKIM ARKADAŞINLA YARDIMLAŞMASI VE İSABETLİ VURUŞ SAYISI
df["NEW_PUT_CHITS"] = df["PutOuts"] * df["CHits"]

# asist ve takım arkadaşı

df["NEW_ASIST_PUT"] = df["Assists"] / df["PutOuts"]
df.dropna(inplace = True)

# hits- error

df["NEW_RUN_ERR"] = df["Hits"] - df["Errors"]

check_df(df)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
binary_cols = [col for col in df.columns if df[col].dtypes == "O" and len(df[col].unique()) == 2]

for col in df.columns:
    label_encoder(df, col)

df.head()
check_df(df)
# scale

rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

# model

X = df.drop('Salary', axis=1)
y = df[["Salary"]]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20,random_state=1)

reg_model = LinearRegression().fit(X_train, y_train)

#ahmin Başarısını Değerlendirme

# Train RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
#0.25

# TRAIN RKARE
reg_model.score(X_train, y_train)
#0.812
# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#0.29
# Test RKARE
reg_model.score(X_test, y_test)
#0.70

# 10 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,X,y,cv=10,scoring="neg_mean_squared_error")))
# 0.30