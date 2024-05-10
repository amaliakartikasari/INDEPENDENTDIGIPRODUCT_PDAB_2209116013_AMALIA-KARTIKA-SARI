import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
import numpy as np
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from scipy import stats
from scipy.stats import norm, skew
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import warnings
warnings.filterwarnings("ignore")

# Menampilkan markdown hanya jika opsi yang dipilih bukanlah "Dashboard"
# Sidebar

st.sidebar.title('Aloha, Welcome')
img = open('KepitingGemas.jpg', 'rb').read()
st.sidebar.image(img, caption='CRAB AGE PREDICTION', use_column_width=True)
selected_option = st.sidebar.selectbox('Select an option:', ['Dashboard', 'Distribution', 'Comparison', 'Composition', 'Relationship', 'Predict'])
# Load data
url1 = 'https://raw.githubusercontent.com/amaliakartikasari/INDEPENDENTDIGIPRODUCT_PDAB_2209116013_AMALIA-KARTIKA-SARI/main/CrabAgePrediction.csv'
url2 = 'https://raw.githubusercontent.com/amaliakartikasari/INDEPENDENTDIGIPRODUCT_PDAB_2209116013_AMALIA-KARTIKA-SARI/main/Data%20Cleaned%20(5).csv'
df= pd.read_csv(url1)
df2 = pd.read_csv(url2)
df_file = df.head(2700)

# Tampilkan konten berdasarkan opsi yang dipilih
if selected_option == 'Dashboard':
    # Misalnya, jika Anda memiliki gambar dalam variabel img
    img = open('cute crab.jpeg', 'rb').read()
    st.image(img)

    st.markdown("""
    # PREDIKSI USIA KEPITING BERDASARKAN FITUR MORFOLOGI
    """)
    # st.write(df_file)  # Menampilkan seluruh data pada halaman "Dashboard"
    # Menampilkan teks dengan rata kanan kiri menggunakan markdown dan HTML
    st.markdown(
        """
        <div style="text-align: justify">
    Kepiting memiliki rasa yang lezat dan banyak negara di seluruh dunia mengimpor jumlah besar kepiting untuk konsumsi setiap tahunnya, ini menunjukkan adanya permintaan yang tinggi untuk produk-produk kepiting. Keuntungan utama dari budidaya kepiting adalah biaya tenaga kerja yang rendah, biaya produksi yang relatif murah, dan pertumbuhan kepiting yang cepat. Bisnis budidaya kepiting komersial tidak hanya berpotensi meningkatkan pendapatan, tetapi juga mengembangkan gaya hidup masyarakat di daerah pesisir. Dengan perawatan dan manajemen yang tepat, pendapatan dari bisnis budidaya kepiting dapat melebihi bisnis budidaya udang. Dalam budidaya kepiting, ada dua sistem utama yang dapat digunakan, yaitu sistem pemeliharaan (grow out farming) dan sistem penggemukkan (fattening systems). Dengan mempertimbangkan potensi pasar yang besar dan keuntungan ekonomi yang signifikan, bisnis budidaya kepiting menjadi salah satu opsi yang menarik untuk dijelajahi dalam industri perikanan.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h1 style='text-align: center;'> TABEL</h1>", unsafe_allow_html=True)


    st.write(df_file)  # Menampilkan seluruh data pada halaman "Dashboard"

    st.markdown(
    """
    <div style="text-align: justify">
    Dalam dataset tersebut terdapat fitur-fitur morfologi pada kepiting, antara lain:

    1. **Length**: Panjang kepiting, bisa dalam satuan sentimeter atau unit yang relevan.
    2. **Diameter**: Diameter kepiting, juga dalam satuan yang relevan.
    3. **Height**: Tinggi kepiting, mungkin juga dalam satuan tertentu.
    4. **Weight**: Berat kepiting, biasanya dalam gram atau satuan berat lainnya.
    5. **Shucked Weight**: Berat daging kepiting setelah dikupas, yang juga dapat dalam gram atau satuan berat lainnya.
    6. **Viscera Weight**: Berat isi perut kepiting, biasanya dalam gram atau satuan berat lainnya.
    7. **Shell Weight**: Berat cangkang kepiting, juga dalam satuan berat.
    8. **Age**: Usia kepiting, mungkin dalam bulan atau unit usia lainnya.
    9. **Sex_F, Sex_I, Sex_M**: Variabel kategoris yang mungkin mengindikasikan jenis kelamin kepiting (betina, intersex, atau jantan).
    </div>
    """,
    unsafe_allow_html=True
    )

if selected_option == 'Composition':
    st.markdown("<h1 style='text-align: center;'>COMPOSITION</h1>", unsafe_allow_html=True)
    selected_composition = st.selectbox('Pilih Data:', ['Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight', 'Age', 'Sex'])
    #    Create a figure and axis object
    if selected_composition == 'Length':
        import matplotlib.pyplot as plt

        # Get the counts of each unique ad type
        ad_type_counts = df_file["Length"].value_counts()

        # Create the scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(ad_type_counts.index, ad_type_counts.values, s=100, c='blue', alpha=0.7, edgecolors='w', linewidths=1.5)
        plt.xlabel('Length')
        plt.ylabel('Frequency')
        plt.title('Scatter Plot: Frequency of Length')
        plt.grid(True)
        plt.tight_layout()

        # Show the plot using Streamlit
        st.pyplot(plt)


        st.markdown(
            """
            <div style="text-align: justify">
            Sumbu horizontal berlabel "Length" dan rentangnya sekitar 0 hingga 2. Sumbu vertikal berlabel "Frequency" dan rentangnya sekitar 0 hingga sekitar 70. Plot menunjukkan sekelompok titik biru yang tersebar di seluruh grafik, dengan konsentrasi titik yang tampaknya dimulai dengan frekuensi rendah pada panjang yang lebih pendek, meningkat dalam frekuensi menuju panjang tengah, dan kemudian berkurang lagi pada panjang yang lebih panjang. Ini menunjukkan mungkin ada hubungan antara panjang dan frekuensi di mana frekuensi mencapai puncak pada panjang tertentu dan kemudian berkurang di kedua sisi. Distribusi atau tren yang tepat tidak jelas, tetapi terlihat adanya peningkatan umum dan kemudian penurunan frekuensi seiring dengan peningkatan panjang.
            </div>
            """,
            unsafe_allow_html=True
        )
    
    if selected_composition == 'Diameter':
        import matplotlib.pyplot as plt

        # Get the counts of each unique ad type
        ad_type_counts = df_file["Diameter"].value_counts()

        # Create the scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(ad_type_counts.index, ad_type_counts.values, s=100, c='blue', alpha=0.7, edgecolors='w', linewidths=1.5)
        plt.xlabel('Diameter')
        plt.ylabel('Frequency')
        plt.title('Scatter Plot: Frequency of Diameter')
        plt.grid(True)
        plt.tight_layout()

        # Show the plot using Streamlit
        st.pyplot(plt)


        st.markdown(
            """
            <div style="text-align: justify">
Pada diagram ini, sumbu horizontal berlabel "Diameter" dengan rentang sekitar 0 hingga 1.6. Sumbu vertikal berlabel "Frequency" dengan rentang sekitar 0 hingga 60. Terdapat sekelompok titik yang tersebar di seluruh grafik, menunjukkan sebaran data frekuensi pada berbagai nilai diameter. Titik-titik tersebut tidak menunjukkan pola atau tren yang jelas, namun terdapat konsentrasi frekuensi tertinggi pada rentang diameter antara 0.4 hingga 1.0. Diagram ini memberikan informasi visual tentang sebaran frekuensi berdasarkan nilai diameter objek yang diamati.            </div>
            """,
            unsafe_allow_html=True
        )
    if selected_composition == 'Height':
        import matplotlib.pyplot as plt

        # Get the counts of each unique ad type
        ad_type_counts = df_file["Height"].value_counts()

        # Create the scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(ad_type_counts.index, ad_type_counts.values, s=100, c='blue', alpha=0.7, edgecolors='w', linewidths=1.5)
        plt.xlabel('Height')
        plt.ylabel('Frequency')
        plt.title('Scatter Plot: Frequency of Height')
        plt.grid(True)
        plt.tight_layout()

        # Show the plot using Streamlit
        st.pyplot(plt)


        st.markdown(
            """
            <div style="text-align: justify">
            Pada diagram ini, sumbu horizontal berlabel "Height" dengan rentang sekitar 0 hingga 2.5. Sumbu vertikal berlabel "Frequency" dengan rentang sekitar 0 hingga 175. Terdapat sekelompok titik yang tersebar di seluruh grafik, menunjukkan sebaran data frekuensi pada berbagai nilai tinggi. Titik-titik tersebut tidak menunjukkan pola atau tren yang jelas, namun terdapat konsentrasi frekuensi tertinggi pada rentang tinggi antara 0.5 hingga 1.0. Diagram ini memberikan informasi visual tentang sebaran frekuensi berdasarkan nilai tinggi objek yang diamati.            </div>
            """,
            unsafe_allow_html=True
        )

    if selected_composition == 'Weight':
        import matplotlib.pyplot as plt

        # Get the counts of each unique ad type
        ad_type_counts = df_file["Weight"].value_counts()

        # Create the scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(ad_type_counts.index, ad_type_counts.values, s=100, c='blue', alpha=0.7, edgecolors='w', linewidths=1.5)
        plt.xlabel('Weight')
        plt.ylabel('Frequency')
        plt.title('Scatter Plot: Frequency of Weight')
        plt.grid(True)
        plt.tight_layout()

        # Show the plot using Streamlit
        st.pyplot(plt)


        st.markdown(
            """
            <div style="text-align: justify">
            Pada diagram ini, sumbu horizontal berlabel "Weight" dengan rentang yang tidak terlihat jelas dalam hasil OCR. Sumbu vertikal berlabel "Frequency" dengan rentang yang juga tidak terlihat jelas dalam hasil OCR. Terdapat beberapa titik yang tersebar di grafik, menunjukkan sebaran data frekuensi berdasarkan nilai berat. Namun, informasi tentang sebaran data secara spesifik tidak dapat diidentifikasi dengan jelas dari hasil OCR yang diberikan. Diagram ini memberikan representasi visual tentang frekuensi kemunculan berdasarkan nilai berat objek yang diamati.            </div>
            """,
            unsafe_allow_html=True
        )
    
    if selected_composition == 'Shucked Weight':
        import matplotlib.pyplot as plt

        # Get the counts of each unique ad type
        ad_type_counts = df_file["Shucked Weight"].value_counts()

        # Create the scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(ad_type_counts.index, ad_type_counts.values, s=100, c='blue', alpha=0.7, edgecolors='w', linewidths=1.5)
        plt.xlabel('Shucked Weight')
        plt.ylabel('Frequency')
        plt.title('Scatter Plot: Frequency of Shucked Weight')
        plt.grid(True)
        plt.tight_layout()

        # Show the plot using Streamlit
        st.pyplot(plt)


        st.markdown(
            """
            <div style="text-align: justify">
            Sumbu horizontal dilabeli "Shucked Weight" dan rentang nilainya dari 0 hingga lebih dari 40, sedangkan sumbu vertikal dilabeli "Frequency" dan rentang nilainya dari 1 hingga sedikit di atas 7. Titik-titik data direpresentasikan oleh lingkaran biru, dan plot menunjukkan konsentrasi titik data antara sekitar 5 dan 25 pada sumbu Shucked Weight. Frekuensi dari bobot ini sebagian besar antara tanda 2 dan 6 pada sumbu Frequency.

Terdapat kepadatan titik yang lebih tinggi dalam rentang bobot shucked yang lebih rendah, menunjukkan bahwa bobot yang lebih rendah lebih umum. Saat bobot meningkat, frekuensinya menurun, ditunjukkan oleh sedikitnya titik pada nilai bobot yang lebih tinggi. Juga terdapat beberapa titik di mana bobot yang sama memiliki beberapa kejadian, ditandai dengan lingkaran yang bertumpuk secara vertikal. Distribusi keseluruhan menunjukkan bahwa bobot shucked sampel memiliki rentang yang luas tetapi lebih sering diamati dalam rentang rendah hingga menengah dari skala yang diberikan.            </div>
            """,
            unsafe_allow_html=True
        )


    if selected_composition == 'Viscera Weight':
        import matplotlib.pyplot as plt

        # Get the counts of each unique ad type
        ad_type_counts = df_file["Viscera Weight"].value_counts()

        # Create the scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(ad_type_counts.index, ad_type_counts.values, s=100, c='blue', alpha=0.7, edgecolors='w', linewidths=1.5)
        plt.xlabel('Viscera Weight')
        plt.ylabel('Frequency')
        plt.title('Scatter Plot: Frequency of Viscara Weight')
        plt.grid(True)
        plt.tight_layout()

        # Show the plot using Streamlit
        st.pyplot(plt)


        st.markdown(
            """
            <div style="text-align: justify">
            Plot ini menunjukkan sebaran titik data di mana sumbu horizontal mewakili "Viscera Weight" yang berkisar dari 0 hingga 20, dan sumbu vertikal mewakili "Frequency" yang berkisar dari 0 hingga sekitar 12. Plot ini terdiri dari titik data berwarna biru yang tersebar di seluruh grafik.

Titik-titik data tersebut terkumpul rapat di antara 0 dan 5 pada sumbu "Viscera Weight", dengan frekuensi yang menurun seiring dengan peningkatan berat. Sebagian besar titik data memiliki berat viscera yang lebih rendah, ditandai dengan frekuensi tertinggi di sekitar ujung bawah skala berat. Seiring dengan peningkatan berat viscera, jumlah titik data berkurang, dengan jumlah titik yang lebih sedikit di antara 15 dan 20.

Plot ini tampaknya merupakan plot 'bee swarm', di mana setiap titik mewakili satu entri data individu, dan titik-titik disesuaikan sepanjang sumbu frekuensi untuk menghindari tumpang tindih dan menampilkan sebaran titik data pada level berat viscera yang berbeda.

Tidak ada tren atau korelasi yang jelas terlihat dalam plot ini; plot ini hanya menggambarkan sebaran pengukuran individu dari berat viscera dan frekuensi masing-masing dalam sebuah dataset. Sumbu x tidak memiliki garis-garis kisi, sementara garis-garis kisi horizontal samar hadir sepanjang sumbu y untuk menunjukkan tingkat frekuensi.            </div>
            """,
            unsafe_allow_html=True
        )
    
    if selected_composition == 'Shell Weight':
        import matplotlib.pyplot as plt

        # Get the counts of each unique ad type
        ad_type_counts = df_file["Shell Weight"].value_counts()

        # Create the scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(ad_type_counts.index, ad_type_counts.values, s=100, c='blue', alpha=0.7, edgecolors='w', linewidths=1.5)
        plt.xlabel('Shell Weight')
        plt.ylabel('Frequency')
        plt.title('Scatter Plot: Frequency of Shell Weight')
        plt.grid(True)
        plt.tight_layout()

        # Show the plot using Streamlit
        st.pyplot(plt)


        st.markdown(
            """
            <div style="text-align: justify">
            Sumbu horizontal dilabeli "Shell Weight", berkisar dari 0 hingga 20, sedangkan sumbu vertikal dilabeli "Frequency", berkisar dari 0 hingga sedikit di atas 30. Plot ini terdiri dari sejumlah besar titik biru yang mewakili frekuensi berat cangkang tertentu.

Titik-titik tersebut lebih terkonsentrasi pada rentang berat cangkang yang lebih rendah, terutama di sekitar rentang 5 hingga 10 pada sumbu x, dan semakin jarang saat berat cangkang meningkat. Frekuensi tertinggi berat cangkang tampaknya berada di sekitar rentang 5 hingga 7,5, dengan frekuensi mencapai sekitar 25 hingga 30. Seiring berat cangkang meningkat melebihi titik ini, frekuensi kejadian menurun secara signifikan.

Pola dari titik-titik tersebut menunjukkan bahwa cangkang yang lebih ringan lebih umum daripada yang lebih berat, dengan frekuensi yang menurun seiring peningkatan berat. Distribusi titik-titik tidak merata dan tampak memiliki puncak di rentang berat cangkang tengah.
                                    </div>
            """,
            unsafe_allow_html=True
        )

    if selected_composition == 'Age':
        import matplotlib.pyplot as plt

        # Get the counts of each unique ad type
        ad_type_counts = df_file["Age"].value_counts()

        # Create the scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(ad_type_counts.index, ad_type_counts.values, s=100, c='blue', alpha=0.7, edgecolors='w', linewidths=1.5)
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.title('Scatter Plot: Frequency of Age')
        plt.grid(True)
        plt.tight_layout()

        # Show the plot using Streamlit
        st.pyplot(plt)


        st.markdown(
            """
            <div style="text-align: justify">
            Plot tersebut menunjukkan titik-titik data yang mewakili frekuensi suatu peristiwa atau karakteristik yang tidak ditentukan terhadap usia, yang berkisar dari 0 hingga hampir 30 pada sumbu x. Sumbu y mewakili frekuensi, yang berkisar dari 0 hingga lebih dari 400.

    Plot tersebut menunjukkan bahwa frekuensi tertinggi terjadi pada usia yang lebih rendah, dengan puncak frekuensi pada usia 1. Frekuensi kemudian turun tajam seiring bertambahnya usia, dengan sedikit peningkatan di sekitar usia 6 sebelum terus menurun. Frekuensi menjadi jauh lebih rendah setelah usia 10 dan terus menurun secara bertahap, mencapai tingkat yang sangat rendah setelah usia 20.

    Ukuran titik-titik tampaknya mewakili besarnya frekuensi, dengan titik yang lebih besar menunjukkan frekuensi yang lebih tinggi. Titik terbesar berada pada usia 1, diikuti oleh titik sedikit lebih kecil pada usia 0, kemudian usia 2, dan seterusnya, dengan titik-titik menjadi semakin kecil seiring bertambahnya usia.

            </div>
            """,
            unsafe_allow_html=True
        )
    # Menampilkan plot pie
    if selected_composition == 'Sex':
        fig, ax = plt.subplots()
        df["Sex"].value_counts().plot(kind="pie", autopct="%.2f", ax=ax)
        st.pyplot(fig)  # Menampilkan plot menggunakan Streamlit
        st.markdown(
            """
            <div style="text-align: justify">
        Gambar tersebut adalah diagram lingkaran dengan tiga bagian berbeda warna yang dilabeli dengan huruf: "M" berwarna biru sebesar 36.86%, "I" berwarna oranye sebesar 31.67%, dan "F" berwarna hijau sebesar 31.47%. Diagram tersebut kemungkinan mewakili jumlah item atau kejadian dalam setiap kategori.
            </div>
            """,
            unsafe_allow_html=True
        )

if selected_option == 'Comparison':
    # Pilih fitur untuk dibandingkan dengan "Sex"
    selected_feature = st.selectbox("Pilih fitur untuk dibandingkan dengan 'Sex'", df.columns[1:])

    # Hitung nilai rata-rata berdasarkan jenis kelamin
    mean_values = df.groupby("Sex")[selected_feature].mean()

    # Plot barplot untuk perbandingan
    fig, ax = plt.subplots()
    mean_values.plot(kind="bar", ax=ax)
    ax.set_title(f"Perbandingan Rata-rata '{selected_feature}' berdasarkan jenis kelamin")
    ax.set_xlabel("Sex")
    ax.set_ylabel(selected_feature)
    st.pyplot(fig)
    st.markdown(
        """
        <div style="text-align: justify">
        <strong>Penjelasan :</strong><br>
Barplot adalah jenis plot yang berguna untuk memvisualisasikan perbandingan antara kategori-kategori atau kelompok-kelompok berbeda dengan menggunakan batang vertikal. Pada contoh kode yang diberikan, kita menggunakan barplot untuk membandingkan nilai rata-rata dari sebuah fitur tertentu (seperti panjang, berat, atau atribut lainnya) berdasarkan jenis kelamin (misalnya, "M" untuk laki-laki dan "F" untuk perempuan). Setiap batang vertikal pada plot menunjukkan nilai rata-rata dari fitur tersebut untuk setiap kelompok, sehingga memudahkan untuk melihat perbedaan atau kesamaan antara kedua kelompok tersebut.        </div>
        """,
        unsafe_allow_html=True
    )



if selected_option == 'Distribution':
    st.subheader("Distribusi Usia Siswa Berdasarkan Gender")
    Female = df[df['Sex'] == 'F']
    Male = df[df['Sex'] == 'M']
    Indeterminate = df[df['Sex'] == 'I']
    

    Male_age_counts = Male['Age'].value_counts()
    Female_age_counts = Female['Age'].value_counts()
    Indeterminate_age_counts = Indeterminate['Age'].value_counts()

    # Plotting histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist([Female['Age'], Male['Age'], Indeterminate['Age']], bins=20, color=['pink', 'blue', 'purple'], alpha=0.5, label=['Female', 'Male', 'Indeterminate'])
    ax.set_title('Distribusi Usia Kepiting Berdasarkan Gender')  # Menggunakan set_title()
    ax.set_xlabel('Kategori Usia (Bulan)')
    ax.set_ylabel('Jumlah Kepiting')
    ax.legend()
    ax.grid(True)

    # Menampilkan plot menggunakan streamlit
    st.subheader("Distribusi Usia Kepiting Berdasarkan Gender")
    st.pyplot(fig)
    st.markdown(
            """
            <div style="text-align: justify">
            <strong>Penjelasan :</strong><br>
Gambar yang Anda berikan adalah diagram batang yang menunjukkan distribusi usia kepiting berdasarkan jenis kelamin. Diagram ini mencakup kategori untuk jenis kelamin jantan, betina, dan tidak dapat ditentukan pada rentang usia yang berbeda dalam bulan. Diagram ini menunjukkan jumlah kepiting dalam setiap kategori jenis kelamin pada interval usia yang berbeda, dengan konsentrasi kepiting tertinggi berada di rentang usia 5 hingga 15 bulan. Diagram ini memberikan representasi visual tentang bagaimana jumlah kepiting bervariasi berdasarkan usia dan jenis kelamin.
            </div>
            """,
            unsafe_allow_html=True
        )

if selected_option == 'Relationship':
    st.markdown("<h1 style='text-align: center;'>RELATIONSHIP</h1>", unsafe_allow_html=True)
    numeric_cols = df_file.select_dtypes(include=['int', 'float'])
    correlation_matrix = numeric_cols.corr()

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Heatmap of Column Correlations')
    plt.show()
    st.pyplot(plt)
    st.markdown(
        """
        <div style="text-align: justify">
        Gambar tersebut adalah heatmap yang menunjukkan korelasi antar kolom pada data. Terdapat beberapa kolom seperti Panjang (Length), Diameter, Tinggi (Height), Berat (Weight), Berat Cangkang (Shell Weight), dan lainnya. Angka di dalam heatmap menunjukkan seberapa kuat korelasi antar kolom tersebut, yang dapat berkisar dari 0 hingga 1. Semakin tinggi nilai korelasi, semakin kuat hubungan antar kolomnya. Misalnya, korelasi antara Diameter dan Tinggi adalah 0.99, menunjukkan hubungan yang sangat kuat di antara keduanya.        </div>
        """,
        unsafe_allow_html=True
    )

if selected_option == 'Predict':
    st.markdown("<h1 style='text-align: center;'>PREDICT</h1>", unsafe_allow_html=True)
    # Input fields
    # Load model menggunakan pickle
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    # Fungsi untuk memprediksi nilai
    def predict_age(Sex, Diameter, Length, Height, Weight, Shucked_Weight, Viscera_Weight, Shell_Weight):
    # Lakukan pre-processing data sesuai kebutuhan
    # Contoh: transformasi data, normalisasi, encoding kategori, dll.

    # Lakukan prediksi menggunakan model
        prediction = model.predict([[Sex, Diameter, Length, Height, Weight, Shucked_Weight, Viscera_Weight, Shell_Weight]])
        return prediction

    # Dropdown untuk kolom "Gender"
    st.write(f"0: Perempuan , 1 : Laki-Laki, 3: Indeterminate")
    Sex = st.selectbox('Pilih Gender', [i for i in sorted(df2['Sex'].unique())])

    # Widget number_input untuk input desimal
    Length = st.number_input("Length (Dalam Feet)", min_value=0.0, step=0.1)

    # Tampilkan nilai input
    st.write(f"Panjang yang dimasukkan: {Length}")

        # Widget number_input untuk input desimal
    Diameter = st.number_input("Diameter (Dalam Feet)", min_value=0.0, step=0.1)

    # Tampilkan nilai input
    st.write(f"Panjang yang dimasukkan: {Diameter}")

        # Widget number_input untuk input desimal
    Height = st.number_input("Height (Dalam Feet)", min_value=0.0, step=0.1)

    # Tampilkan nilai input
    st.write(f"Panjang yang dimasukkan: {Height}")

        # Widget number_input untuk input desimal
    Weight = st.number_input("Weight (Dalam Pound)", min_value=0.0, step=0.1)

    # Tampilkan nilai input
    st.write(f"Panjang yang dimasukkan: {Weight}")

    # Widget number_input untuk input desimal
    Shucked_Weight = st.number_input("Shucked_Weight (Dalam Pound)", min_value=0.0, step=0.1)

    # Tampilkan nilai input
    st.write(f"Panjang yang dimasukkan: {Shucked_Weight}")

    # Widget number_input untuk input desimal
    Viscera_Weight = st.number_input("Viscera_Weight (Dalam Pound)", min_value=0.0, step=0.1)

    # Tampilkan nilai input
    st.write(f"Panjang yang dimasukkan: {Viscera_Weight}")

    # Widget number_input untuk input desimal
    Shell_Weight = st.number_input("Shell_Weight (Dalam Pound)", min_value=0.0, step=0.1)

    # Tampilkan nilai input
    st.write(f"Panjang yang dimasukkan: {Shell_Weight}")


            # Make prediction
    if st.button("Predict"):
        prediction = predict_age(Sex, Length, Diameter, Height, Weight, Shucked_Weight, Viscera_Weight, Shell_Weight)
        st.write(f"Prediksi Umur Kepiting: {prediction}")
        # Menampilkan informasi rentang umur kepiting berdasarkan hasil prediksi
        # Menampilkan informasi rentang umur kepiting berdasarkan hasil prediksi
        if prediction is not None:
            if prediction <= 12:
                st.write("Kepiting Termasuk Kategori Anak-Anak (Juvenile)")
            elif prediction > 12 and prediction <= 24:
                st.write("Kepiting Termasuk Kategori Muda (Young)")
            elif prediction > 24 and prediction <= 36:
                st.write("Kepiting Termasuk Kategori Dewasa (Adult)")
            elif prediction > 60:
                st.write("Kepiting Termasuk Kategori Tua (Elderly)")
            else:
                st.write("Tidak Ditemukan")