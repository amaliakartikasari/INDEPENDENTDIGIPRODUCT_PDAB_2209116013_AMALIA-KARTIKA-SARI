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
st.sidebar.title('Halaman')
selected_option = st.sidebar.selectbox('Select an option:', ['Dashboard', 'Distribution', 'Comparison', 'Composition', 'Relationship', 'Predict'])
# Load data
url1 = 'https://raw.githubusercontent.com/amaliakartikasari/INDEPENDENTDIGIPRODUCT_PDAB_2209116013_AMALIA-KARTIKA-SARI/main/CrabAgePrediction.csv'
url2 = 'https://raw.githubusercontent.com/amaliakartikasari/INDEPENDENTDIGIPRODUCT_PDAB_2209116013_AMALIA-KARTIKA-SARI/main/Data%20Cleaned%20(4).csv'
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
    selected_comparison = st.selectbox('Pilih Data:', ['Length', 'Diameter', 'Height', 'Weight', 'Shucked Weight', 'Viscara Weight', 'Shell Weight', 'Age', 'Sex_F', 'Sex_I', 'Sex_M'])
    #    Create a figure and axis object
    if selected_comparison == 'Length':
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
            <strong>Interpretasi:</strong><br>
            Data barplot komposisi menunjukkan bahwa:
            - Tipe pekerjaan standar (data 0) menyumbang sekitar 40.7% dari keseluruhan pekerjaan dalam dataset.
            - Tipe pekerjaan standout (data 1) menyumbang sekitar 36.6% dari keseluruhan pekerjaan dalam dataset.
            - Tipe pekerjaan auto_increment (data 2) menyumbang sekitar 22.6% dari keseluruhan pekerjaan dalam dataset.<br><br>

            <strong>Insight:</strong><br>
            - Komposisi tipe pekerjaan dalam dataset menunjukkan proporsi masing-masing tipe pekerjaan terhadap total pekerjaan.
            - Tipe pekerjaan standar dan standout merupakan tipe pekerjaan yang dominan dalam dataset, dengan masing-masing menyumbang lebih dari 30% dari total pekerjaan.
            - Tipe pekerjaan auto_increment, meskipun memiliki proporsi yang lebih kecil, tetap merupakan bagian yang signifikan dalam distribusi tipe pekerjaan.<br><br>

            <strong>Action Insight:</strong><br>
            - Lakukan analisis lebih lanjut terhadap karakteristik pekerjaan berdasarkan tipe, seperti analisis kompensasi, tingkat kepuasan kerja, atau prospek karir untuk setiap tipe pekerjaan.
            - Identifikasi apakah terdapat perbedaan signifikan dalam kinerja atau karakteristik pekerjaan antara tipe pekerjaan standar, standout, dan auto_increment.
            - Jika dataset ini digunakan untuk pengambilan keputusan atau pembuatan model prediktif, pastikan untuk mempertimbangkan proporsi tipe pekerjaan ini dalam analisis atau prediksi yang akan dilakukan.
            </div>
            """,
            unsafe_allow_html=True
        )
    
    if selected_comparison == 'Diameter':
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
            <strong>Interpretasi:</strong><br>
            Data barplot komposisi menunjukkan bahwa:
            - Tipe pekerjaan standar (data 0) menyumbang sekitar 40.7% dari keseluruhan pekerjaan dalam dataset.
            - Tipe pekerjaan standout (data 1) menyumbang sekitar 36.6% dari keseluruhan pekerjaan dalam dataset.
            - Tipe pekerjaan auto_increment (data 2) menyumbang sekitar 22.6% dari keseluruhan pekerjaan dalam dataset.<br><br>

            <strong>Insight:</strong><br>
            - Komposisi tipe pekerjaan dalam dataset menunjukkan proporsi masing-masing tipe pekerjaan terhadap total pekerjaan.
            - Tipe pekerjaan standar dan standout merupakan tipe pekerjaan yang dominan dalam dataset, dengan masing-masing menyumbang lebih dari 30% dari total pekerjaan.
            - Tipe pekerjaan auto_increment, meskipun memiliki proporsi yang lebih kecil, tetap merupakan bagian yang signifikan dalam distribusi tipe pekerjaan.<br><br>

            <strong>Action Insight:</strong><br>
            - Lakukan analisis lebih lanjut terhadap karakteristik pekerjaan berdasarkan tipe, seperti analisis kompensasi, tingkat kepuasan kerja, atau prospek karir untuk setiap tipe pekerjaan.
            - Identifikasi apakah terdapat perbedaan signifikan dalam kinerja atau karakteristik pekerjaan antara tipe pekerjaan standar, standout, dan auto_increment.
            - Jika dataset ini digunakan untuk pengambilan keputusan atau pembuatan model prediktif, pastikan untuk mempertimbangkan proporsi tipe pekerjaan ini dalam analisis atau prediksi yang akan dilakukan.
            </div>
            """,
            unsafe_allow_html=True
        )
    if selected_comparison == 'Height':
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
            <strong>Interpretasi:</strong><br>
            Data barplot komposisi menunjukkan bahwa:
            - Tipe pekerjaan standar (data 0) menyumbang sekitar 40.7% dari keseluruhan pekerjaan dalam dataset.
            - Tipe pekerjaan standout (data 1) menyumbang sekitar 36.6% dari keseluruhan pekerjaan dalam dataset.
            - Tipe pekerjaan auto_increment (data 2) menyumbang sekitar 22.6% dari keseluruhan pekerjaan dalam dataset.<br><br>

            <strong>Insight:</strong><br>
            - Komposisi tipe pekerjaan dalam dataset menunjukkan proporsi masing-masing tipe pekerjaan terhadap total pekerjaan.
            - Tipe pekerjaan standar dan standout merupakan tipe pekerjaan yang dominan dalam dataset, dengan masing-masing menyumbang lebih dari 30% dari total pekerjaan.
            - Tipe pekerjaan auto_increment, meskipun memiliki proporsi yang lebih kecil, tetap merupakan bagian yang signifikan dalam distribusi tipe pekerjaan.<br><br>

            <strong>Action Insight:</strong><br>
            - Lakukan analisis lebih lanjut terhadap karakteristik pekerjaan berdasarkan tipe, seperti analisis kompensasi, tingkat kepuasan kerja, atau prospek karir untuk setiap tipe pekerjaan.
            - Identifikasi apakah terdapat perbedaan signifikan dalam kinerja atau karakteristik pekerjaan antara tipe pekerjaan standar, standout, dan auto_increment.
            - Jika dataset ini digunakan untuk pengambilan keputusan atau pembuatan model prediktif, pastikan untuk mempertimbangkan proporsi tipe pekerjaan ini dalam analisis atau prediksi yang akan dilakukan.
            </div>
            """,
            unsafe_allow_html=True
        )

    if selected_comparison == 'Weight':
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
            <strong>Interpretasi:</strong><br>
            Data barplot komposisi menunjukkan bahwa:
            - Tipe pekerjaan standar (data 0) menyumbang sekitar 40.7% dari keseluruhan pekerjaan dalam dataset.
            - Tipe pekerjaan standout (data 1) menyumbang sekitar 36.6% dari keseluruhan pekerjaan dalam dataset.
            - Tipe pekerjaan auto_increment (data 2) menyumbang sekitar 22.6% dari keseluruhan pekerjaan dalam dataset.<br><br>

            <strong>Insight:</strong><br>
            - Komposisi tipe pekerjaan dalam dataset menunjukkan proporsi masing-masing tipe pekerjaan terhadap total pekerjaan.
            - Tipe pekerjaan standar dan standout merupakan tipe pekerjaan yang dominan dalam dataset, dengan masing-masing menyumbang lebih dari 30% dari total pekerjaan.
            - Tipe pekerjaan auto_increment, meskipun memiliki proporsi yang lebih kecil, tetap merupakan bagian yang signifikan dalam distribusi tipe pekerjaan.<br><br>

            <strong>Action Insight:</strong><br>
            - Lakukan analisis lebih lanjut terhadap karakteristik pekerjaan berdasarkan tipe, seperti analisis kompensasi, tingkat kepuasan kerja, atau prospek karir untuk setiap tipe pekerjaan.
            - Identifikasi apakah terdapat perbedaan signifikan dalam kinerja atau karakteristik pekerjaan antara tipe pekerjaan standar, standout, dan auto_increment.
            - Jika dataset ini digunakan untuk pengambilan keputusan atau pembuatan model prediktif, pastikan untuk mempertimbangkan proporsi tipe pekerjaan ini dalam analisis atau prediksi yang akan dilakukan.
            </div>
            """,
            unsafe_allow_html=True
        )
    
    if selected_comparison == 'Shucked Weight':
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
            <strong>Interpretasi:</strong><br>
            Data barplot komposisi menunjukkan bahwa:
            - Tipe pekerjaan standar (data 0) menyumbang sekitar 40.7% dari keseluruhan pekerjaan dalam dataset.
            - Tipe pekerjaan standout (data 1) menyumbang sekitar 36.6% dari keseluruhan pekerjaan dalam dataset.
            - Tipe pekerjaan auto_increment (data 2) menyumbang sekitar 22.6% dari keseluruhan pekerjaan dalam dataset.<br><br>

            <strong>Insight:</strong><br>
            - Komposisi tipe pekerjaan dalam dataset menunjukkan proporsi masing-masing tipe pekerjaan terhadap total pekerjaan.
            - Tipe pekerjaan standar dan standout merupakan tipe pekerjaan yang dominan dalam dataset, dengan masing-masing menyumbang lebih dari 30% dari total pekerjaan.
            - Tipe pekerjaan auto_increment, meskipun memiliki proporsi yang lebih kecil, tetap merupakan bagian yang signifikan dalam distribusi tipe pekerjaan.<br><br>

            <strong>Action Insight:</strong><br>
            - Lakukan analisis lebih lanjut terhadap karakteristik pekerjaan berdasarkan tipe, seperti analisis kompensasi, tingkat kepuasan kerja, atau prospek karir untuk setiap tipe pekerjaan.
            - Identifikasi apakah terdapat perbedaan signifikan dalam kinerja atau karakteristik pekerjaan antara tipe pekerjaan standar, standout, dan auto_increment.
            - Jika dataset ini digunakan untuk pengambilan keputusan atau pembuatan model prediktif, pastikan untuk mempertimbangkan proporsi tipe pekerjaan ini dalam analisis atau prediksi yang akan dilakukan.
            </div>
            """,
            unsafe_allow_html=True
        )


    if selected_comparison == 'Viscara Weight':
        import matplotlib.pyplot as plt

        # Get the counts of each unique ad type
        ad_type_counts = df_file["Viscara Weight"].value_counts()

        # Create the scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(ad_type_counts.index, ad_type_counts.values, s=100, c='blue', alpha=0.7, edgecolors='w', linewidths=1.5)
        plt.xlabel('Viscara Weight')
        plt.ylabel('Frequency')
        plt.title('Scatter Plot: Frequency of Viscara Weight')
        plt.grid(True)
        plt.tight_layout()

        # Show the plot using Streamlit
        st.pyplot(plt)


        st.markdown(
            """
            <div style="text-align: justify">
            <strong>Interpretasi:</strong><br>
            Data barplot komposisi menunjukkan bahwa:
            - Tipe pekerjaan standar (data 0) menyumbang sekitar 40.7% dari keseluruhan pekerjaan dalam dataset.
            - Tipe pekerjaan standout (data 1) menyumbang sekitar 36.6% dari keseluruhan pekerjaan dalam dataset.
            - Tipe pekerjaan auto_increment (data 2) menyumbang sekitar 22.6% dari keseluruhan pekerjaan dalam dataset.<br><br>

            <strong>Insight:</strong><br>
            - Komposisi tipe pekerjaan dalam dataset menunjukkan proporsi masing-masing tipe pekerjaan terhadap total pekerjaan.
            - Tipe pekerjaan standar dan standout merupakan tipe pekerjaan yang dominan dalam dataset, dengan masing-masing menyumbang lebih dari 30% dari total pekerjaan.
            - Tipe pekerjaan auto_increment, meskipun memiliki proporsi yang lebih kecil, tetap merupakan bagian yang signifikan dalam distribusi tipe pekerjaan.<br><br>

            <strong>Action Insight:</strong><br>
            - Lakukan analisis lebih lanjut terhadap karakteristik pekerjaan berdasarkan tipe, seperti analisis kompensasi, tingkat kepuasan kerja, atau prospek karir untuk setiap tipe pekerjaan.
            - Identifikasi apakah terdapat perbedaan signifikan dalam kinerja atau karakteristik pekerjaan antara tipe pekerjaan standar, standout, dan auto_increment.
            - Jika dataset ini digunakan untuk pengambilan keputusan atau pembuatan model prediktif, pastikan untuk mempertimbangkan proporsi tipe pekerjaan ini dalam analisis atau prediksi yang akan dilakukan.
            </div>
            """,
            unsafe_allow_html=True
        )
    
    if selected_comparison == 'Shell Weight':
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
            <strong>Interpretasi:</strong><br>
            Data barplot komposisi menunjukkan bahwa:
            - Tipe pekerjaan standar (data 0) menyumbang sekitar 40.7% dari keseluruhan pekerjaan dalam dataset.
            - Tipe pekerjaan standout (data 1) menyumbang sekitar 36.6% dari keseluruhan pekerjaan dalam dataset.
            - Tipe pekerjaan auto_increment (data 2) menyumbang sekitar 22.6% dari keseluruhan pekerjaan dalam dataset.<br><br>

            <strong>Insight:</strong><br>
            - Komposisi tipe pekerjaan dalam dataset menunjukkan proporsi masing-masing tipe pekerjaan terhadap total pekerjaan.
            - Tipe pekerjaan standar dan standout merupakan tipe pekerjaan yang dominan dalam dataset, dengan masing-masing menyumbang lebih dari 30% dari total pekerjaan.
            - Tipe pekerjaan auto_increment, meskipun memiliki proporsi yang lebih kecil, tetap merupakan bagian yang signifikan dalam distribusi tipe pekerjaan.<br><br>

            <strong>Action Insight:</strong><br>
            - Lakukan analisis lebih lanjut terhadap karakteristik pekerjaan berdasarkan tipe, seperti analisis kompensasi, tingkat kepuasan kerja, atau prospek karir untuk setiap tipe pekerjaan.
            - Identifikasi apakah terdapat perbedaan signifikan dalam kinerja atau karakteristik pekerjaan antara tipe pekerjaan standar, standout, dan auto_increment.
            - Jika dataset ini digunakan untuk pengambilan keputusan atau pembuatan model prediktif, pastikan untuk mempertimbangkan proporsi tipe pekerjaan ini dalam analisis atau prediksi yang akan dilakukan.
            </div>
            """,
            unsafe_allow_html=True
        )

    if selected_comparison == 'Age':
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
            <strong>Interpretasi:</strong><br>
            Data barplot komposisi menunjukkan bahwa:
            - Tipe pekerjaan standar (data 0) menyumbang sekitar 40.7% dari keseluruhan pekerjaan dalam dataset.
            - Tipe pekerjaan standout (data 1) menyumbang sekitar 36.6% dari keseluruhan pekerjaan dalam dataset.
            - Tipe pekerjaan auto_increment (data 2) menyumbang sekitar 22.6% dari keseluruhan pekerjaan dalam dataset.<br><br>

            <strong>Insight:</strong><br>
            - Komposisi tipe pekerjaan dalam dataset menunjukkan proporsi masing-masing tipe pekerjaan terhadap total pekerjaan.
            - Tipe pekerjaan standar dan standout merupakan tipe pekerjaan yang dominan dalam dataset, dengan masing-masing menyumbang lebih dari 30% dari total pekerjaan.
            - Tipe pekerjaan auto_increment, meskipun memiliki proporsi yang lebih kecil, tetap merupakan bagian yang signifikan dalam distribusi tipe pekerjaan.<br><br>

            <strong>Action Insight:</strong><br>
            - Lakukan analisis lebih lanjut terhadap karakteristik pekerjaan berdasarkan tipe, seperti analisis kompensasi, tingkat kepuasan kerja, atau prospek karir untuk setiap tipe pekerjaan.
            - Identifikasi apakah terdapat perbedaan signifikan dalam kinerja atau karakteristik pekerjaan antara tipe pekerjaan standar, standout, dan auto_increment.
            - Jika dataset ini digunakan untuk pengambilan keputusan atau pembuatan model prediktif, pastikan untuk mempertimbangkan proporsi tipe pekerjaan ini dalam analisis atau prediksi yang akan dilakukan.
            </div>
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
        Nilai dalam matriks ini merepresentasikan koefisien korelasi antara variabel-variabel tersebut. Koefisien korelasi bernilai 1 menunjukkan korelasi positif sempurna, artinya kedua variabel berbanding lurus. Nilai -1 menunjukkan korelasi negatif sempurna, artinya kedua variabel berbanding terbalik. Nilai 0 menunjukkan tidak ada korelasi antar variabel.

        1. Jenis Iklan (adtype) dan Jenis Iklan (adtype): Korelasi antara Jenis Iklan dengan dirinya sendiri selalu bernilai 1, menunjukkan korelasi positif sempurna.
        2. Mata Uang Gaji (salary currency) dan Jenis Iklan (adtype): Korelasinya berwarna putih, artinya data untuk hubungan ini tidak tersedia atau tidak signifikan.
        3. Gaji Minimum (salarymin) dan Jenis Iklan (adtype): Korelasinya -0.01, menunjukkan korelasi negatif yang sangat lemah. Ini menunjukkan sedikit kecenderungan iklan dengan gaji lebih rendah terkait dengan jenis iklan tertentu.
        4. Jenis Iklan (adtype) dan Gaji Maksimum (salarymax): Korelasinya -0.01, mendekati 0, dan menunjukkan tidak ada korelasi yang signifikan.  Artinya, tidak ada hubungan jelas antara jenis iklan dan gaji maksimum yang ditawarkan.
        5. Periode Gaji (salary period) dan Jenis Iklan (adtype): Korelasinya berwarna putih, artinya data untuk hubungan ini tidak tersedia atau tidak signifikan.
        6. Kategori Lokasi (locationcategory) dan Jenis Iklan (adtype): Korelasinya -0.07, menunjukkan korelasi negatif yang sangat lemah. Ini menunjukkan sedikit kecenderungan jenis iklan tertentu terkait dengan lokasi dengan rata-rata gaji lebih rendah.

        Korelasi antar variabel lainnya:

        1. Gaji Minimum (salarymin) dan Gaji Minimum (salarymin): Korelasi selalu bernilai 1, menunjukkan korelasi positif sempurna (sama seperti Jenis Iklan).
        2. Gaji Minimum (salarymin) dan Gaji Maksimum (salarymax): Korelasinya 0.96, menunjukkan korelasi positif yang sangat kuat. Ini berarti ada hubungan yang erat antara gaji minimum dan gaji maksimum yang ditawarkan.
        3. Gaji Minimum (salarymin) dan Kategori Lokasi (locationcategory): Korelasinya 0.06, menunjukkan korelasi positif yang sangat lemah. Ini menunjukkan sedikit kecenderungan gaji minimum lebih tinggi terkait dengan lokasi tertentu.

        Korelasi yang tersisa memiliki interpretasi serupa (korelasi lemah positif atau negatif) dan menunjukkan hubungan yang tidak terlalu kuat antara variabel-variabel tersebut.
        </div>
        """,
        unsafe_allow_html=True
    )

if selected_option == 'Predict':
    st.markdown("<h1 style='text-align: center;'>PREDICT</h1>", unsafe_allow_html=True)
    # Input fields
    # Load model menggunakan pickle
    with open('modelSVR.pkl', 'rb') as file:
        model = pickle.load(file)
    # Fungsi untuk memprediksi nilai
    def predict_age(Sex, Diameter, Length, Height, Shucked_Weight, Viscera_Weight, Shell_Weight,Sex_F, Sex_M, Sex_I):
    # Lakukan pre-processing data sesuai kebutuhan
    # Contoh: transformasi data, normalisasi, encoding kategori, dll.

    # Lakukan prediksi menggunakan model
        prediction = model.predict([[Sex, Diameter, Length, Height, Shucked_Weight, Viscera_Weight, Shell_Weight, Sex_F, Sex_M, Sex_I]])
        return prediction

    # Dropdown untuk kolom "Gender"
    st.write(f"apakah bergender perempuan?   0 : tidak , 1 : ya")
    Sex_F = st.selectbox('Pilih Gender Female', [i for i in sorted(df2['Sex_F'].unique())])
    st.write(f"apakah bergender laki-laki?   0 : tidak , 1 : ya")
    Sex_M = st.selectbox('Pilih Gender Male', [i for i in sorted(df2['Sex_M'].unique())])
    st.write(f"apakah bergender indeterminate?   0 : tidak , 1 : ya")
    Sex_I = st.selectbox('Pilih Gender Indeterminate', [i for i in sorted(df2['Sex_I'].unique())])

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
        prediction = predict_age(Length, Diameter, Height, Weight, Shucked_Weight, Viscera_Weight,Shell_Weight, Sex_F, Sex_M, Sex_I)
        st.write(f"Predicted Actual Productivity: {prediction}")








            
    