import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Kavak Market Analyzer", layout="wide")
pd.options.display.float_format = '{:,.0f}'.format

st.title("Valuador de Mercado de Autos Seminuevos")
st.markdown("""
Esta herramienta utiliza **Inteligencia Artificial (K-Means Clustering)** para identificar oportunidades de compra 
en el mercado de autos seminuevos en México. Detecta anomalías de precio y clasifica los vehículos por su ciclo de vida.
""")

def get_term_data(df, term):
    clean_term_cars = df.loc[(df['Plazo'] == term),
        ['ID_Auto', 'Brand', 'Model', 'Precio', 'Km', 'Year', 'Interes_%',
         'Version', 'Caja', 'Tipo', 'Total_a_Pagar', 'Plazo', 'Sucursal']].copy()
    
    clean_term_cars = clean_term_cars.sort_values(by=['Year', 'Km', 'Precio'], na_position='last')
    clean_term_cars = clean_term_cars.dropna()
    clean_term_cars = clean_term_cars.drop_duplicates(subset=['ID_Auto'], keep='first')
    #clean_term_cars = clean_term_cars.drop(columns=['ID_Auto'])
    return clean_term_cars

def standardize_X(X):
    X_mean = X.mean()
    X_std = X.std()    
    X['Precio_z'] = (X['Precio'] - X_mean['Precio']) / X_std['Precio']
    X['Km_z']     = (X['Km'] - X_mean['Km']) / X_std['Km']
    X['Year_z']   = (X['Year'] - X_mean['Year']) / X_std['Year']
    X['Interes_%_z']   = (X['Year'] - X_mean['Year']) / X_std['Year'] # NO ENTRARA AL MODELO
    X_scaled = X[['Precio_z', 'Km_z', 'Year_z']].copy()
    return X_scaled, X

@st.cache_data
def load_and_train_model():
    try:
        df = pd.read_csv('data.csv', encoding='utf-8')
    except FileNotFoundError:
        st.error("No se encontro el archivo CSV. Asegúrate de subir 'data.csv' a la misma carpeta.")
        st.stop()

    # Limpieza
    clean_term_cars = get_term_data(df, term=12)
    
    # Feature Engineering (Z-Scores)
    features = ['Precio', 'Km', 'Year', 'Interes_%']
    X = clean_term_cars[features].copy()
    X_scaled, X = standardize_X(X)
    
    # Entrenamiento del Modelo (K=3)    
    k_means = KMeans(n_clusters=3, random_state=97, n_init=10)
    k_means.fit(X_scaled)
    
    # Asignacion de Resultados
    df_results = clean_term_cars.copy()
    df_results['Cluster'] = k_means.labels_
    
    # Asignacion Dinamica de Nombres
    means = df_results.groupby('Cluster')['Precio'].mean().sort_values()
    cluster_names = {
        means.index[0]: 'Alto Kilometraje',
        means.index[1]: 'Standard',
        means.index[2]: 'Premium'
    }
    df_results['Segment'] = df_results['Cluster'].map(cluster_names)
    
    return df_results, cluster_names

# Llamamos a la funcion
df_results, cluster_names = load_and_train_model()

# SIDEBAR
st.sidebar.header("🔍 Explorador de Modelos")
all_brands = sorted(df_results['Brand'].unique())
selected_brand = st.sidebar.selectbox("Selecciona una Marca", all_brands)

models_of_brand = sorted(df_results[df_results['Brand'] == selected_brand]['Model'].unique())
models_of_brand.insert(0, "Todos los modelos")
selected_model = st.sidebar.selectbox("Selecciona un Modelo", models_of_brand)

all_years = sorted(df_results[df_results['Model'] == selected_model]['Year'].unique())
all_years = [int(year) for year in all_years]
all_years.insert(0, "Todos los años")
selected_year = st.sidebar.selectbox("Selecciona un Año", all_years)

# SECCION 1: INSIGHTS
st.header("1. Segmentacion del Mercado")
col1, col2 = st.columns([1, 2])

format_mapping = {
    'Precio': '${:,.0f}',
    'Km': '{:,.0f} km',
    'Year': '{:.0f}',
    'Interes_%': '{:.0f}%'
}

with col1:
    st.write("Perfiles promedio detectados por el algoritmo:")
    profiles = df_results[['Precio', 'Km', 'Year', 'Interes_%', 'Segment']].groupby('Segment').mean().sort_values('Precio')
    st.dataframe(profiles.style.format(format_mapping))
    
    st.info("""
    **Hallazgo:** Existe una "Barrera de Depreciacion" cerca de los 70,000 km donde los autos pasan del segmento Standard al de Alto Kilometraje.
    """)

with col2:
    # Grafico General (Scatter)
    fig_general = px.scatter(
        df_results, x='Km', y='Precio', color='Segment',
        title="Mapa General del Mercado (Todos los Autos)",
        color_discrete_map={'Alto Kilometraje': '#ef553b', 'Standard': '#636efa', 'Premium': '#00cc96'},
        opacity=0.5
    )
    fig_general.add_vline(x=70000, line_dash="dash", line_color="red", annotation_text="Barrera 70k km")
    fig_general.add_hline(y=350000, line_dash="dash", line_color="red", annotation_text="Techo Premium")
    st.plotly_chart(fig_general, width='stretch')

# SECCION 2: ANALISIS POR MODELO 
st.markdown("---")
st.header(f"2. Analisis Profundo: {selected_model}")

# Filtrado de datos
model_mask = (df_results['Model'] == selected_model) | (selected_model == 'Todos los modelos')
year_mask = (df_results['Year'] == selected_year) | (selected_year == 'Todos los años')
model_data = df_results[model_mask & year_mask]

def display_df_point_selected(points, df):
    return

if model_data.empty:
    st.warning("No hay suficientes datos para este modelo.")
else:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precio Promedio", f"${model_data['Precio'].mean():,.0f}")
    m2.metric("Kilometraje Promedio", f"{model_data['Km'].mean():,.0f} km")
    m3.metric("Unidades Disponibles", len(model_data))
    m4.metric("Ciudades", model_data['Sucursal'].nunique())

    segmentation_options = ['Version', 'Caja', 'Sucursal', 'Segment', 'Year']
    filter_label = st.segmented_control("Filtrado por: ", segmentation_options, selection_mode='single')

    fig_model = px.scatter(
        model_data,
        x='Interes_%', 
        y='Km', 
        color=filter_label,
        #symbol='Tipo', # Forma del punto
        hover_data=['ID_Auto', 'Precio', 'Year', 'Sucursal', 'Total_a_Pagar'],
        title=f'Riesgo Financiero vs Desgaste: {selected_model}',
        height=600
    )
    fig_model.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
    event = st.plotly_chart(fig_model, width='stretch', selection_mode='points', key='ID_Auto', on_select='rerun')

    #Muestra en detalle de los autos seleccionados en la grafica
    points = [point['customdata'][0] for point in event['selection']['points']]
    if points:
        df_selected_points = model_data[model_data['ID_Auto'].isin(points)]
        st.dataframe(df_selected_points)

# SECCION 3: SIMULADOR
st.markdown("---")
st.header("3. Evaluador de Ofertas")
st.write("¿Viste un auto y quieres saber si el precio es justo?")

sc1, sc2, sc3 = st.columns(3)
input_km = sc1.number_input("Kilometraje", value=50000, step=1000)
input_price = sc2.number_input("Precio ($)", value=250000, step=5000)
input_year = sc3.number_input("Año", value=2020, step=1)

# Calculamos referencia
referencia = model_data[
    (model_data['Km'] > input_km - 10000) & 
    (model_data['Km'] < input_km + 10000) &
    (model_data['Year'] == input_year)
    ]

if st.button("Evaluar Precio"):
    if referencia.empty:
        st.warning(f"No tenemos suficientes datos de {selected_model} del año {input_year} para comparar.")
    else:
        avg_market = referencia['Precio'].mean()
        diff = input_price - avg_market
        
        st.write(f"Precio Justo de Mercado (aprox): **${avg_market:,.0f}**")
        
        if diff < -15000:
            st.success(f"¡OFERTA! Esta ${abs(diff):,.0f} por debajo del mercado")
        elif diff > 15000:
            st.error(f"CARO. Esta ${diff:,.0f} por encima del mercado. Intenta negociar.")
        else:
            st.info("PRECIO JUSTO. Esta dentro del rango normal del mercado.")

# FOOTER
st.markdown("---")
st.caption("Desarrollado con Python & Streamlit • Modelo de ML: K-Means Clustering")