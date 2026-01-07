import gc
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px


@st.cache_resource
def get_terms():
    try:
        df = pd.read_csv('data.csv', encoding='utf-8', usecols=['Plazo'], dtype='Int16')
        terms = df['Plazo'].unique()
        return terms
    except FileNotFoundError:
        st.text("No se pudo encontrar el archivo 'data.csv'")
        st.stop()

def get_term_data(df, term, aliado_flag=False):
    cols = ['ID_Auto', 'Brand', 'Model', 'Precio', 'Km', 'Year', 'Interes_%',
         'Version', 'Caja', 'Tipo', 'Total_a_Pagar', 'Plazo', 'Sucursal']

    if aliado_flag:
        filtering_conditions = (df['Plazo'] == term)
    else:
        filtering_conditions = (df['Sucursal'] != 'Aliado') & (df['Plazo'] == term)

    clean_term_cars = df.loc[filtering_conditions, cols].copy()    
    clean_term_cars = clean_term_cars.sort_values(by=['Year', 'Km', 'Precio'], na_position='last')
    clean_term_cars = clean_term_cars.dropna()
    clean_term_cars = clean_term_cars.drop_duplicates(subset=['ID_Auto'], keep='first')
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

def load_data():
    dtypes = {
        'ID_Auto': 'string',
        'Precio': 'float32', 
        'Tasa_Servicio': 'float32', 
        'Plazo': 'Int16',         # Cambiado a Int16 por seguridad (hasta 32k meses)
        'Mensualidad': 'Int32',   # Nullable Int, soporta NaNs
        'Tasa': 'float32',        # Corregido: float8 no existe
        'Seguro': 'float32', 
        'Enganche_Simulado': 'float32', 
        'Enganche_Min': 'float32', 
        'Enganche_Max': 'float32', 
        'Brand': 'category',
        'Model': 'category', 
        'Version': 'category', 
        'Tipo': 'category', 
        'Total_a_Pagar': 'float64', 
        'Interes': 'float32', 
        'Interes_%': 'float32',
        'Enganche_Min_%': 'float32', 
        'Enganche_Max_%': 'float32', 
        'Sucursal': 'category',   
        'Year': 'float32',
        'Km': 'float32',          # float32 soporta NaN
        'Caja': 'category',       # Optimizado: string -> category (Soporta NaN)
        'Oferta': 'Int8'          # Corregido syntax. 'Int8' soporta NaN (<NA>)
    }

    cols_to_load = list(dtypes.keys())

    try:
        #Carga optimizada
        df = pd.read_csv('data.csv', encoding='utf-8', usecols=lambda c: c in cols_to_load, dtype=dtypes)
        return df
        
    except FileNotFoundError:
        st.error("No se encontro el archivo CSV. Asegúrate de subir 'data.csv' a la misma carpeta.")
        st.stop()

@st.cache_resource
def load_and_train_model(term=12, aliado_flag=False):

    df = load_data()

    # Limpieza
    clean_term_cars = get_term_data(df, term, aliado_flag=aliado_flag)
    
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
    
    return df_results

def display_sidebar(terms):
    st.sidebar.header("🔍 Explorador de Modelos")
    selected_data_opt = st.sidebar.checkbox("Considerar a los aliados de kavak", value=False)

    all_terms = sorted(terms)
    selected_term = st.sidebar.selectbox("Selecciona un Plazo", all_terms)

    if selected_data_opt:
        # Cargamos y entranamos al modelo
        df_results = load_and_train_model(selected_term, selected_data_opt)
    else:
        df_results = load_and_train_model(selected_term, selected_data_opt)

    all_brands = sorted(df_results['Brand'].unique())
    all_brands.insert(0, "Todas las marcas")
    selected_brand = st.sidebar.selectbox("Selecciona una Marca", all_brands)

    models_of_brand = sorted(df_results[df_results['Brand'] == selected_brand]['Model'].unique())
    models_of_brand.insert(0, "Todos los modelos")
    selected_model = st.sidebar.selectbox("Selecciona un Modelo", models_of_brand)

    all_years = sorted(df_results[df_results['Model'] == selected_model]['Year'].unique())
    all_years = [int(year) for year in all_years]
    all_years.insert(0, "Todos los años")
    selected_year = st.sidebar.selectbox("Selecciona un Año", all_years)

    return df_results, selected_brand, selected_model, selected_year, selected_data_opt

# SECCION 1: INSIGHTS
def display_insights_section(df_results):
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

def axis_abstraction(selected_model, model_data, filter_label, x_axis='Interes_%', y_axis='Km'):
    fig_model = px.scatter(
        model_data,
        x=x_axis, 
        y=y_axis, 
        color=filter_label,
        #symbol='Tipo', # Forma del punto
        hover_data=['ID_Auto', 'Model', 'Precio', 'Year', 'Sucursal', 'Total_a_Pagar'],
        title=f'{x_axis} vs {y_axis}: {selected_model}',
        height=600
    )
    fig_model.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
    event = st.plotly_chart(fig_model, width='stretch', selection_mode='points', key='ID_Auto', on_select='rerun')
    return event

# SECCION 2: ANALISIS POR MODELO
def display_analysis_model_section(df_results, selected_brand, selected_model, selected_year):
    st.markdown("---")
    st.header(f"2. Analisis Profundo: {selected_model}")

    # Filtrado de datos
    brand_mask = (df_results['Brand'] == selected_brand) | (selected_brand == 'Todas las marcas')
    model_mask = (df_results['Model'] == selected_model) | (selected_model == 'Todos los modelos')
    year_mask = (df_results['Year'] == selected_year) | (selected_year == 'Todos los años')
    model_data_simulator = df_results[brand_mask & model_mask]
    model_data = df_results[brand_mask & model_mask & year_mask]

    if model_data.empty:
        st.warning("No hay suficientes datos para este modelo.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precio Promedio", f"${model_data['Precio'].mean():,.0f}")
        m2.metric("Kilometraje Promedio", f"{model_data['Km'].mean():,.0f} km")
        m3.metric("Unidades Disponibles", len(model_data))
        m4.metric("Ciudades", model_data['Sucursal'].nunique())

        with st.expander("Configurar Ejes y Filtros de la Grafica", expanded=False):

            col3, col4, col5 = st.columns(3, gap='small', vertical_alignment='top', width='stretch')
            
            with col3:
                x_axis_labels = ['Precio', 'Km', 'Interes_%', 'Total_a_Pagar']
                y_axis_labels = ['Precio', 'Km', 'Interes_%', 'Total_a_Pagar']
                x_axis = st.segmented_control("Eje X: ", x_axis_labels, selection_mode='single')
                if x_axis:
                    y_axis_labels.remove(x_axis)
            
            with col4:
                y_axis = st.segmented_control("Eje y: ", y_axis_labels, selection_mode='single')
            
            with col5:
                segmentation_options = ['Caja', 'Sucursal', 'Segment', 'Year']
                if selected_brand != 'Todas las marcas':
                    segmentation_options.extend(['Model'])
                    if selected_model != 'Todos los modelos':
                        segmentation_options.extend(['Version'])

                filter_label = st.segmented_control("Filtrado por: ", segmentation_options, default='Segment', selection_mode='single')

        if x_axis and y_axis:
            event = axis_abstraction(selected_model, model_data, filter_label, x_axis, y_axis)
        else:
            event = axis_abstraction(selected_model, model_data, filter_label)

        #Muestra en detalle los autos seleccionados en la grafica
        points = [point['customdata'][0] for point in event['selection']['points']]
        if points:
            df_selected_points = model_data[model_data['ID_Auto'].isin(points)]
            st.dataframe(df_selected_points)

    del model_data
    return model_data_simulator


# SECCION 3: SIMULADOR
def display_simulator_section(selected_model, selected_data_opt, selected_brand, model_data_simulator):
    st.markdown("---")
    st.header("3. Evaluador de Ofertas")
    if selected_model != "Todos los modelos" and selected_data_opt == False:
        st.write(f"¿Viste un {selected_brand} {selected_model} y quieres saber si el precio es justo?")

        sc1, sc2, sc3 = st.columns(3)
        input_km = sc1.number_input("Kilometraje", value=50000, step=1000)
        input_price = sc2.number_input("Precio ($)", value=250000, step=5000)
        input_year = sc3.number_input("Año", value=2020, step=1)

        # Calculamos referencia
        modelos_de_referencia = model_data_simulator[
            (model_data_simulator['Km'] > input_km - 10000) & 
            (model_data_simulator['Km'] < input_km + 10000) &
            (model_data_simulator['Year'] == input_year)
            ]

        if st.button("Evaluar Precio"):
            if modelos_de_referencia.empty:
                st.warning(f"No tenemos suficientes datos de {selected_model} del año {input_year} para comparar.")
            else:
                avg_market = modelos_de_referencia['Precio'].mean()
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
    else:
        st.write(f"Selecciona un modelo para poder acceder al Evaluador de Ofertas. Y desactiva la opcion 'Considerar a aliados de kavak'.")

def main():
    gc.collect()

    st.set_page_config(page_title="Kavak Market Analyzer", layout="wide")
    #pd.options.display.float_format = '{:,.0f}'.format

    st.title("Valuador de Mercado de Autos Seminuevos")
    st.markdown("""
    Esta herramienta utiliza **Inteligencia Artificial (K-Means Clustering)** para identificar oportunidades de compra 
    en el mercado de autos seminuevos en México. Detecta anomalías de precio y clasifica los vehículos por su ciclo de vida.
    """)

    terms = get_terms()

    df_results, selected_brand, selected_model, selected_year, selected_data_opt = display_sidebar(terms)

    display_insights_section(df_results)

    model_data_simulator = display_analysis_model_section(df_results, selected_brand, selected_model, selected_year)

    display_simulator_section(selected_model, selected_data_opt, selected_brand, model_data_simulator)

main()