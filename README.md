# Kavak Market Analyzer

Herramienta de análisis de mercado de autos seminuevos en México, construida con Streamlit y K-Means Clustering. Permite identificar oportunidades de compra, segmentar el mercado por perfil de vehículo y evaluar si un precio es justo.

## Funcionalidades

- **Segmentación del mercado** — agrupa los autos en tres segmentos (Alto Kilometraje, Standard, Premium) usando K-Means sobre precio, kilometraje y año.
- **Análisis profundo por modelo** — filtra por marca, modelo y año; visualiza scatter plots con ejes configurables y segmentación por caja, ciudad, segmento o versión. Los puntos son seleccionables para ver el detalle de cada unidad.
- **Evaluador de ofertas** — dado un kilometraje, precio y año ingresados, compara contra el mercado y dictamina si la oferta es BUENA, CARA o PRECIO JUSTO.
- **Soporte para autos aliados** — opción para incluir o excluir el inventario de aliados de Kavak en el análisis.

## Stack

| Librería | Uso |
|---|---|
| Streamlit | UI e interactividad |
| Pandas | Carga y manipulación de datos |
| scikit-learn | K-Means Clustering |
| Plotly | Visualizaciones interactivas |

## Instalación

```bash
git clone <repo-url>
cd kavak-analyzer
pip install -r requirements.txt
```

## Uso

```bash
streamlit run app.py
```

Coloca el archivo `data.csv` en el mismo directorio que `app.py` antes de ejecutar.

## Estructura del proyecto

```
kavak-analyzer/
├── app.py           # Aplicación principal de Streamlit
├── data.csv         # Dataset de autos seminuevos (no incluido en el repo)
└── requirements.txt # Dependencias
```

## Datos esperados (`data.csv`)

El archivo debe contener al menos las siguientes columnas:

| Columna | Tipo | Descripción |
|---|---|---|
| `ID_Auto` | string | Identificador único del auto |
| `Marca` | category | Marca del vehículo |
| `Modelo` | category | Modelo del vehículo |
| `Version` | category | Versión/trim |
| `Tipo` | category | Tipo (SUV, Sedan, etc.) |
| `Caja` | category | Transmisión (Manual / Automático) |
| `Ciudad` | category | Ciudad de venta |
| `Año` | float | Año del modelo |
| `Km` | float | Kilometraje |
| `Precio` | float | Precio de venta (MXN) |
| `Plazo` | Int16 | Plazo de financiamiento en meses |
| `Interes_%` | float | Tasa de interés total en % |
| `Total_a_Pagar` | float | Total a pagar con financiamiento |

## Lógica de segmentación

El modelo K-Means (K=3) se entrena con los Z-scores de Precio, Km y Año. Los clusters se nombran dinámicamente según el precio promedio de cada grupo: el de menor precio recibe el nombre **Alto Kilometraje**, el intermedio **Standard** y el mayor **Premium**.

> La "Barrera de Depreciación" observada empíricamente se ubica cerca de los 70,000 km, donde los autos tienden a saltar del segmento Standard al de Alto Kilometraje.
