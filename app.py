import pandas as pd
import pydeck as pdk
import streamlit as st
import pydeck as pdk
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import requests

DATA_URL = "https://raw.githubusercontent.com/etodd67/streamlit_water_app/main/Harbor_Water_Quality3.csv"

def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows, parse_dates=[['Sample_Date', 'Sample_Time']])
    data.dropna(subset=['lat', 'lon'], inplace=True)
    return data

# Load data initially
data = load_data(1000000)
original_data = data

# Add a button to reload the data
#if st.button("Reload Data"):
 #   data = load_data(100000)
  #  st.success("Data reloaded.")

# Convert columns to numeric, setting non-convertible values to NaN
numeric_columns = ["Top_Sample_Temperature", "Top_Salinity", "Percentage_O2_Saturation_Top_Sample",
                    "Winkler_Method_Top_Dissolved_Oxygen_mg_per_L", "Top_PH", "Top_Total_Coliform_Cells_per_100_mL",
                    "Top_Fecal_Coliform_Bacteria_Cells_per_100mL", "Top_Enterococci_Bacteria_Cells_per_100mL",
                    "Top_Ammonium_mg_per_L", "Top_Total_Suspended_Solid_mg_per_L", "lon", "lat"]

for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with NaN in 'lat' and 'lon'
data.dropna(subset=['lat', 'lon'], inplace=True)

st.title("Harbor Water Quality in New York City - Streamlit Dashboard 🗽  🌊")
st.markdown("Coded in python to explore the NYC Open Data harbor water sampling information - updated November 29, 2023")

# Dropdown to select numeric column for 3D map
selected_column = st.selectbox("Select Numeric Column for 3D Map", numeric_columns[:-2])  # Exclude 'lat' and 'lon'

selected_data = data.copy()
selected_data.dropna(subset=[selected_column], inplace=True)

# Create a Pydeck 3D Hexagon Extruded Map with adjusted parameters
hexagon_layer = pdk.Layer(
    "HexagonLayer",
    selected_data,
    get_position=["lon", "lat"],
    auto_highlight=True,
    elevation_scale=4,
    elevation_range=[0, 1000],
    extruded=True,
    coverage=1,
    radius=120,
    opacity=0.7,
    pickable=True,
    color_range=[
        [255, 255, 0, 200],
        [255, 165, 0, 200],
        [255, 0, 0, 200],
    ],
    color_range_type="linear",
    color_range_steps=6,
    get_fill_color=f"[255 * {selected_column}, 255 * (1 - {selected_column}), 0, 200]",  # Gradient from yellow to red based on normalized value
)

view_state = pdk.ViewState(
    longitude=-74.006,
    latitude=40.7128,
    zoom=10,
    pitch=90,
    min_zoom=5,
    max_zoom=15,
)

r = pdk.Deck(
    layers=[hexagon_layer],
    initial_view_state=view_state,
    tooltip={"html": f"<b>{selected_column}</b>: {{elevationValue}}"},
)

st.pydeck_chart(r)

# Show raw data
if st.checkbox("Show Raw Data", False):
    st.subheader('First 50 Rows of Raw Data')
    st.write(selected_data.head(50))

# Function to get distribution type
def get_distribution_type(skewness, kurtosis):
    if abs(skewness) < 1 and abs(kurtosis - 3) < 1:
        return "Approximately Normal"
    elif skewness < -1 and kurtosis > 3:
        return "Moderately Negatively Skewed with Heavy Tails"
    elif skewness > 1 and kurtosis > 3:
        return "Moderately Positively Skewed with Heavy Tails"
    elif skewness < -1 and kurtosis < 3:
        return "Strongly Negatively Skewed"
    elif skewness > 1 and kurtosis < 3:
        return "Strongly Positively Skewed"
    else:
        return "Other"
        
# EPA standards for drinking and recreation
epa_standards = {
    "Top_Sample_Temperature": {"Drinking": 50, "Recreation": 75},
    "Top_Total_Suspended_Solid_mg_per_L": {"Drinking": 30, "Recreation": 158},
    "Top_PH": {"Lower Bound Drinking": 6.5, "Upper Bound Drinking": 8.5},
    "Top_Total_Coliform_Cells_per_100_mL": {"Drinking": 0, "Recreation": None},
    "Top_Ammonium_mg_per_L": {"Drinking": 0.5, "Recreation": 17},
}

table_data = []

#Plotly graphs showing the distributions of each variable
for col in numeric_columns[:-2]:  # Exclude 'lat' and 'lon'
    fig = px.histogram(original_data, x=col, title=f'Distribution of {col}', histnorm='probability density')
    fig.update_traces(opacity=0.6) 
    fig.update_layout(bargap=0.1)
    
    skewness = original_data[col].dropna().skew()
    kurtosis = original_data[col].dropna().kurt()

    # type of distribution based on skewness and kurtosis
    distribution_type = get_distribution_type(skewness, kurtosis)

    # text annotation
    fig.add_annotation(
        text=f"Type: {distribution_type}",
        showarrow=False,
        xref="paper", yref="paper",
        x=0.3, y=0.95,
        font=dict(size=10),
    )

    #EPA standards as vertical lines
    if col in epa_standards:
        for label, value in epa_standards[col].items():
            if value is not None:
                fig.add_shape(
                    go.layout.Shape(
                        type="line",
                        x0=value,
                        x1=value,
                        y0=0,
                        y1=1,
                        xref="x",
                        yref="paper",
                        line=dict(color="orange", width=2, dash="dash"),
                    ),
                )
                fig.add_annotation(
                    text=f"EPA {label} Standard: {value}",
                    showarrow=False,
                    x=value,
                    y=1.05,
                    xref="x",
                    yref="paper",
                    font=dict(size=10, color="orange"),
                )

    st.plotly_chart(fig)
    
#Add data to the table
    median_value = original_data[col].median()
    std_dev = original_data[col].std()

    table_data.append({
        'Variable': col,
        'Median': round(median_value, 2),
        'Standard Deviation': round(std_dev, 2)
    })

table_df = pd.DataFrame(table_data)
st.table(table_df)
    
#Time series
selected_variable = st.selectbox("Select Variable for Time Series Plot", numeric_columns[:-2])
data['Sample_Date_Sample_Time'] = pd.to_datetime(data['Sample_Date_Sample_Time'], errors='coerce')
data.dropna(subset=['Sample_Date_Sample_Time'], inplace=True)

yearly_data = data.groupby(pd.Grouper(key='Sample_Date_Sample_Time', freq='Y'))[selected_variable].median().reset_index()
yearly_median_values = data.groupby(pd.Grouper(key='Sample_Date_Sample_Time', freq='Y'))[selected_variable].median().reset_index()
smooth_yearly_median_values = yearly_median_values.interpolate(method='spline')

fig = go.Figure()

fig.add_trace(go.Scatter(x=smooth_yearly_median_values['Sample_Date_Sample_Time'],
                         y=smooth_yearly_median_values[selected_variable],
                         mode='lines', name='Smooth Yearly Median',
                         line=dict(width=4, color = 'darkgrey')))

fig.add_trace(go.Scatter(x=smooth_yearly_median_values['Sample_Date_Sample_Time'],
                         y=smooth_yearly_median_values[selected_variable].rolling(window=4).mean(),
                         mode='lines', name='Trendline',
                         line=dict(color='blue', width=2)))

fig.update_layout(title=f'Time Series of {selected_variable} - Smooth Yearly Median with Trendline',
                  xaxis_title='Year Starting',
                  yaxis_title=selected_variable,
                  showlegend=False)

st.plotly_chart(fig)




#write out python script
SCRIPT_URL = "https://raw.githubusercontent.com/etodd67/streamlit_water_app/main/app.py"
response = requests.get(SCRIPT_URL)
script_content = response.text
st.markdown("## Python Script")
st.code(script_content, language="python")
