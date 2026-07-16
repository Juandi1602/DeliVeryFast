import streamlit as st
import folium
import osmnx as ox
import networkx as nx
import re
import joblib
import pandas as pd
from urllib.parse import unquote
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim

st.set_page_config(page_title="DeliVeryFast", page_icon="🛵", layout="wide")

# ── CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

html, body, .stApp { background-color: #0D0D0D !important; font-family: 'Inter', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111111 !important;
    border-right: 2px solid #FF6B35 !important;
    min-width: 550px !important;
    max-width: 550px !important;
}
[data-testid="stSidebar"] section {
    padding: 24px 20px !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 0 !important;
}
[data-testid="stSidebar"] .stSelectbox,
[data-testid="stSidebar"] .stTextInput,
[data-testid="stSidebar"] .stSlider {
    margin-bottom: 12px !important;
}

/* Quitar padding del iframe de voz */
iframe {
    margin: 0 !important;
    padding: 0 !important;
    display: block !important;
}

/* Inputs */
.stTextInput input {
    background: #1A1A1A !important;
    border: 1px solid #2A2A2A !important;
    border-radius: 8px !important;
    color: #F5F5F5 !important;
    font-size: 14px !important;
}
.stTextInput input:focus {
    border-color: #FF6B35 !important;
    box-shadow: 0 0 0 2px rgba(255,107,53,0.15) !important;
}

/* Selectbox */
[data-baseweb="select"] > div {
    background: #1A1A1A !important;
    border: 1px solid #2A2A2A !important;
    border-radius: 8px !important;
    color: #F5F5F5 !important;
}

/* Slider */
[data-testid="stSlider"] [role="slider"] { background: #FF6B35 !important; }
[data-testid="stSlider"] [data-testid="stSliderThumb"] { background: #FF6B35 !important; }

/* Botones */
.stButton > button {
    background: linear-gradient(135deg, #FF6B35 0%, #E84A1A 100%) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-size: 15px !important; padding: 12px 20px !important;
    width: 100% !important; transition: all 0.2s ease !important;
    letter-spacing: 0.3px !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(255,107,53,0.35) !important;
}

/* Métricas */
[data-testid="metric-container"] {
    background: #1A1A1A !important;
    border: 1px solid #252525 !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
}
[data-testid="stMetricLabel"] p { color: #CCCCCC !important; font-size: 14px !important; letter-spacing: 0.5px; font-weight: 600 !important; }
[data-testid="stMetricValue"] { color: #FF6B35 !important; font-size: 30px !important; font-weight: 800 !important; }
[data-testid="stMetricDelta"] { font-size: 12px !important; }

/* Labels */
label, .stSelectbox label, .stTextInput label, .stSlider label {
    color: #DDDDDD !important; font-size: 13px !important;
    text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600 !important;
}

/* Info/Success/Error */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* Divider */
hr { border-color: #1F1F1F !important; margin: 20px 0 !important; }

/* Spinner */
[data-testid="stSpinner"] p { color: #FF6B35 !important; }

/* Reducir espacio debajo del mapa */
[data-testid="stIFrame"] {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}
.element-container:has(iframe) {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}

</style>
""", unsafe_allow_html=True)

# ── Cargar mapa ──
@st.cache_resource
def cargar_mapa_lima():
    return ox.load_graphml('lima_map.graphml')

# ── Modelo entrenado (VotingRegressor: SVR + Ridge, R²=0.82) ──
@st.cache_resource
def cargar_modelo():
    modelo = joblib.load('modelo_final.pkl')
    le_weather = joblib.load('le_weather.pkl')
    le_traffic = joblib.load('le_traffic.pkl')
    le_time = joblib.load('le_time.pkl')
    le_vehicle = joblib.load('le_vehicle.pkl')
    return modelo, le_weather, le_traffic, le_time, le_vehicle

modelo_ml, le_weather, le_traffic, le_time, le_vehicle = cargar_modelo()

# Mapear valores en español (UI) a los valores en inglés con los que se entrenó el modelo
CLIMA_EN    = {'Despejado': 'Clear', 'Ventoso': 'Windy', 'Neblina': 'Foggy', 'Lluvioso': 'Rainy', 'Nevado': 'Snowy'}
TRAFICO_EN  = {'Bajo': 'Low', 'Medio': 'Medium', 'Alto': 'High'}
HORA_EN     = {'Mañana': 'Morning', 'Tarde': 'Afternoon', 'Noche': 'Evening', 'Madrugada': 'Night'}
VEHICULO_EN = {'Moto': 'Car', 'Scooter': 'Scooter', 'Bicicleta': 'Bike'}

# Mismos códigos de tráfico/clima usados como feature engineering en el notebook
TRAFICO_NUM = {'Low': 1, 'Medium': 2, 'High': 3}

def predecir_tiempo_total(distancia_km, vehiculo, clima, trafico, hora, preparacion, experiencia):
    """Predice el tiempo TOTAL de entrega (incluye preparación) usando el modelo entrenado."""
    clima_en    = CLIMA_EN[clima]
    trafico_en  = TRAFICO_EN[trafico]
    hora_en     = HORA_EN[hora]
    vehiculo_en = VEHICULO_EN[vehiculo]

    distancia_al_cuadrado = distancia_km ** 2
    distancia_x_trafico = distancia_km * TRAFICO_NUM[trafico_en]

    fila = pd.DataFrame([{
        'Distance_km': distancia_km,
        'Distancia_al_cuadrado': distancia_al_cuadrado,
        'Distancia_x_Trafico': distancia_x_trafico,
        'Weather_enc': le_weather.transform([clima_en])[0],
        'Traffic_Level_enc': le_traffic.transform([trafico_en])[0],
        'Time_of_Day_enc': le_time.transform([hora_en])[0],
        'Vehicle_Type_enc': le_vehicle.transform([vehiculo_en])[0],
        'Preparation_Time_min': preparacion,
        'Courier_Experience_yrs': experiencia,
    }])

    pred = modelo_ml.predict(fila)[0]
    return max(round(pred), 1)

@st.cache_data
def reverse_geocode(lat, lon):
    geolocator = Nominatim(user_agent='delivery_app')
    try:
        loc = geolocator.reverse((lat, lon), exactly_one=True, language='es')
        return loc.address if loc else None
    except Exception:
        return None

@st.cache_data
def obtener_segmentos_ruta(_G, ruta, vehiculo, clima, trafico, hora, experiencia):
    segmentos = []
    ultimo_nombre = None
    for u, v in zip(ruta[:-1], ruta[1:]):
        data = _G.get_edge_data(u, v)
        if isinstance(data, dict):
            edge = next(iter(data.values()))
        else:
            edge = data
        nombre = edge.get('name')
        if not nombre:
            nombre = edge.get('highway')
            if isinstance(nombre, list):
                nombre = ', '.join(nombre)
        if not nombre:
            nombre = 'Calle sin nombre'
        longitud = edge.get('length', 0)
        tiempo = predecir_tiempo_total(longitud / 1000, vehiculo, clima, trafico, hora, 0, experiencia)
        if nombre == ultimo_nombre and segmentos:
            segmentos[-1]['length'] += longitud
            segmentos[-1]['time'] += tiempo
        else:
            segmentos.append({'name': nombre, 'length': longitud, 'time': tiempo})
            ultimo_nombre = nombre
    return segmentos

# ── Parsear voz ──
def parsear_voz(texto):
    texto = texto.lower()
    resultado = {}

    texto = re.sub(r'[,.;:]+', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    texto = re.sub(r'^(?:quiero|quiero ir|voy a|vamos a|me gustaria|me gustaría|necesito|llevarme|ir|salir|partir)\s+', '', texto)

    def limpiar_lugar(texto):
        texto = texto.strip()
        texto = re.sub(r'^(?:the|a|an|of|la|el|al|del|de|desde|hasta|para|hacia)\s+', '', texto, flags=re.I)
        return texto.strip()

    # Extraer origen y destino con varios conectores: de/desde/hasta/a la/al/para
    # Usar lookahead que permite parámetros o fin de cadena, para capturar destinos con varias palabras
    patrones = [
        r'(?:de|desde|desde la|desde el|desde el barrio|del)\s+(.+?)\s+(?:a|hasta|hacia|para|a la|al|a los|a las)\s+(.+?)(?=(?:\s+(?:clima|tr[aá]fico|trafico|hora|veh[ií]culo|vehiculo|preparaci[oó]n|preparacion|experiencia))|$)',
        r'(?:de|desde|desde la|desde el|desde el barrio|del)\s+(.+?)\s+(?:a|hasta|hacia|para)\s+(.+?)(?=(?:\s+(?:clima|tr[aá]fico|trafico|hora|veh[ií]culo|vehiculo|preparaci[oó]n|preparacion|experiencia))|$)',
    ]

    combined_matched = False
    # debug info
    debug_matches = []
    for patron in patrones:
        m = re.search(patron, texto)
        if m:
            origen_raw = m.group(1).strip()
            destino_raw = m.group(2).strip()
            debug_matches.append({'pattern': patron, 'origen_group': origen_raw, 'destino_group': destino_raw})
            origen_raw = re.sub(r'^(?:ir|vamos|llevar(?:me)?|salir|partir|desde|hasta)\s+', '', origen_raw)
            destino_raw = re.sub(r'^(?:la|el|al|a la|a los|a las|hasta|hacia|para)\s+', '', destino_raw)
            origen_raw = limpiar_lugar(origen_raw)
            destino_raw = limpiar_lugar(destino_raw)
            resultado['origen'] = origen_raw.title() + ', Lima, Perú'
            resultado['destino'] = destino_raw.title() + ', Lima, Perú'
            combined_matched = True
            break

    if not combined_matched:
        m = re.search(r'\b(?:de|desde|desde la|desde el|del)\s+(.+?)\s+(?:a|hasta|hacia|para|a la|al|a los|a las)\s+(.+?)(?=(?:\s+(?:clima|tr[aá]fico|trafico|hora|veh[ií]culo|vehiculo|preparaci[oó]n|preparacion|experiencia))|$)', texto)
        if m:
            origen_raw = limpiar_lugar(m.group(1).strip())
            destino_raw = limpiar_lugar(m.group(2).strip())
            debug_matches.append({'pattern': 'fallback', 'origen_group': origen_raw, 'destino_group': destino_raw})
            resultado['origen'] = origen_raw.title() + ', Lima, Perú'
            resultado['destino'] = destino_raw.title() + ', Lima, Perú'

    clima_vals = {
        'despejado': 'Despejado', 'soleado': 'Despejado', 'nublado': 'Despejado',
        'ventoso': 'Ventoso', 'viento': 'Ventoso',
        'lluvioso': 'Lluvioso', 'lluvia': 'Lluvioso',
        'neblina': 'Neblina', 'niebla': 'Neblina',
        'nevado': 'Nevado', 'nieve': 'Nevado',
    }
    for clave, valor in clima_vals.items():
        if clave in texto:
            resultado['clima'] = valor
            break

    trafico_vals = {'bajo': 'Bajo', 'medio': 'Medio', 'alto': 'Alto', 'tránsito': 'Alto', 'transito': 'Alto'}
    for clave, valor in trafico_vals.items():
        if clave in texto or f'tráfico {clave}' in texto or f'trafico {clave}' in texto:
            resultado['trafico'] = valor
            break

    hora_vals = {'mañana': 'Mañana', 'manana': 'Mañana', 'tarde': 'Tarde', 'noche': 'Noche', 'madrugada': 'Madrugada', 'amanecer': 'Mañana', 'atardecer': 'Tarde'}
    for clave, valor in hora_vals.items():
        if clave in texto:
            resultado['hora'] = valor
            break

    vehiculo_vals = {'moto': 'Moto', 'motocicleta': 'Moto', 'scooter': 'Scooter', 'bicicleta': 'Bicicleta', 'bici': 'Bicicleta', 'auto': 'Moto', 'carro': 'Moto', 'coche': 'Moto'}
    for clave, valor in vehiculo_vals.items():
        if clave in texto:
            resultado['vehiculo'] = valor
            break

    numero_palabras = {
        'cero': 0, 'uno': 1, 'dos': 2, 'tres': 3, 'cuatro': 4, 'cinco': 5,
        'seis': 6, 'siete': 7, 'ocho': 8, 'nueve': 9, 'diez': 10,
        'once': 11, 'doce': 12,
    }

    for campo, patron in [('prep', r'preparaci[oó]n\s+([0-9]+|[a-záéíóú]+)'), ('exp', r'experiencia\s+([0-9]+|[a-záéíóú]+)')]:
        m = re.search(patron, texto)
        if m:
            valor = m.group(1).strip()
            if valor.isdigit():
                resultado[campo] = int(valor)
            elif valor in numero_palabras:
                resultado[campo] = numero_palabras[valor]

    # attach debug info
    if debug_matches:
        resultado['_debug_matches'] = debug_matches
    else:
        resultado['_debug_matches'] = []

    return resultado

# ── Voz desde query params ──
voz_texto = st.query_params.get('voz', '')
voz = parsear_voz(voz_texto) if voz_texto else {}

# ── Session state ──
defaults = {'origen': 'Miraflores, Lima, Perú', 'destino': 'San Isidro, Lima, Perú',
            'clima': 'Despejado', 'trafico': 'Bajo', 'hora': 'Mañana',
            'vehiculo': 'Moto', 'prep': 10, 'exp': 2}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v
if 'show_voice_panel' not in st.session_state:
    st.session_state.show_voice_panel = False
if voz:
    for k, v in voz.items(): st.session_state[k] = v

# ── Diccionarios ──
clima_map    = {'Despejado':'Clear','Ventoso':'Windy','Lluvioso':'Rainy','Neblina':'Foggy','Nevado':'Snowy'}
trafico_map  = {'Bajo':'Low','Medio':'Medium','Alto':'High'}
hora_map     = {'Mañana':'Morning','Tarde':'Afternoon','Noche':'Evening','Madrugada':'Night'}
vehiculo_map = {'Moto':'Car','Scooter':'Scooter','Bicicleta':'Bike'}
iconos       = {'Moto':'🏍️','Scooter':'🛵','Bicicleta':'🚲'}

# ── Componente de voz ──
VOZ_JS = """
<script>
let recognition = null, escuchando = false;
function toggleVoz() {
    const btn = document.getElementById('btn-voz');
    const output = document.getElementById('texto-voz');
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        output.innerText = 'Usa Chrome.'; return;
    }
    if (escuchando) { recognition.stop(); return; }
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SR();
    recognition.lang = 'es-PE';
    recognition.continuous = false;
    recognition.interimResults = false;
    btn.innerText = '🔴 Escuchando...';
    btn.style.background = '#dc2626';
    escuchando = true;
    recognition.onresult = function(event) {
        const texto = event.results[0][0].transcript;
        output.innerText = '"' + texto + '"';
        escuchando = false;
        btn.innerText = '🎙️ Hablar';
        btn.style.background = '#FF6B35';
        const url = new URL(window.parent.location.href);
        url.searchParams.set('voz', texto);
        window.parent.history.pushState({}, '', url.toString());
        window.parent.location.reload();
    };
    recognition.onerror = function(e) {
        output.innerText = 'Error: ' + e.error;
        escuchando = false;
        btn.innerText = '🎙️ Hablar';
        btn.style.background = '#FF6B35';
    };
    recognition.onend = function() {
        if (escuchando) { escuchando = false; btn.innerText = '🎙️ Hablar'; btn.style.background = '#FF6B35'; }
    };
    recognition.start();
    output.innerText = 'Escuchando...';
}
</script>
<div style="background:#121212;border:1px solid #2A2A2A;border-radius:10px;padding:12px;margin:0;width:100%;box-sizing:border-box;font-family:sans-serif;">
    <div style="color:#E5E5E5;font-weight:700;font-size:13px;letter-spacing:0.8px;margin-bottom:6px;">🎙️ Voz opcional</div>
    <div style="color:#BBBBBB;font-size:11px;margin-bottom:10px;line-height:1.4;">Toca y di tu ruta en voz alta si prefieres evitar escribir.</div>
    <button id="btn-voz" onclick="toggleVoz()"
        style="background:#FF6B35;color:white;border:none;padding:8px 0;border-radius:8px;cursor:pointer;font-size:13px;font-weight:700;width:100%;letter-spacing:0.5px;">
        🎙️ Hablar
    </button>
    <div style="margin-top:8px;color:#999999;font-size:11px;">
        Escuché: <span id="texto-voz" style="color:#FF6B35;font-style:italic;">—</span>
    </div>
</div>
"""

# ── SIDEBAR ──
with st.sidebar:
    col1, col2 = st.columns([9,1])
    with col1:
        st.markdown("""
        <div style="padding:10px 0 20px 0;">
            <div style="font-size:30px;font-weight:900;color:#FF6B35;letter-spacing:-0.5px;">🛵 DeliVeryFast</div>
            <div style="font-size:12px;color:#999;letter-spacing:1px;margin-top:2px;">LIMA · PREDICCIÓN DE DELIVERY</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        # Mic icon button (subtle)
        if st.button('🎙️', key='mic_icon_btn', help='Activar/desactivar voz'):
            st.session_state.show_voice_panel = not st.session_state.get('show_voice_panel', False)

    if st.session_state.show_voice_panel:
        with st.expander('Herramienta de voz', expanded=True):
            st.components.v1.html(VOZ_JS, height=150)

    if voz_texto:
        st.success(f'✅ Voz aplicada')
        if st.button('🗑️ Limpiar voz'):
            st.query_params.clear()
            for k in defaults: st.session_state[k] = defaults[k]
            st.rerun()

    st.markdown('<div style="color:#FF6B35;font-size:18px;font-weight:700;letter-spacing:1px;margin:16px 0 8px 0;">📍 UBICACIONES</div>', unsafe_allow_html=True)
    origen  = st.text_input('Punto de partida', value=st.session_state.origen)
    destino = st.text_input('Punto de llegada', value=st.session_state.destino)

    st.markdown('<div style="color:#FF6B35;font-size:18px;font-weight:700;letter-spacing:1px;margin:16px 0 8px 0;">🌦️ CONDICIONES</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        clima   = st.selectbox('Clima', list(clima_map.keys()), index=list(clima_map.keys()).index(st.session_state.clima))
        hora    = st.selectbox('Hora', list(hora_map.keys()), index=list(hora_map.keys()).index(st.session_state.hora))
    with c2:
        trafico  = st.selectbox('Tráfico', list(trafico_map.keys()), index=list(trafico_map.keys()).index(st.session_state.trafico))
        vehiculo = st.selectbox('Vehículo', list(vehiculo_map.keys()), index=list(vehiculo_map.keys()).index(st.session_state.vehiculo))

    st.markdown('<div style="color:#FF6B35;font-size:18px;font-weight:700;letter-spacing:1px;margin:16px 0 8px 0;">⚙️ PARÁMETROS</div>', unsafe_allow_html=True)
    prep = st.slider('Preparación (min)', 1, 40, st.session_state.prep)
    exp  = st.slider('Experiencia (años)', 0, 10, st.session_state.exp)

    st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)

# ── ÁREA PRINCIPAL ──
with st.spinner('Cargando mapa de Lima...'):
    G = cargar_mapa_lima()

mid_lat = -12.0464
mid_lon = -77.0428

mapa = folium.Map(location=[mid_lat, mid_lon], zoom_start=12, tiles='OpenStreetMap')
route_valid = False
route_message = None
route_message_type = 'info'

tiempo_viaje = None

tiempo_total = None

distancia = None

if origen.strip() and destino.strip():
    if origen.strip().lower() == destino.strip().lower():
        route_message = 'El punto de partida y llegada no pueden ser iguales.'
        route_message_type = 'error'
    else:
        with st.spinner('Calculando ruta...'):
            try:
                geolocator = Nominatim(user_agent='delivery_app')
                loc1 = geolocator.geocode(origen)
                loc2 = geolocator.geocode(destino)

                if loc1 is None or loc2 is None:
                    route_message = 'No se encontró una de las direcciones.'
                    route_message_type = 'error'
                else:
                    nodo1 = ox.nearest_nodes(G, loc1.longitude, loc1.latitude)
                    nodo2 = ox.nearest_nodes(G, loc2.longitude, loc2.latitude)
                    ruta = nx.shortest_path(G, nodo1, nodo2, weight='length')

                    distancia = sum(
                        G[ruta[i]][ruta[i+1]][0]['length']
                        for i in range(len(ruta)-1)) / 1000

                    tiempo_total = predecir_tiempo_total(distancia, vehiculo, clima, trafico, hora, prep, exp)
                    tiempo_viaje = tiempo_total - prep
                    route_valid = True

                    mid_lat = (loc1.latitude + loc2.latitude) / 2
                    mid_lon = (loc1.longitude + loc2.longitude) / 2
                    mapa = folium.Map(location=[mid_lat, mid_lon], zoom_start=14, tiles='OpenStreetMap')

                    puntos = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in ruta]
                    folium.PolyLine(puntos, color='#FF6B35', weight=5, opacity=0.9).add_to(mapa)
                    # Ajustar la vista del mapa para que muestre todo el trayecto
                    try:
                        lats = [p[0] for p in puntos]
                        lons = [p[1] for p in puntos]
                        bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
                        mapa.fit_bounds(bounds, padding=(30, 30))
                    except Exception:
                        try:
                            mapa.fit_bounds(puntos)
                        except Exception:
                            pass

                    folium.Marker([loc1.latitude, loc1.longitude],
                        popup=folium.Popup(f'<b>📍 Partida</b><br>{loc1.address}<br><b>Lat:</b> {loc1.latitude:.5f} <b>Lon:</b> {loc1.longitude:.5f}', max_width=280),
                        tooltip='📍 Punto de partida',
                        icon=folium.Icon(color='green', icon='play')).add_to(mapa)

                    folium.Marker([loc2.latitude, loc2.longitude],
                        popup=folium.Popup(f'<b>🏁 Llegada</b><br>{loc2.address}<br><b>Lat:</b> {loc2.latitude:.5f} <b>Lon:</b> {loc2.longitude:.5f}', max_width=280),
                        tooltip='🏁 Punto de llegada',
                        icon=folium.Icon(color='red', icon='flag')).add_to(mapa)
            except Exception as e:
                route_message = f'Error: {e}'
                route_message_type = 'error'
else:
    route_message = 'Ingresa origen y destino para ver la ruta actualizada en tiempo real.'
    route_message_type = 'info'

st_folium(mapa, width=None, height=650, returned_objects=[])

if route_message:
    if route_message_type == 'error':
        st.error(route_message)
    else:
        st.info(route_message)
elif route_valid:
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px;">
        <div style="background:#1A1A1A;border:1px solid #2A2A2A;border-radius:14px;padding:20px;text-align:center;">
            <div style="color:#AAAAAA;font-size:12px;font-weight:600;letter-spacing:1px;margin-bottom:8px;">🍳 PREPARACIÓN</div>
            <div style="color:#FF6B35;font-size:32px;font-weight:900;">{prep}</div>
            <div style="color:#666;font-size:12px;">minutos</div>
        </div>
        <div style="background:#1A1A1A;border:1px solid #2A2A2A;border-radius:14px;padding:20px;text-align:center;">
            <div style="color:#AAAAAA;font-size:12px;font-weight:600;letter-spacing:1px;margin-bottom:8px;">🛵 VIAJE</div>
            <div style="color:#FF6B35;font-size:32px;font-weight:900;">{tiempo_viaje}</div>
            <div style="color:#666;font-size:12px;">minutos</div>
        </div>
        <div style="background:#FF6B35;border:1px solid #FF6B35;border-radius:14px;padding:20px;text-align:center;">
            <div style="color:#fff;font-size:12px;font-weight:600;letter-spacing:1px;margin-bottom:8px;">⏱️ TOTAL ESTIMADO</div>
            <div style="color:#fff;font-size:32px;font-weight:900;">{tiempo_total}</div>
            <div style="color:#FFD4C2;font-size:12px;">minutos</div>
        </div>
        <div style="background:#1A1A1A;border:1px solid #2A2A2A;border-radius:14px;padding:20px;text-align:center;">
            <div style="color:#AAAAAA;font-size:12px;font-weight:600;letter-spacing:1px;margin-bottom:8px;">📏 DISTANCIA</div>
            <div style="color:#FF6B35;font-size:32px;font-weight:900;">{distancia:.2f}</div>
            <div style="color:#666;font-size:12px;">kilómetros</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    otros = [v for v in ['Moto', 'Scooter', 'Bicicleta'] if v != vehiculo]
    cards_html = ''
    for veh in otros:
        t_otro = predecir_tiempo_total(distancia, veh, clima, trafico, hora, prep, exp)
        dif = t_otro - tiempo_total
        signo = '+' if dif > 0 else ''
        color_dif = '#ef4444' if dif > 0 else '#22c55e'
        cards_html += (
            '<div style="background:#1A1A1A;border:1px solid #2A2A2A;border-radius:14px;padding:20px;text-align:center;">'
            f'<div style="color:#AAAAAA;font-size:12px;font-weight:600;letter-spacing:1px;margin-bottom:8px;">{iconos[veh]} {veh.upper()}</div>'
            f'<div style="color:#FF6B35;font-size:32px;font-weight:900;">{t_otro}</div>'
            '<div style="color:#666;font-size:12px;">minutos</div>'
            f'<div style="color:{color_dif};font-size:12px;margin-top:6px;font-weight:600;">{signo}{dif} min vs tu vehículo</div>'
            '</div>'
        )

    cards_section = (
        '<div style="color:#FFFFFF;font-size:18px;font-weight:700;letter-spacing:1px;margin-bottom:12px;">🔄 COMPARACIÓN DE VEHÍCULOS</div>'
        '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-bottom:24px;">'
        f'{cards_html}'
        '</div>'
    )
    st.markdown(cards_section, unsafe_allow_html=True)
else:
    st.info('Selecciona origen y destino válidos para ver la ruta actualizada en tiempo real.')
