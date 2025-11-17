import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modelado.FCN8s import FCN8s
from modelado.modeloA import UNet
from modelado.modeloC_tiny import AttentionUNetTiny
from modelado.segNet import SegNet

# Configuración de la página
st.set_page_config(
    page_title="Proyecto Data Science - UVG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("Sistema de Predicción con Modelos de Machine Learning")
st.markdown("### Universidad del Valle de Guatemala - CC3084")
st.markdown("---")

# Función para cargar modelos
@st.cache_resource
def load_models():
    """Carga los 4 modelos entrenados desde archivos .pth"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Parámetros del modelo
        num_classes = 2  # Número de clases
        
        models = {}
        model_files = {
            'FCN': 'models/best_fcn8s.pth',
            'SegNet': 'models/best_segnet_base.pth',
            'UNet': 'models/unet_modificado.pth',
            'Attention UNet': 'models/best_attention_unet_tiny.pth'
        }
        
        for name, file_path in model_files.items():
            if name == 'FCN':
                model = FCN8s(num_classes=num_classes, pretrained=True).to(device)
            elif name == 'SegNet':
                model = SegNet(in_channels=3, num_classes=num_classes).to(device)
            elif name == 'UNet':
                model = UNet(in_channels=3).to(device)
            elif name == 'Attention UNet':
                model = AttentionUNetTiny(in_channels=3, num_classes=num_classes).to(device)

            # Cargar checkpoint completo
            checkpoint = torch.load(file_path, map_location=device, weights_only=False)
            
            # Extraer solo el state_dict del modelo
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                
                epoch = checkpoint.get('epoch', 'N/A')
                dice_score = checkpoint.get('dice_score', 'N/A')
                history = checkpoint.get('history', 'N/A')
            else:
                # Si es solo el state_dict
                model.load_state_dict(checkpoint)
            
            model.eval()  # Modo evaluación
            models[name] = [model, epoch, dice_score, history]
        
        return models, device
    except FileNotFoundError as e:
        st.error(f"Error al cargar los modelos: {e}")
        st.info("Asegúrate de que los archivos .pth estén en el mismo directorio que este script.")
        return None, None
    except Exception as e:
        st.error(f"Error al cargar los modelos: {e}")
        st.info("Verifica que la arquitectura del modelo coincida con los modelos guardados.")
        return None, None

# Función para cargar métricas de rendimiento (personaliza con tus datos reales)
@st.cache_data
def load_performance_metrics():
    """Retorna las métricas de rendimiento de cada modelo"""
    # Reemplaza estos valores con las métricas reales de tus modelos
    metrics = {
        'FCN': {'Dice Score': 0.6240, 'Precision': 0.7141, 'Recall': 0.5620, 'F1-Score': 0.6240},
        'SegNet': {'Dice Score': 0.5640, 'Precision': 0.6203, 'Recall': 0.5950, 'F1-Score': 0.5604},
        'UNet': {'Dice Score': 0.82, 'Precision': 0.80, 'Recall': 0.85, 'F1-Score': 0.82},
        'Attention UNet': {'Dice Score': 0.90, 'Precision': 0.89, 'Recall': 0.91, 'F1-Score': 0.90}
    }
    return metrics

# Sidebar para navegación
st.sidebar.title("📋 Navegación")
page = st.sidebar.radio(
    "Selecciona una sección:",
    ["Inicio", "Predicción Individual", "Comparación de Modelos", "Visualizaciones"]
)

# Cargar modelos
models_data = load_models()
if models_data[0] is not None:
    models, device = models_data
else:
    models, device = None, None
metrics = load_performance_metrics()

# ============== PÁGINA DE INICIO ==============
if page == "Inicio":
    st.header("Bienvenido al Sistema de Predicción")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Acerca del Proyecto")
        st.write("""
        Esta aplicación integra 4 modelos de Machine Learning entrenados para realizar predicciones
        sobre la segmentación de vasos sanguíneos en imágenes histológicas de tejido renal. 
        
        **Funcionalidades principales:**
        - Predicción individual con cada modelo
        - Comparación de rendimiento entre modelos
        - Visualizaciones interactivas
        - Análisis de importancia de características
        """)
    
    with col2:
        st.subheader("Modelos Disponibles")
        if models:
            for model_name in models.keys():
                st.success(f"{model_name}")
        else:
            st.error("❌ No se pudieron cargar los modelos")
    
    # Mostrar métricas generales
    st.subheader("📊 Rendimiento General de los Modelos")
    if metrics:
        metrics_df = pd.DataFrame(metrics).T
        st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)

# ============== PÁGINA DE PREDICCIÓN ==============
elif page == "Predicción Individual":
    st.header("Predicción Individual")
    
    if models is None:
        st.error("No se pueden realizar predicciones sin los modelos cargados.")
    else:

        # Selección de modelo
        selected_model = st.selectbox("Selecciona un modelo:", list(models.keys()))
        
        st.subheader("Sube una imagen para segmentar:")
        
        # Subir imagen
        uploaded_file = st.file_uploader("Selecciona una imagen", type=['jpg', 'jpeg', 'png'])

        # Inicializar variables persistentes
        if "prediction" not in st.session_state:
            st.session_state.prediction = None
            st.session_state.image = None
            st.session_state.probabilities = None

        if uploaded_file is not None:
            from PIL import Image
            import torchvision.transforms as transforms
            import torch.nn.functional as F

            # Cargar imagen original
            image = Image.open(uploaded_file).convert('RGB')
            st.session_state.image = image
            original_size = image.size  # (W, H)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Imagen Original")
                st.image(image, use_column_width=True)

            # BOTÓN DE PREDICCIÓN (SE GUARDA EN SESSION_STATE)
            if st.button("🎯 Realizar Segmentación", type="primary"):
                with st.spinner("Procesando imagen..."):

                    # Preprocesar imagen
                    transform = transforms.Compose([
                        transforms.Resize((256, 256)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                             std=[0.229, 0.224, 0.225])
                    ])
                    
                    input_tensor = transform(image).unsqueeze(0).to(device)

                    # Cargar modelo
                    model, epoch, dice_score, history = models[selected_model]

                    # --- Predicción ---
                    with torch.no_grad():
                        output = model(input_tensor)  # (1,2,256,256)
                        prediction = torch.argmax(output, dim=1).cpu().numpy()[0]
                        
                        # PROBABILIDADES
                        probabilities = torch.softmax(output, dim=1).cpu().numpy()

                        # --- Redimensionar máscara a tamaño original ---
                        pred_torch = torch.from_numpy(prediction).unsqueeze(0).unsqueeze(0).float()
                        upscaled = F.interpolate(pred_torch, size=(image.height, image.width),
                                                 mode='nearest')
                        prediction_resized = upscaled.squeeze().numpy()

                    # Guardar en sesión
                    st.session_state.prediction = prediction_resized
                    st.session_state.probabilities = probabilities

            # ============================
            # SI YA TENEMOS PREDICCIÓN
            # ============================
            if st.session_state.prediction is not None:

                prediction = st.session_state.prediction
                probabilities = st.session_state.probabilities
                image = st.session_state.image

                with col2:
                    st.subheader("Máscara de Segmentación")
                    fig = px.imshow(prediction, color_continuous_scale='Viridis')
                    fig.update_layout(
                        coloraxis_showscale=True,
                        width=850,
                        height=850
                    )
                    st.plotly_chart(fig, use_container_width=False)

                # MÉTRICAS
                st.success("✅ Segmentación completada!")

                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    pixels_clase0 = np.sum(prediction == 0)
                    st.metric("Píxeles Clase 0", f"{pixels_clase0:,}")
                with col_m2:
                    pixels_clase1 = np.sum(prediction == 1)
                    st.metric("Píxeles Clase 1", f"{pixels_clase1:,}")
                with col_m3:
                    porcentaje = (pixels_clase1 / (pixels_clase0 + pixels_clase1)) * 100
                    st.metric("% Clase 1", f"{porcentaje:.2f}%")

                # Probabilidades promedio
                st.subheader("Confianza Promedio por Clase")
                avg_probs = probabilities.mean(axis=(2, 3))[0]
                prob_df = pd.DataFrame({
                    'Clase': [f'Clase {i}' for i in range(len(avg_probs))],
                    'Confianza Promedio': avg_probs
                })
                fig_prob = px.bar(prob_df, x='Clase', y='Confianza Promedio')
                st.plotly_chart(fig_prob, use_container_width=True)

                # Overlay
                # Opción de overlay
                if st.checkbox("Mostrar overlay de segmentación"):
                    import matplotlib.pyplot as plt
                    import matplotlib.colors as mcolors

                    # Imagen + máscara más pequeñas
                    fig, ax = plt.subplots(figsize=(3, 3))  # <-- TAMAÑO REDUCIDO

                    ax.imshow(image)  # imagen original

                    # Máscara reescalada al tamaño original
                    mask = prediction.astype(np.uint8)

                    cmap = mcolors.ListedColormap([
                        (0, 0, 0, 0),     # Clase 0 → transparente
                        (1, 0, 0, 0.5)    # Clase 1 → rojo semitransparente
                    ])

                    ax.imshow(mask, cmap=cmap, interpolation='nearest')
                    ax.axis('off')

                    st.pyplot(fig)


        else:
            st.info("Por favor, sube una imagen para comenzar la segmentación.")



# ============== PÁGINA DE COMPARACIÓN ==============
elif page == "Comparación de Modelos":
    st.header("Comparación de Modelos")
    
    if metrics:
        # Gráfico de barras comparativo
        st.subheader("Comparación de Métricas de Rendimiento")
        
        metrics_df = pd.DataFrame(metrics).T.reset_index()
        metrics_df.columns = ['Modelo'] + list(metrics_df.columns[1:])
        
        fig = go.Figure()
        
        for metric in ['Dice Score', 'Precision', 'Recall', 'F1-Score']:
            fig.add_trace(go.Bar(
                name=metric,
                x=metrics_df['Modelo'],
                y=metrics_df[metric],
                text=metrics_df[metric].round(3),
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Comparación de Métricas entre Modelos",
            xaxis_title="Modelo",
            yaxis_title="Valor de Métrica",
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Radar chart
        st.subheader("Gráfico Radar de Rendimiento")
        
        selected_models = st.multiselect(
            "Selecciona modelos para comparar:",
            list(metrics.keys()),
            default=list(metrics.keys())[:2]
        )
        
        if selected_models:
            fig = go.Figure()
            
            for model in selected_models:
                values = list(metrics[model].values())
                categories = list(metrics[model].keys())
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=model
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

# ============== PÁGINA DE VISUALIZACIONES ==============
elif page == "Visualizaciones":
    st.header("Curvas de aprendizaje")
    
    st.subheader("Evolución de métricas durante el entrenamiento")
    
    selected_model_curve = st.selectbox("Selecciona un modelo:", list(models.keys()) if models else [], key="curve_model")
    
    if selected_model_curve and models:
        # Extraer el history del modelo seleccionado
        model, epoch, dice_score, history = models[selected_model_curve]
        
        if (history != "N/A"):
            epochs = np.arange(1, len(history['train_loss']) + 1)
            
            # Gráfico de Loss
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=epochs, y=history['train_loss'], 
                mode='lines+markers', name='Training Loss',
                line=dict(color='blue', width=2)
            ))
            fig_loss.add_trace(go.Scatter(
                x=epochs, y=history['val_loss'], 
                mode='lines+markers', name='Validation Loss',
                line=dict(color='red', width=2)
            ))
            fig_loss.update_layout(
                title=f"Curva de Loss - {selected_model_curve}",
                xaxis_title="Época",
                yaxis_title="Loss",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig_loss, use_container_width=True)
            
            # Gráfico de Dice Score e IoU
            fig_dice = go.Figure()
            fig_dice.add_trace(go.Scatter(
                x=epochs, y=history['val_dice'], 
                mode='lines+markers', name='Validation Dice',
                line=dict(color='green', width=2)
            ))
            fig_dice.add_trace(go.Scatter(
                x=epochs, y=history['val_iou'], 
                mode='lines+markers', name='Validation IoU',
                line=dict(color='orange', width=2)
            ))
            fig_dice.update_layout(
                title=f"Curvas de Dice Score e IoU - {selected_model_curve}",
                xaxis_title="Época",
                yaxis_title="Score",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig_dice, use_container_width=True)
            
            # Gráfico de Precision, Recall y F1
            fig_metrics = go.Figure()
            fig_metrics.add_trace(go.Scatter(
                x=epochs, y=history['val_precision'], 
                mode='lines+markers', name='Precision',
                line=dict(color='purple', width=2)
            ))
            fig_metrics.add_trace(go.Scatter(
                x=epochs, y=history['val_recall'], 
                mode='lines+markers', name='Recall',
                line=dict(color='cyan', width=2)
            ))
            fig_metrics.add_trace(go.Scatter(
                x=epochs, y=history['val_f1'], 
                mode='lines+markers', name='F1-Score',
                line=dict(color='magenta', width=2)
            ))
            fig_metrics.update_layout(
                title=f"Métricas de Clasificación - {selected_model_curve}",
                xaxis_title="Época",
                yaxis_title="Score",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig_metrics, use_container_width=True)
            
            # Mostrar estadísticas finales
            st.subheader("Estadísticas de Convergencia")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                best_dice = np.max(history['val_dice'])
                best_epoch_dice = np.argmax(history['val_dice']) + 1
                st.metric("Mejor Dice Score", f"{best_dice:.4f}", 
                        delta=f"Época {best_epoch_dice}")
            
            with col2:
                final_loss = history['val_loss'][-1]
                st.metric("Loss Final", f"{final_loss:.4f}")
            
            with col3:
                final_f1 = history['val_f1'][-1]
                st.metric("F1-Score Final", f"{final_f1:.4f}")
            
            with col4:
                if selected_model_curve == "SegNet":
                    st.metric("Tiempo Promedio/Época", "89.05s", delta="Total: 2670s")
                else:
                    total_time = sum(history['epoch_times'])
                    avg_time = np.mean(history['epoch_times'])
                    st.metric("Tiempo Promedio/Época", f"{avg_time:.2f}s",
                            delta=f"Total: {total_time:.0f}s")
        else:
            st.warning("Lastimosamente no se guardó el historial de este modelo para ver la evolución")
    else:
        st.warning("Selecciona un modelo para ver sus curvas de aprendizaje")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Proyecto Data Science - 2025**  
Universidad del Valle de Guatemala  
CC3084
""")