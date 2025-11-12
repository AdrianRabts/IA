"""
Backend Flask - Sistema de Control de Calidad
Clasificación de Productos de Cómputo usando Deep Learning
Arquitecturas: YOLO11, EfficientNetV2, DETR
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
import io
import time
from datetime import datetime
import traceback
import os

# ============================================
# IMPORTACIONES PARA MODELOS (Descomentar cuando tengas los modelos)
# ============================================

# Para YOLO11
# from ultralytics import YOLO

# Para EfficientNetV2
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# Para DETR
# from transformers import DetrImageProcessor, DetrForObjectDetection
# import torch

# ============================================
# CONFIGURACIÓN DE FLASK
# ============================================

app = Flask(__name__)
CORS(app)  # Permitir peticiones desde el frontend

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Límite 16MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'

# Crear carpetas si no existen
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

# ============================================
# CONFIGURACIÓN - PRODUCTOS DE CÓMPUTO
# ============================================

# Categorías de productos de cómputo que el sistema puede clasificar
PRODUCT_CLASSES = [
    'CPU',
    'Mouse',
    'Teclado',
    'Arduino',
    'Raspberry Pi',
    'Memoria RAM',
    'Disco Duro',
    'SSD',
    'Tarjeta Gráfica',
    'Placa Madre',
    'Fuente de Poder',
    'Cable USB',
    'Adaptador',
    'Cooler/Ventilador',
    'Webcam'
]

# Estados de control de calidad
QUALITY_STATUS = {
    'APROBADO': {'color': '#4CAF50', 'icon': '✓'},
    'RECHAZADO': {'color': '#F44336', 'icon': '✗'},
    'REVISION': {'color': '#FF9800', 'icon': '⚠'}
}

# Defectos comunes detectables
DEFECT_TYPES = [
    'Rayado',
    'Doblado',
    'Roto',
    'Sucio',
    'Decolorado',
    'Faltante de componente',
    'Mal ensamblado',
    'Oxidado',
    'Sin defectos'
]

# Umbrales de confianza para control de calidad
CONFIDENCE_THRESHOLDS = {
    'APROBADO': 90.0,    # >= 90% de confianza
    'REVISION': 70.0,     # 70-90% de confianza
    'RECHAZADO': 0.0      # < 70% de confianza o defectos detectados
}

# Tamaños de entrada para cada modelo
IMG_SIZE_YOLO = (640, 640)
IMG_SIZE_EFFICIENT = (224, 224)
IMG_SIZE_DETR = (800, 800)

# ============================================
# CLASE PARA GESTIONAR MODELOS
# ============================================

class QualityControlSystem:
    def __init__(self):
        """Inicializa el sistema de control de calidad"""
        self.yolo_model = None
        self.efficient_model = None
        self.detr_model = None
        self.detr_processor = None
        self.inspection_count = 0  # Contador de inspecciones
        
        print("🔄 Inicializando Sistema de Control de Calidad...")
        self.load_models()
        print("✅ Sistema listo para inspección")
    
    def load_models(self):
        """
        Carga los 3 modelos de Deep Learning
        
        TODO: Tu compañero debe proporcionar los modelos entrenados
        Formato esperado:
        - YOLO11: archivo .pt
        - EfficientNetV2: archivo .h5 o .keras
        - DETR: carpeta con modelo fine-tuned
        """
        
        # ==================== YOLO11 ====================
        try:
            # TODO: Reemplazar con ruta del modelo entrenado
            # self.yolo_model = YOLO('models/yolo11_computer_parts.pt')
            print("⚠️  YOLO11: Modo demostración (modelo no cargado)")
            print("    📁 Ubicación esperada: models/yolo11_computer_parts.pt")
        except Exception as e:
            print(f"❌ Error cargando YOLO11: {e}")
        
        # ==================== EfficientNetV2 ====================
        try:
            # TODO: Reemplazar con ruta del modelo entrenado
            # self.efficient_model = load_model('models/efficientnetv2_qc.h5')
            print("⚠️  EfficientNetV2: Modo demostración (modelo no cargado)")
            print("    📁 Ubicación esperada: models/efficientnetv2_qc.h5")
        except Exception as e:
            print(f"❌ Error cargando EfficientNetV2: {e}")
        
        # ==================== DETR ====================
        try:
            # TODO: Reemplazar con ruta del modelo entrenado
            # self.detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            # self.detr_model = DetrForObjectDetection.from_pretrained("models/detr_computer_parts")
            print("⚠️  DETR: Modo demostración (modelo no cargado)")
            print("    📁 Ubicación esperada: models/detr_computer_parts/")
        except Exception as e:
            print(f"❌ Error cargando DETR: {e}")
    
    # ==================== YOLO11: Detección y Clasificación ====================
    def inspect_with_yolo(self, image):
        """
        Inspección con YOLO11 - Detección rápida de productos
        
        Args:
            image: PIL Image
        
        Returns:
            dict: Resultados de inspección
        """
        try:
            if self.yolo_model is None:
                return self._generate_demo_inspection("YOLO11")
            
            # TODO: Implementar con modelo real
            """
            # Redimensionar imagen
            img_resized = image.resize(IMG_SIZE_YOLO)
            
            # Realizar detección
            results = self.yolo_model(img_resized, conf=0.5)
            
            # Procesar resultados
            detections = results[0].boxes.data.cpu().numpy()
            
            if len(detections) > 0:
                # Obtener mejor detección
                best_detection = detections[np.argmax(detections[:, 4])]
                class_id = int(best_detection[5])
                confidence = float(best_detection[4]) * 100
                
                # Evaluar calidad
                quality_status = self._evaluate_quality(confidence, None)
                
                return {
                    'product': PRODUCT_CLASSES[class_id],
                    'confidence': confidence,
                    'quality_status': quality_status,
                    'defects_detected': [],
                    'bbox': best_detection[:4].tolist(),
                    'inference_time': results[0].speed['inference'] / 1000,
                    'probabilities': self._calculate_class_probabilities(detections)
                }
            """
            
            return self._generate_demo_inspection("YOLO11")
            
        except Exception as e:
            print(f"❌ Error en YOLO11: {e}")
            return {'error': str(e)}
    
    # ==================== EfficientNetV2: Clasificación Precisa ====================
    def inspect_with_efficient(self, image):
        """
        Inspección con EfficientNetV2 - Clasificación de alta precisión
        
        Args:
            image: PIL Image
        
        Returns:
            dict: Resultados de inspección
        """
        try:
            if self.efficient_model is None:
                return self._generate_demo_inspection("EfficientNetV2")
            
            # TODO: Implementar con modelo real
            """
            # Preprocesar imagen
            img_resized = image.resize(IMG_SIZE_EFFICIENT)
            img_array = np.array(img_resized)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predicción
            start_time = time.time()
            predictions = self.efficient_model.predict(img_array, verbose=0)
            inference_time = time.time() - start_time
            
            # Obtener resultados
            class_id = np.argmax(predictions[0])
            confidence = float(predictions[0][class_id]) * 100
            
            # Evaluar calidad
            quality_status = self._evaluate_quality(confidence, None)
            
            # Probabilidades por clase
            probabilities = {
                PRODUCT_CLASSES[i]: float(predictions[0][i]) * 100 
                for i in range(len(PRODUCT_CLASSES))
            }
            
            return {
                'product': PRODUCT_CLASSES[class_id],
                'confidence': confidence,
                'quality_status': quality_status,
                'defects_detected': [],
                'inference_time': inference_time,
                'probabilities': probabilities
            }
            """
            
            return self._generate_demo_inspection("EfficientNetV2")
            
        except Exception as e:
            print(f"❌ Error en EfficientNetV2: {e}")
            return {'error': str(e)}
    
    # ==================== DETR: Detección con Transformers ====================
    def inspect_with_detr(self, image):
        """
        Inspección con DETR - Detection Transformer
        
        Args:
            image: PIL Image
        
        Returns:
            dict: Resultados de inspección
        """
        try:
            if self.detr_model is None or self.detr_processor is None:
                return self._generate_demo_inspection("DETR")
            
            # TODO: Implementar con modelo real
            """
            # Procesar imagen
            inputs = self.detr_processor(images=image, return_tensors="pt")
            
            # Detección
            start_time = time.time()
            outputs = self.detr_model(**inputs)
            inference_time = time.time() - start_time
            
            # Post-procesar
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.detr_processor.post_process_object_detection(
                outputs, 
                target_sizes=target_sizes, 
                threshold=0.5
            )[0]
            
            if len(results['scores']) > 0:
                # Mejor detección
                best_idx = torch.argmax(results['scores'])
                class_id = results['labels'][best_idx].item()
                confidence = results['scores'][best_idx].item() * 100
                
                # Evaluar calidad
                quality_status = self._evaluate_quality(confidence, None)
                
                return {
                    'product': PRODUCT_CLASSES[class_id],
                    'confidence': confidence,
                    'quality_status': quality_status,
                    'defects_detected': [],
                    'bbox': results['boxes'][best_idx].tolist(),
                    'inference_time': inference_time,
                    'probabilities': self._calculate_class_probabilities_detr(results)
                }
            """
            
            return self._generate_demo_inspection("DETR")
            
        except Exception as e:
            print(f"❌ Error en DETR: {e}")
            return {'error': str(e)}
    
    # ==================== EVALUACIÓN DE CALIDAD ====================
    def _evaluate_quality(self, confidence, defects=None):
        """
        Evalúa el estado de calidad del producto
        
        Args:
            confidence: Nivel de confianza de la predicción
            defects: Lista de defectos detectados
        
        Returns:
            str: Estado de calidad (APROBADO/RECHAZADO/REVISION)
        """
        # Si hay defectos críticos, rechazar
        if defects and any(d in ['Roto', 'Mal ensamblado', 'Faltante de componente'] for d in defects):
            return 'RECHAZADO'
        
        # Evaluar por confianza
        if confidence >= CONFIDENCE_THRESHOLDS['APROBADO']:
            return 'APROBADO' if not defects or defects == ['Sin defectos'] else 'REVISION'
        elif confidence >= CONFIDENCE_THRESHOLDS['REVISION']:
            return 'REVISION'
        else:
            return 'RECHAZADO'
    
    # ==================== RESULTADO DE DEMOSTRACIÓN ====================
    def _generate_demo_inspection(self, model_name):
        """
        Genera inspección de demostración
        Esto se usa mientras no tengas los modelos reales
        """
        self.inspection_count += 1
        
        # Seleccionar producto aleatorio
        product_idx = np.random.randint(0, len(PRODUCT_CLASSES))
        product = PRODUCT_CLASSES[product_idx]
        
        # Generar confianza aleatoria
        confidence = np.random.uniform(75, 98)
        
        # Decidir si hay defectos (30% de probabilidad)
        has_defect = np.random.random() < 0.3
        if has_defect:
            defects = [np.random.choice(DEFECT_TYPES[:-1])]  # Excluir "Sin defectos"
        else:
            defects = ['Sin defectos']
        
        # Evaluar calidad
        quality_status = self._evaluate_quality(confidence, defects)
        
        # Generar probabilidades para todas las clases
        probabilities = np.random.rand(len(PRODUCT_CLASSES))
        probabilities[product_idx] = confidence / 100
        probabilities = probabilities / probabilities.sum() * 100
        
        # Tiempos de inferencia realistas
        inference_times = {
            'YOLO11': np.random.uniform(0.015, 0.035),
            'EfficientNetV2': np.random.uniform(0.050, 0.120),
            'DETR': np.random.uniform(0.100, 0.250)
        }
        
        return {
            'product': product,
            'confidence': round(confidence, 2),
            'quality_status': quality_status,
            'defects_detected': defects,
            'inference_time': round(inference_times.get(model_name, 0.1), 3),
            'probabilities': {
                PRODUCT_CLASSES[i]: round(float(probabilities[i]), 2)
                for i in range(len(PRODUCT_CLASSES))
            },
            'inspection_id': f"INS-{self.inspection_count:05d}",
            'demo_mode': True
        }

# ============================================
# INICIALIZAR SISTEMA
# ============================================

qc_system = QualityControlSystem()

# ============================================
# FUNCIONES AUXILIARES
# ============================================

def process_image(image_file):
    """Procesa y valida la imagen"""
    try:
        image = Image.open(io.BytesIO(image_file.read()))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Error al procesar imagen: {e}")

def validate_image_file(file):
    """Valida el archivo de imagen"""
    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
    
    if not file or file.filename == '':
        return False, "No se seleccionó ningún archivo"
    
    ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    
    if ext not in allowed_extensions:
        return False, f"Formato no válido. Use: {', '.join(allowed_extensions)}"
    
    return True, "OK"

# ============================================
# RUTAS DEL API
# ============================================

@app.route('/', methods=['GET'])
def index():
    """Página de inicio del API"""
    return jsonify({
        'system': 'Sistema de Control de Calidad - Productos de Cómputo',
        'version': '1.0.0',
        'status': 'Operativo',
        'arquitecturas': ['YOLO11', 'EfficientNetV2', 'DETR'],
        'productos_soportados': len(PRODUCT_CLASSES),
        'endpoints': {
            '/': 'Información del sistema',
            '/health': 'Estado del servidor',
            '/products': 'Lista de productos soportados',
            '/inspect': 'Inspección de calidad (POST)',
            '/stats': 'Estadísticas del sistema'
        },
        'documentation': 'https://github.com/tu-proyecto/docs'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Verifica el estado del sistema"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'uptime': 'N/A',
        'models_status': {
            'yolo11': {
                'loaded': qc_system.yolo_model is not None,
                'status': '🟢 Operativo' if qc_system.yolo_model else '🟡 Demo'
            },
            'efficientnetv2': {
                'loaded': qc_system.efficient_model is not None,
                'status': '🟢 Operativo' if qc_system.efficient_model else '🟡 Demo'
            },
            'detr': {
                'loaded': qc_system.detr_model is not None,
                'status': '🟢 Operativo' if qc_system.detr_model else '🟡 Demo'
            }
        },
        'inspections_completed': qc_system.inspection_count
    })

@app.route('/products', methods=['GET'])
def get_products():
    """Retorna información sobre productos soportados"""
    return jsonify({
        'total_products': len(PRODUCT_CLASSES),
        'products': PRODUCT_CLASSES,
        'defect_types': DEFECT_TYPES,
        'quality_statuses': list(QUALITY_STATUS.keys()),
        'confidence_thresholds': CONFIDENCE_THRESHOLDS
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    """Estadísticas del sistema"""
    return jsonify({
        'total_inspections': qc_system.inspection_count,
        'products_supported': len(PRODUCT_CLASSES),
        'architectures_available': 3,
        'avg_inference_time': {
            'yolo11': '~0.025s',
            'efficientnetv2': '~0.085s',
            'detr': '~0.175s'
        }
    })

@app.route('/inspect', methods=['POST'])
def quality_inspection():
    """
    Endpoint principal: Inspección de calidad con las 3 arquitecturas
    
    Esperado:
        - Archivo 'image' en form-data
    
    Retorna:
        JSON con resultados de inspección de las 3 arquitecturas
    """
    try:
        # Validar imagen
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No se encontró imagen en la petición'
            }), 400
        
        file = request.files['image']
        
        is_valid, message = validate_image_file(file)
        if not is_valid:
            return jsonify({'success': False, 'error': message}), 400
        
        # Procesar imagen
        image = process_image(file)
        
        print(f"\n{'='*60}")
        print(f"🔍 NUEVA INSPECCIÓN")
        print(f"📸 Imagen: {image.size[0]}x{image.size[1]} - {image.mode}")
        print(f"📅 Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        
        # Realizar inspección con las 3 arquitecturas
        start_total = time.time()
        
        print("⏳ Inspeccionando con YOLO11...")
        yolo_result = qc_system.inspect_with_yolo(image)
        
        print("⏳ Inspeccionando con EfficientNetV2...")
        efficient_result = qc_system.inspect_with_efficient(image)
        
        print("⏳ Inspeccionando con DETR...")
        detr_result = qc_system.inspect_with_detr(image)
        
        total_time = time.time() - start_total
        
        # Determinar veredicto final (consenso mayoritario)
        statuses = [
            yolo_result.get('quality_status'),
            efficient_result.get('quality_status'),
            detr_result.get('quality_status')
        ]
        final_verdict = max(set(statuses), key=statuses.count)
        
        print(f"✅ Inspección completada en {total_time:.3f}s")
        print(f"📋 Veredicto final: {final_verdict}")
        print(f"{'='*60}\n")
        
        # Construir respuesta
        response = {
            'success': True,
            'inspection_id': f"INS-{qc_system.inspection_count:05d}",
            'timestamp': datetime.now().isoformat(),
            'total_processing_time': round(total_time, 3),
            'final_verdict': final_verdict,
            'verdict_details': QUALITY_STATUS[final_verdict],
            'architectures': {
                'yolo11': yolo_result,
                'efficientnetv2': efficient_result,
                'detr': detr_result
            },
            'image_info': {
                'width': image.size[0],
                'height': image.size[1],
                'mode': image.mode,
                'format': file.filename.split('.')[-1].upper()
            },
            'recommendations': generate_recommendations(final_verdict, [
                yolo_result, efficient_result, detr_result
            ])
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"❌ ERROR EN INSPECCIÓN: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc() if app.debug else None
        }), 500

def generate_recommendations(verdict, results):
    """Genera recomendaciones basadas en la inspección"""
    recommendations = []
    
    if verdict == 'APROBADO':
        recommendations.append("✅ Producto aprobado para envío")
        recommendations.append("📦 Puede proceder al empaquetado")
    elif verdict == 'REVISION':
        recommendations.append("⚠️  Producto requiere inspección manual")
        recommendations.append("🔍 Verificar defectos menores detectados")
    else:  # RECHAZADO
        recommendations.append("❌ Producto rechazado - No apto para venta")
        recommendations.append("♻️  Enviar a reprocesamiento o descarte")
    
    # Agregar defectos comunes si existen
    all_defects = []
    for result in results:
        if 'defects_detected' in result:
            all_defects.extend(result['defects_detected'])
    
    unique_defects = list(set(all_defects))
    if unique_defects and unique_defects != ['Sin defectos']:
        recommendations.append(f"🔧 Defectos: {', '.join(unique_defects)}")
    
    return recommendations

# ============================================
# MANEJO DE ERRORES
# ============================================

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        'success': False,
        'error': 'Imagen demasiado grande (máximo 16MB)'
    }), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint no encontrado'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Error interno del servidor'
    }), 500

# ============================================
# EJECUTAR SERVIDOR
# ============================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🏭 SISTEMA DE CONTROL DE CALIDAD - PRODUCTOS DE CÓMPUTO")
    print("="*70)
    print(f"🚀 Servidor iniciado en: http://localhost:5000")
    print(f"📊 Productos soportados: {len(PRODUCT_CLASSES)}")
    print(f"🏗️  Arquitecturas: YOLO11, EfficientNetV2, DETR")
    print(f"🔍 Defectos detectables: {len(DEFECT_TYPES)}")
    print(f"⚙️  Modo: {'DEMOSTRACIÓN' if qc_system.yolo_model is None else 'PRODUCCIÓN'}")
    print("="*70)
    print("📋 Endpoints disponibles:")
    print("   • GET  /          - Información del sistema")
    print("   • GET  /health    - Estado del servidor")
    print("   • GET  /products  - Productos soportados")
    print("   • POST /inspect   - Inspección de calidad")
    print("   • GET  /stats     - Estadísticas")
    print("="*70 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True  # Cambiar a False en producción
    )