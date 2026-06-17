"""
API predictiva de churn con monitoreo básico y mejoras técnicas.

Esta versión incorpora:
1. Logging a archivo y consola.
2. Medición de latencia mediante middleware.
3. Conteo acumulado de solicitudes, errores y predicciones.
4. Detección de valores fuera del rango histórico.
5. Endpoint GET /metrics para consultar un resumen acumulado.
6. Nivel de riesgo granular (Muy Alto, Alto, Medio, Bajo).
7. Descripción personalizada por perfil de cliente.
8. Recomendaciones comerciales por nivel de riesgo.
9. Metadatos: versión, autor y fecha de entrenamiento.

Importante:
- Las métricas se almacenan temporalmente en memoria.
- Si la API se reinicia, los contadores vuelven a cero.
- Esta solución tiene fines académicos y no reemplaza una plataforma
  empresarial de monitoreo.
"""

# ============================================================
# BLOQUE 1. IMPORTACIÓN DE LIBRERÍAS
# ============================================================

from collections import Counter
from pathlib import Path
from threading import Lock
from time import perf_counter
import logging
import json
from datetime import datetime

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field

# ============================================================
# BLOQUE 2. CONFIGURACIÓN GENERAL DEL PROYECTO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Rutas de modelos y metadatos
MODEL_PATH = PROJECT_ROOT / "models" / "modelo_churn_v1.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "modelo_churn_v1_metadata.json"

# Carpeta de logs
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOGS_DIR / "monitor_api.log"

# Información que se mostrará en las respuestas de la API
VERSION_MODELO = "modelo_churn_v1"
AUTOR = "MARÍA DEL ROSARIO NINA YUCRA"  # CAMBIAR POR TU NOMBRE COMPLETO
FECHA_ENTRENAMIENTO = "13-06-2026"

# ============================================================
# BLOQUE 3. CARGA DE METADATOS DEL MODELO
# ============================================================

def cargar_metadatos() -> dict:
    """Carga los metadatos del modelo desde el archivo JSON."""
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "version": VERSION_MODELO,
        "autor": AUTOR,
        "fecha_entrenamiento": FECHA_ENTRENAMIENTO,
        "variables": ["antiguedad", "cargo_mensual", "reclamos"],
        "descripcion": "Modelo de predicción de churn (abandono de clientes)"
    }

METADATOS_MODELO = cargar_metadatos()

# ============================================================
# BLOQUE 4. RANGOS HISTÓRICOS DE REFERENCIA
# ============================================================

RANGOS_HISTORICOS = {
    "antiguedad": (1, 72),
    "cargo_mensual": (20.0, 150.0),
    "reclamos": (0, 7),
}

# ============================================================
# BLOQUE 5. LOGGING A ARCHIVO Y CONSOLA
# ============================================================

LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("api_churn")

# ============================================================
# BLOQUE 6. VERIFICACIÓN Y CARGA DEL MODELO
# ============================================================

if not MODEL_PATH.exists():
    raise RuntimeError(
        "No se encontró el modelo serializado. "
        "Ejecute primero: python src\\entrenar_modelo.py"
    )

modelo = joblib.load(MODEL_PATH)
logger.info("Modelo cargado correctamente: %s", VERSION_MODELO)
logger.info("Metadatos: %s", METADATOS_MODELO)

# ============================================================
# BLOQUE 7. CONTADORES DE MÉTRICAS EN MEMORIA
# ============================================================

metricas = {
    "solicitudes_totales": 0,
    "errores_validacion": 0,
    "errores_internos": 0,
    "predicciones_validas": 0,
    "predicciones_alto_riesgo": 0,
    "predicciones_bajo_riesgo": 0,
    "solicitudes_con_anomalias": 0,
    "latencia_acumulada_ms": 0.0,
    "latencia_maxima_ms": 0.0,
    "codigos_http": Counter(),
}

metricas_lock = Lock()

# ============================================================
# BLOQUE 8. MODELOS DE DATOS Y VALIDACIÓN DE ENTRADAS
# ============================================================

class ClienteEntrada(BaseModel):
    """Define los datos requeridos por POST /predict."""
    
    antiguedad: int = Field(
        ...,
        ge=0,
        le=240,
        description="Antigüedad del cliente expresada en meses",
        examples=[12],
    )
    cargo_mensual: float = Field(
        ...,
        ge=0,
        le=5000,
        description="Cargo mensual del cliente",
        examples=[95.5],
    )
    reclamos: int = Field(
        ...,
        ge=0,
        le=100,
        description="Cantidad de reclamos recientes",
        examples=[3],
    )

class PrediccionSalida(BaseModel):
    """Define la estructura de respuesta de POST /predict."""
    
    prediccion: str
    probabilidad: float
    version_modelo: str
    autor: str
    fecha_entrenamiento: str
    nivel_riesgo: str
    descripcion: str
    recomendacion: str
    alertas_datos: list[str]

# ============================================================
# BLOQUE 9. DETECCIÓN DE VALORES FUERA DEL RANGO HISTÓRICO
# ============================================================

def detectar_anomalias(datos: ClienteEntrada) -> list[str]:
    """
    Identifica valores técnicamente permitidos por la API, pero atípicos
    frente al histórico utilizado durante el entrenamiento.
    """
    alertas: list[str] = []
    valores = datos.model_dump()

    for variable, valor in valores.items():
        minimo, maximo = RANGOS_HISTORICOS[variable]
        if valor < minimo or valor > maximo:
            alertas.append(
                f"{variable}={valor} fuera del rango histórico "
                f"[{minimo}, {maximo}]"
            )
    return alertas

# ============================================================
# BLOQUE 10. FUNCIONES DE DESCRIPCIÓN Y RECOMENDACIÓN
# ============================================================

def generar_descripcion(datos: ClienteEntrada, probabilidad: float) -> str:
    """Genera una descripción personalizada basada en el perfil del cliente."""
    
    if datos.reclamos >= 3 and datos.cargo_mensual > 70:
        return (
            "Cliente con alta insatisfacción (múltiples reclamos) "
            "y costo elevado. Probabilidad significativa de abandono."
        )
    elif datos.antiguedad < 6 and datos.cargo_mensual > 80:
        return (
            "Cliente nuevo con tarifa alta. "
            "Riesgo temprano de churn por falta de fidelización."
        )
    elif datos.reclamos >= 2:
        return (
            "Cliente con reclamos recientes. "
            "Monitorear satisfacción para evitar deterioro de la relación."
        )
    elif datos.cargo_mensual > 90:
        return (
            "Cliente con cargo mensual muy elevado. "
            "Evaluar si el servicio justifica el costo."
        )
    elif datos.antiguedad > 24 and probabilidad < 0.3:
        return (
            "Cliente antiguo y estable. "
            "Bajo riesgo de abandono según el modelo."
        )
    else:
        return (
            "Cliente dentro de un perfil moderado. "
            "Riesgo estimado según comportamiento histórico."
        )

def generar_recomendacion(probabilidad: float) -> str:
    """Genera una recomendación comercial basada en el nivel de riesgo."""
    
    if probabilidad >= 0.75:
        return (
            "ACCIÓN URGENTE: Asignar a agente de retención, "
            "ofrecer descuento significativo (20-30%) y resolver reclamos pendientes."
        )
    elif probabilidad >= 0.50:
        return (
            "ACCIÓN PREVENTIVA: Enviar oferta personalizada, "
            "contactar para encuesta de satisfacción y ofrecer beneficios adicionales."
        )
    elif probabilidad >= 0.25:
        return (
            "MONITOREO: Evaluar comportamiento próximo, "
            "enviar newsletter con promociones y verificar uso del servicio."
        )
    else:
        return (
            "CLIENTE LEAL: Mantener beneficios actuales, "
            "enviar programa de referidos y fidelizar con pequeño reconocimiento."
        )

def obtener_nivel_riesgo(probabilidad: float) -> str:
    """Determina el nivel de riesgo granular."""
    
    if probabilidad >= 0.75:
        return "Muy Alto"
    elif probabilidad >= 0.50:
        return "Alto"
    elif probabilidad >= 0.25:
        return "Medio"
    else:
        return "Bajo"

# ============================================================
# BLOQUE 11. PREPARACIÓN DEL RESUMEN DE MÉTRICAS
# ============================================================

def resumen_metricas() -> dict:
    """Devuelve una copia segura y legible de las métricas acumuladas."""
    
    with metricas_lock:
        total = metricas["solicitudes_totales"]
        latencia_promedio = (
            metricas["latencia_acumulada_ms"] / total
            if total
            else 0.0
        )

        return {
            "version_modelo": VERSION_MODELO,
            "autor": AUTOR,
            "fecha_entrenamiento": FECHA_ENTRENAMIENTO,
            "solicitudes_totales": total,
            "errores_validacion": metricas["errores_validacion"],
            "errores_internos": metricas["errores_internos"],
            "predicciones_validas": metricas["predicciones_validas"],
            "predicciones_alto_riesgo": metricas["predicciones_alto_riesgo"],
            "predicciones_bajo_riesgo": metricas["predicciones_bajo_riesgo"],
            "solicitudes_con_anomalias": metricas["solicitudes_con_anomalias"],
            "latencia_promedio_ms": round(latencia_promedio, 3),
            "latencia_maxima_ms": round(metricas["latencia_maxima_ms"], 3),
            "codigos_http": dict(metricas["codigos_http"]),
        }

# ============================================================
# BLOQUE 12. CREACIÓN DE LA APLICACIÓN FASTAPI
# ============================================================

app = FastAPI(
    title="API de predicción de churn con monitoreo básico",
    description="Servicio académico ML-Ops con métricas, logs y mejoras técnicas.",
    version="2.0.0",
)

# ============================================================
# BLOQUE 13. MIDDLEWARE PARA MEDIR LATENCIA Y CONTAR SOLICITUDES
# ============================================================

@app.middleware("http")
async def registrar_solicitud(request: Request, call_next):
    """Observa todas las solicitudes HTTP procesadas por la API."""
    
    inicio = perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        with metricas_lock:
            metricas["errores_internos"] += 1
        logger.exception("Error interno no controlado en %s", request.url.path)
        raise

    latencia_ms = (perf_counter() - inicio) * 1000

    with metricas_lock:
        metricas["solicitudes_totales"] += 1
        metricas["latencia_acumulada_ms"] += latencia_ms
        metricas["latencia_maxima_ms"] = max(
            metricas["latencia_maxima_ms"],
            latencia_ms,
        )
        metricas["codigos_http"][str(response.status_code)] += 1

    logger.info(
        "Solicitud | metodo=%s | ruta=%s | estado=%s | latencia_ms=%.3f",
        request.method,
        request.url.path,
        response.status_code,
        latencia_ms,
    )

    response.headers["X-Process-Time-ms"] = f"{latencia_ms:.3f}"
    return response

# ============================================================
# BLOQUE 14. MANEJO DE ERRORES DE VALIDACIÓN
# ============================================================

@app.exception_handler(RequestValidationError)
async def registrar_error_validacion(
    request: Request,
    exc: RequestValidationError,
):
    """Incrementa el contador cuando FastAPI rechaza los datos de entrada."""
    
    with metricas_lock:
        metricas["errores_validacion"] += 1

    logger.warning(
        "Error de validación | ruta=%s | detalle=%s",
        request.url.path,
        exc.errors(),
    )

    return await request_validation_exception_handler(request, exc)

# ============================================================
# BLOQUE 15. ENDPOINT DE INICIO
# ============================================================

@app.get("/")
def inicio() -> dict[str, str]:
    """Confirma que el servicio está activo."""
    
    return {
        "mensaje": "Servicio ML-Ops activo",
        "estado": "ok",
        "autor": "MARÍA DEL ROSARIO NINA YUCRA",
        "modelo": "V1 cargado correctamente",
        "fecha_entrenamiento": "16-06-2026",
        "monitoreo": "activo"
    }

# ============================================================
# BLOQUE 16. ENDPOINT DE SALUD
# ============================================================

@app.get("/health")
def health() -> dict[str, str]:
    """Confirma que la API funciona y que el monitoreo está activo."""
    
    return {
        "estado": "ok",
        "modelo": "V1 cargado correctamente",
        "autor": "MARÍA DEL ROSARIO NINA YUCRA",
        "fecha_entrenamiento": "16-06-2026",
        "monitoreo": "activo",
    }

# ============================================================
# BLOQUE 17. ENDPOINT GET /metrics
# ============================================================

@app.get("/metrics")
def metrics() -> dict:
    """Devuelve las métricas acumuladas desde que se inició la API."""
    
    return resumen_metricas()

# ============================================================
# BLOQUE 18. ENDPOINT POST /predict
# ============================================================

@app.post("/predict", response_model=PrediccionSalida)
def predict(datos: ClienteEntrada) -> PrediccionSalida:
    """
    Recibe los datos del cliente y genera una predicción de churn.
    
    Flujo:
    1. Detectar valores fuera del rango histórico.
    2. Construir la entrada esperada por el modelo.
    3. Calcular la probabilidad.
    4. Asignar alto_riesgo o bajo_riesgo.
    5. Generar nivel de riesgo, descripción y recomendación.
    6. Actualizar métricas.
    7. Registrar eventos.
    8. Devolver la respuesta.
    """
    
    try:
        # Paso 1. Detectar datos atípicos.
        alertas = detectar_anomalias(datos)

        # Paso 2. Preparar los datos.
        X = [[
            datos.antiguedad,
            datos.cargo_mensual,
            datos.reclamos,
        ]]

        # Paso 3. Calcular la probabilidad.
        probabilidad = float(modelo.predict_proba(X)[0][1])

        # Paso 4. Asignar etiqueta con umbral del 50%.
        etiqueta = "alto_riesgo" if probabilidad >= 0.50 else "bajo_riesgo"

        # Paso 5. Generar mejoras técnicas.
        nivel_riesgo = obtener_nivel_riesgo(probabilidad)
        descripcion = generar_descripcion(datos, probabilidad)
        recomendacion = generar_recomendacion(probabilidad)

        # Paso 6. Actualizar métricas.
        with metricas_lock:
            metricas["predicciones_validas"] += 1
            metricas[f"predicciones_{etiqueta}"] += 1
            if alertas:
                metricas["solicitudes_con_anomalias"] += 1

        # Paso 7. Registrar eventos.
        if alertas:
            logger.warning(
                "Valores fuera de rango histórico: %s",
                alertas,
            )

        logger.info(
            "Predicción | resultado=%s | probabilidad=%.4f | "
            "nivel_riesgo=%s | alertas=%s",
            etiqueta,
            probabilidad,
            nivel_riesgo,
            len(alertas),
        )

        # Paso 8. Devolver respuesta.
        return PrediccionSalida(
            prediccion=etiqueta,
            probabilidad=round(probabilidad, 4),
            version_modelo=VERSION_MODELO,
            autor=AUTOR,
            fecha_entrenamiento=FECHA_ENTRENAMIENTO,
            nivel_riesgo=nivel_riesgo,
            descripcion=descripcion,
            recomendacion=recomendacion,
            alertas_datos=alertas,
        )

    except Exception as exc:
        with metricas_lock:
            metricas["errores_internos"] += 1
        logger.exception("No fue posible generar la predicción")
        raise HTTPException(
            status_code=500,
            detail="No fue posible generar la predicción.",
        ) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
