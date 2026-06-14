"""
API de predicción de churn con FastAPI.

La API carga un modelo serializado, valida los datos de entrada
y devuelve una predicción junto con su probabilidad,
incluyendo nivel de riesgo, descripción y recomendación.
"""

from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "modelo_churn_v1.joblib"

VERSION_MODELO = "modelo_churn_v1"
AUTOR = "MARÍA DEL ROSARIO NINA YUCRA"

if not MODEL_PATH.exists():
    raise RuntimeError(
        "No se encontró el modelo serializado. "
        "Ejecute primero: python src\\entrenar_modelo.py"
    )

modelo = joblib.load(MODEL_PATH)


class ClienteEntrada(BaseModel):
    antiguedad: int = Field(
        ...,
        ge=0,
        le=120,
        description="Antigüedad del cliente expresada en meses",
        examples=[12],
    )
    cargo_mensual: float = Field(
        ...,
        ge=0,
        le=1000,
        description="Cargo mensual del cliente",
        examples=[95.5],
    )
    reclamos: int = Field(
        ...,
        ge=0,
        le=50,
        description="Cantidad de reclamos recientes",
        examples=[3],
    )


class PrediccionSalida(BaseModel):
    prediccion: str
    probabilidad: float
    version_modelo: str
    autor: str
    nivel_riesgo: str
    descripcion: str
    recomendacion: str


app = FastAPI(
    title="API de predicción de churn",
    description="Servicio académico ML-Ops para estimar riesgo de abandono.",
    version="2.0.0",
)


@app.get("/")
def inicio() -> dict[str, str]:
    return {
        "mensaje": "Servicio ML-Ops activo",
        "estado": "ok",
        "autor": "MARÍA DEL ROSARIO NINA YUCRA",
        "modelo": "modelo_churn_v1",
        "fecha_entrenamiento": "13-06-2026"
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "estado": "ok",
        "modelo": VERSION_MODELO,
    }


@app.post("/predict", response_model=PrediccionSalida)
def predict(datos: ClienteEntrada) -> PrediccionSalida:
    try:
        # Preparar datos para predicción
        X = [[
            datos.antiguedad,
            datos.cargo_mensual,
            datos.reclamos,
        ]]

        # Obtener probabilidad de churn (clase 1)
        probabilidad = float(modelo.predict_proba(X)[0][1])
        etiqueta = "alto_riesgo" if probabilidad >= 0.50 else "bajo_riesgo"

        # Determinar nivel de riesgo granular
        if probabilidad >= 0.75:
            nivel_riesgo = "Muy Alto"
        elif probabilidad >= 0.50:
            nivel_riesgo = "Alto"
        elif probabilidad >= 0.25:
            nivel_riesgo = "Medio"
        else:
            nivel_riesgo = "Bajo"

        # Generar descripción basada en perfil del cliente
        if datos.reclamos >= 3 and datos.cargo_mensual > 70:
            descripcion = (
                "Cliente con alta insatisfacción (múltiples reclamos) "
                "y costo elevado. Probabilidad significativa de abandono."
            )
        elif datos.antiguedad < 6 and datos.cargo_mensual > 80:
            descripcion = (
                "Cliente nuevo con tarifa alta. "
                "Riesgo temprano de churn por falta de fidelización."
            )
        elif datos.reclamos >= 2:
            descripcion = (
                "Cliente con reclamos recientes. "
                "Monitorear satisfacción para evitar deterioro de la relación."
            )
        elif datos.cargo_mensual > 90:
            descripcion = (
                "Cliente con cargo mensual muy elevado. "
                "Evaluar si el servicio justifica el costo."
            )
        elif datos.antiguedad > 24 and probabilidad < 0.3:
            descripcion = (
                "Cliente antiguo y estable. "
                "Bajo riesgo de abandono según el modelo."
            )
        else:
            descripcion = (
                "Cliente dentro de un perfil moderado. "
                "Riesgo estimado según comportamiento histórico."
            )

        # Generar recomendación comercial
        if probabilidad >= 0.75:
            recomendacion = (
                "ACCIÓN URGENTE: Asignar a agente de retención, "
                "ofrecer descuento significativo (20-30%) y resolver reclamos pendientes."
            )
        elif probabilidad >= 0.50:
            recomendacion = (
                "ACCIÓN PREVENTIVA: Enviar oferta personalizada, "
                "contactar para encuesta de satisfacción y ofrecer beneficios adicionales."
            )
        elif probabilidad >= 0.25:
            recomendacion = (
                "MONITOREO: Evaluar comportamiento próximo, "
                "enviar newsletter con promociones y verificar uso del servicio."
            )
        else:
            recomendacion = (
                "CLIENTE LEAL: Mantener beneficios actuales, "
                "enviar programa de referidos y fidelizar con pequeño reconocimiento."
            )

        return PrediccionSalida(
            prediccion=etiqueta,
            probabilidad=round(probabilidad, 4),
            version_modelo=VERSION_MODELO,
            autor=AUTOR,
            nivel_riesgo=nivel_riesgo,
            descripcion=descripcion,
            recomendacion=recomendacion,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="No fue posible generar la predicción.",
        ) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
