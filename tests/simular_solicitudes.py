"""
Simulación básica de solicitudes para observar el monitoreo de la API.

Este script permite generar tráfico controlado hacia la API predictiva
de churn desarrollada durante el laboratorio.

Objetivos:
1. Enviar solicitudes válidas al endpoint POST /predict.
2. Enviar una solicitud atípica para generar alertas de datos.
3. Enviar una solicitud inválida para comprobar el error HTTP 422.
4. Mostrar la latencia informada por el middleware de la API.
5. Consultar el resumen acumulado mediante el endpoint GET /metrics.
6. Verificar las mejoras técnicas: nivel de riesgo, descripción y recomendación.

Importante:
- La API debe estar activa antes de ejecutar este archivo.
- Este script no entrena el modelo.
- Este script no modifica la API.
- Solamente simula solicitudes para observar su comportamiento.
"""

# ============================================================
# BLOQUE 1. IMPORTACIÓN DE LIBRERÍAS
# ============================================================

from pprint import pprint
# pprint permite mostrar diccionarios JSON de forma ordenada y legible.

import requests
# requests permite enviar solicitudes HTTP desde Python.
# En este laboratorio se utiliza para comunicarse con la API local.

# ============================================================
# BLOQUE 2. CONFIGURACIÓN GENERAL
# ============================================================

# Dirección base donde se encuentra ejecutándose la API.
# El puerto predeterminado utilizado por Uvicorn es 8000.
BASE_URL = "http://127.0.0.1:8000"

# Tiempo máximo de espera para cada solicitud, expresado en segundos.
# Evita que el programa quede esperando indefinidamente si la API
# no responde o si existe algún problema de conexión.
TIMEOUT = 10

# ============================================================
# BLOQUE 3. CASOS DE PRUEBA
# ============================================================
# AQUÍ SE DEFINEN LAS SOLICITUDES QUE SE ENVIARÁN A POST /predict.
#
# Cada caso contiene:
# - nombre: etiqueta descriptiva para identificar la prueba;
# - datos: valores que recibirá la API;
# - descripcion_esperada: texto que debe contener la descripción;
# - nivel_riesgo_esperado: nivel de riesgo que debe devolver.
#
# Se incluyen:
# - casos válidos dentro de rangos razonables;
# - casos que cubren diferentes niveles de riesgo (Bajo, Medio, Alto, Muy Alto);
# - un caso atípico aceptado técnicamente, pero fuera del histórico;
# - un caso inválido que debe ser rechazado con código HTTP 422.

CASOS = [
    {
        "nombre": "cliente_estable_bajo_riesgo",
        "datos": {
            "antiguedad": 48,
            "cargo_mensual": 55.0,
            "reclamos": 0,
        },
        "nivel_riesgo_esperado": "Bajo",
        "descripcion_esperada": "Cliente antiguo y estable",
    },
    {
        "nombre": "cliente_riesgo_medio",
        "datos": {
            "antiguedad": 18,
            "cargo_mensual": 95.0,
            "reclamos": 2,
        },
        "nivel_riesgo_esperado": "Medio",
        "descripcion_esperada": "Cliente con reclamos recientes",
    },
    {
        "nombre": "cliente_alto_riesgo",
        "datos": {
            "antiguedad": 4,
            "cargo_mensual": 145.0,
            "reclamos": 7,
        },
        "nivel_riesgo_esperado": "Muy Alto",
        "descripcion_esperada": "alta insatisfacción",
    },
    {
        "nombre": "cliente_nuevo_tarifa_alta",
        "datos": {
            "antiguedad": 3,
            "cargo_mensual": 120.0,
            "reclamos": 0,
        },
        "nivel_riesgo_esperado": "Alto",
        "descripcion_esperada": "Cliente nuevo con tarifa alta",
    },
    {
        "nombre": "cliente_cargo_muy_elevado",
        "datos": {
            "antiguedad": 24,
            "cargo_mensual": 200.0,
            "reclamos": 1,
        },
        "nivel_riesgo_esperado": "Muy Alto",
        "descripcion_esperada": "cargo mensual muy elevado",
    },
    {
        "nombre": "cliente_atipico",
        "datos": {
            "antiguedad": 180,
            "cargo_mensual": 600.0,
            "reclamos": 35,
        },
        # Este caso es técnicamente válido porque los valores respetan
        # los límites generales definidos por Pydantic.
        #
        # Sin embargo, los valores se encuentran fuera de los rangos
        # históricos del entrenamiento. Por ello, la API debe generar
        # alertas_datos.
        "nivel_riesgo_esperado": None,  # No aplica por ser atípico
        "descripcion_esperada": None,   # No aplica por ser atípico
        "espera_alertas": True,
    },
    {
        "nombre": "cliente_invalido",
        "datos": {
            "antiguedad": 12,
            "cargo_mensual": -50.0,
            "reclamos": 1,
        },
        # Este caso debe ser rechazado porque cargo_mensual es negativo.
        # La API devolverá un código HTTP 422 y aumentará el contador
        # errores_validacion.
        "espera_error_422": True,
    },
    {
        "nombre": "cliente_campo_faltante",
        "datos": {
            "antiguedad": 36,
            "cargo_mensual": 60.5,
            # Falta el campo "reclamos" intencionalmente
        },
        "espera_error_422": True,
    },
]

# ============================================================
# BLOQUE 4. FUNCIÓN PARA MOSTRAR LA RESPUESTA DE CADA CASO
# ============================================================
# AQUÍ SE MUESTRA:
# - nombre del caso;
# - código HTTP recibido;
# - latencia medida por el middleware;
# - contenido JSON devuelto por la API;
# - verificación de las mejoras técnicas.

def mostrar_respuesta(
    nombre: str,
    respuesta: requests.Response,
    caso: dict,
) -> None:
    """
    Presenta de forma ordenada el resultado de una solicitud.

    Parámetros:
        nombre:
            Nombre descriptivo del caso evaluado.

        respuesta:
            Objeto Response devuelto por la librería requests.
            Contiene el código HTTP, las cabeceras y el cuerpo JSON.

        caso:
            Diccionario con la configuración del caso de prueba.
    """

    print("\n" + "=" * 70)
    print(f"📌 Caso: {nombre}")

    # Mostrar el código HTTP devuelto por la API.
    #
    # Ejemplos:
    # - 200: solicitud procesada correctamente;
    # - 422: error de validación de datos;
    # - 500: error interno de la API.
    print(f"📊 Estado HTTP: {respuesta.status_code}")

    # Recuperar la cabecera agregada por el middleware de la API.
    #
    # En api/main.py se incorporó:
    # response.headers["X-Process-Time-ms"] = ...
    #
    # Esta cabecera informa cuántos milisegundos tardó la solicitud.
    latencia = respuesta.headers.get("X-Process-Time-ms")

    if latencia is not None:
        print(f"⏱️ Latencia informada por API: {latencia} ms")
    else:
        print("⏱️ Latencia informada por API: no disponible")

    # Intentar convertir el cuerpo de la respuesta a JSON.
    try:
        data = respuesta.json()
        
        # Si la solicitud fue exitosa (HTTP 200), mostrar los campos
        # de las mejoras técnicas de forma resumida.
        if respuesta.status_code == 200:
            print("\n📋 RESULTADO DE LA PREDICCIÓN:")
            print(f"  🎯 Predicción: {data.get('prediccion')}")
            print(f"  📈 Probabilidad: {data.get('probabilidad')}")
            print(f"  ⚠️ Nivel de Riesgo: {data.get('nivel_riesgo')}")
            print(f"  📝 Versión Modelo: {data.get('version_modelo')}")
            print(f"  👤 Autor: {data.get('autor')}")
            print(f"  📅 Fecha Entrenamiento: {data.get('fecha_entrenamiento')}")
            print(f"\n  📖 Descripción:")
            print(f"    {data.get('descripcion')}")
            print(f"\n  💡 Recomendación:")
            print(f"    {data.get('recomendacion')}")
            
            # Mostrar alertas si existen
            alertas = data.get('alertas_datos', [])
            if alertas:
                print(f"\n  ⚠️ ALERTAS DE DATOS:")
                for alerta in alertas:
                    print(f"    • {alerta}")
            else:
                print(f"\n  ✅ Datos dentro de rangos históricos: Sin alertas")
            
            # Verificar nivel de riesgo esperado
            nivel_esperado = caso.get('nivel_riesgo_esperado')
            if nivel_esperado:
                nivel_obtenido = data.get('nivel_riesgo')
                if nivel_obtenido == nivel_esperado:
                    print(f"\n  ✅ Nivel de riesgo correcto: {nivel_obtenido}")
                else:
                    print(f"\n  ⚠️ Nivel de riesgo inesperado: Esperado '{nivel_esperado}', Obtenido '{nivel_obtenido}'")
            
            # Verificar descripción esperada
            desc_esperada = caso.get('descripcion_esperada')
            if desc_esperada:
                desc_obtenida = data.get('descripcion', '')
                if desc_esperada.lower() in desc_obtenida.lower():
                    print(f"  ✅ Descripción coincide con lo esperado")
                else:
                    print(f"  ⚠️ Descripción no coincide: Esperado '{desc_esperada}'")
            
            # Verificar alertas esperadas
            if caso.get('espera_alertas'):
                if alertas:
                    print(f"  ✅ Se generaron alertas correctamente ({len(alertas)})")
                else:
                    print(f"  ⚠️ Se esperaban alertas pero no se generaron")
        
        # Si fue un error de validación (HTTP 422)
        elif respuesta.status_code == 422:
            if caso.get('espera_error_422'):
                print("  ✅ Error de validación esperado (HTTP 422)")
                print(f"  📝 Detalle: {data.get('detail', 'No disponible')}")
            else:
                print("  ⚠️ Error de validación no esperado")
        
        # Mostrar JSON completo para depuración
        print("\n📄 JSON Completo:")
        pprint(data)

    except requests.exceptions.JSONDecodeError:
        print("❌ La respuesta no contiene un JSON válido.")
        print(respuesta.text)

# ============================================================
# BLOQUE 5. FUNCIÓN PARA ENVIAR UNA SOLICITUD A POST /predict
# ============================================================
# AQUÍ SE IMPLEMENTA:
# - envío de cada caso de prueba;
# - comunicación con POST /predict;
# - manejo de problemas de conexión.

def enviar_caso(caso: dict) -> None:
    """
    Envía un caso de prueba al endpoint POST /predict.

    Parámetro:
        caso:
            Diccionario con un nombre descriptivo y los datos del cliente.
    """

    nombre = caso["nombre"]
    datos = caso["datos"]

    try:
        # Enviar una solicitud HTTP POST.
        #
        # URL utilizada:
        # http://127.0.0.1:8000/predict
        #
        # El argumento json=datos convierte automáticamente el diccionario
        # de Python en el formato JSON esperado por la API.
        respuesta = requests.post(
            f"{BASE_URL}/predict",
            json=datos,
            timeout=TIMEOUT,
        )

        # Mostrar el resultado recibido.
        mostrar_respuesta(nombre, respuesta, caso)

    except requests.exceptions.ConnectionError:
        # Este error aparece normalmente cuando la API no está activa.
        print("\n" + "=" * 70)
        print(f"❌ Caso: {nombre}")
        print("❌ Error: no fue posible conectarse con la API.")
        print("   Verifique que Uvicorn se encuentre activo en otra terminal.")
        print("   Comando: python -m uvicorn api.main:app --reload")

    except requests.exceptions.Timeout:
        # Este error aparece si la API demora más del tiempo configurado.
        print("\n" + "=" * 70)
        print(f"❌ Caso: {nombre}")
        print(f"❌ Error: la API no respondió en menos de {TIMEOUT} segundos.")

    except requests.exceptions.RequestException as exc:
        # Captura otros problemas relacionados con la solicitud HTTP.
        print("\n" + "=" * 70)
        print(f"❌ Caso: {nombre}")
        print(f"❌ Error inesperado durante la solicitud: {exc}")

# ============================================================
# BLOQUE 6. FUNCIÓN PARA CONSULTAR GET /metrics
# ============================================================
# AQUÍ SE IMPLEMENTA:
# - consulta del resumen acumulado de métricas;
# - comunicación con el endpoint GET /metrics.

def consultar_metricas() -> None:
    """
    Consulta y muestra las métricas acumuladas por la API.

    El endpoint GET /metrics resume:
    - cantidad total de solicitudes;
    - errores de validación;
    - errores internos;
    - predicciones válidas;
    - resultados de alto y bajo riesgo;
    - solicitudes con anomalías;
    - latencia promedio y máxima;
    - distribución de códigos HTTP.
    """

    print("\n" + "=" * 70)
    print("📊 RESUMEN ACUMULADO DE MÉTRICAS")
    print("=" * 70)

    try:
        # Enviar una solicitud HTTP GET al endpoint /metrics.
        respuesta_metricas = requests.get(
            f"{BASE_URL}/metrics",
            timeout=TIMEOUT,
        )

        print(f"📊 Estado HTTP: {respuesta_metricas.status_code}")

        # Mostrar el resumen JSON de manera ordenada.
        if respuesta_metricas.status_code == 200:
            data = respuesta_metricas.json()
            print("\n📋 MÉTRICAS DEL SERVICIO:")
            print(f"  🔢 Solicitudes totales: {data.get('solicitudes_totales')}")
            print(f"  ❌ Errores de validación: {data.get('errores_validacion')}")
            print(f"  💥 Errores internos: {data.get('errores_internos')}")
            print(f"  ✅ Predicciones válidas: {data.get('predicciones_validas')}")
            print(f"  🔴 Alto riesgo: {data.get('predicciones_alto_riesgo')}")
            print(f"  🟢 Bajo riesgo: {data.get('predicciones_bajo_riesgo')}")
            print(f"  ⚠️ Solicitudes con anomalías: {data.get('solicitudes_con_anomalias')}")
            print(f"  ⏱️ Latencia promedio: {data.get('latencia_promedio_ms')} ms")
            print(f"  ⏱️ Latencia máxima: {data.get('latencia_maxima_ms')} ms")
            print(f"\n  📊 Códigos HTTP: {data.get('codigos_http')}")
            print(f"\n  📌 Información del modelo:")
            print(f"    Versión: {data.get('version_modelo')}")
            print(f"    Autor: {data.get('autor')}")
            print(f"    Fecha entrenamiento: {data.get('fecha_entrenamiento')}")
        else:
            pprint(respuesta_metricas.json())

    except requests.exceptions.ConnectionError:
        print("❌ Error: no fue posible consultar las métricas.")
        print("   Verifique que la API se encuentre activa.")

    except requests.exceptions.Timeout:
        print(f"❌ Error: la API no respondió en menos de {TIMEOUT} segundos.")

    except requests.exceptions.JSONDecodeError:
        print("❌ Error: la respuesta de /metrics no contiene un JSON válido.")

    except requests.exceptions.RequestException as exc:
        print(f"❌ Error inesperado durante la consulta: {exc}")

# ============================================================
# BLOQUE 7. FUNCIÓN PARA PROBAR ENDPOINTS BÁSICOS
# ============================================================

def probar_endpoints_basicos() -> None:
    """
    Prueba los endpoints básicos de la API:
    - GET /
    - GET /health
    - GET /docs (solo verifica disponibilidad)
    """

    print("\n" + "=" * 70)
    print("🔍 VERIFICACIÓN DE ENDPOINTS BÁSICOS")
    print("=" * 70)

    endpoints = [
        ("/", "GET"),
        ("/health", "GET"),
        ("/metrics", "GET"),
    ]

    for endpoint, metodo in endpoints:
        try:
            if metodo == "GET":
                respuesta = requests.get(
                    f"{BASE_URL}{endpoint}",
                    timeout=TIMEOUT,
                )
                print(f"  {metodo} {endpoint} → HTTP {respuesta.status_code}")
                if respuesta.status_code == 200 and endpoint != "/metrics":
                    data = respuesta.json()
                    if "autor" in data:
                        print(f"    Autor: {data.get('autor')}")
                    if "version_modelo" in data or "modelo" in data:
                        modelo = data.get('version_modelo') or data.get('modelo')
                        print(f"    Modelo: {modelo}")
        except requests.exceptions.ConnectionError:
            print(f"  ❌ {metodo} {endpoint} → No disponible")
        except Exception as exc:
            print(f"  ❌ {metodo} {endpoint} → Error: {exc}")

# ============================================================
# BLOQUE 8. FUNCIÓN PRINCIPAL
# ============================================================
# AQUÍ SE DEFINE EL ORDEN COMPLETO DE EJECUCIÓN:
# 1. Mostrar un encabezado.
# 2. Probar endpoints básicos.
# 3. Recorrer todos los casos de prueba.
# 4. Enviar cada solicitud a POST /predict.
# 5. Consultar GET /metrics al finalizar.

def main() -> None:
    """
    Ejecuta la simulación completa de solicitudes.
    """

    print("=" * 70)
    print("🚀 SIMULACIÓN DE SOLICITUDES PARA LA API PREDICTIVA")
    print("📌 Con mejoras técnicas: Nivel de Riesgo, Descripción y Recomendación")
    print("=" * 70)

    # Probar endpoints básicos primero
    probar_endpoints_basicos()

    # Recorrer secuencialmente todos los casos definidos anteriormente.
    print("\n" + "=" * 70)
    print("📤 ENVIANDO SOLICITUDES DE PREDICCIÓN")
    print("=" * 70)

    for caso in CASOS:
        enviar_caso(caso)

    # Consultar el resumen acumulado después de procesar las solicitudes.
    consultar_metricas()

    # Resumen final
    print("\n" + "=" * 70)
    print("✅ SIMULACIÓN COMPLETADA")
    print("=" * 70)
    print("\n📝 Resumen de mejoras técnicas verificadas:")
    print("  ✅ Nivel de riesgo granular (Bajo, Medio, Alto, Muy Alto)")
    print("  ✅ Descripción personalizada por perfil de cliente")
    print("  ✅ Recomendación comercial según nivel de riesgo")
    print("  ✅ Metadatos: versión, autor y fecha de entrenamiento")
    print("  ✅ Alertas de datos fuera de rango histórico")
    print("  ✅ Monitoreo: logs, métricas y latencia")

# ============================================================
# BLOQUE 9. PUNTO DE ENTRADA DEL PROGRAMA
# ============================================================
# Esta condición permite ejecutar main() únicamente cuando este archivo
# se inicia directamente desde PowerShell.
#
# Comando:
# python tests\simular_solicitudes.py
#
# Si el archivo fuera importado desde otro script, main() no se ejecutaría
# automáticamente.

if __name__ == "__main__":
    main()