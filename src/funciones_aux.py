"""Funciones auxiliares del taller de renta fija."""

from bisect import bisect_left
from math import exp
from typing import cast

import pandas as pd

TOLERANCIA_PRECIO = 1e-8


def interpolar_factor(
    plazo_buscado: float,
    plazos: list[float],
    factores: list[float],
) -> float:
    """Obtiene un factor de descuento mediante interpolación exponencial."""
    # Primero confirmamos que podemos interpolar, los puntos deberían encontrarse en la curva
    if not plazos:
        mensaje = "La curva no contiene puntos."
        raise ValueError(mensaje)

    if plazo_buscado < plazos[0] or plazo_buscado > plazos[-1]:
        mensaje = "El plazo buscado está fuera de la curva."
        raise ValueError(mensaje)

    # Buscamos los puntos que esten cerca del plazo
    # Con bisect_left obtneemos la posición de donde estaría el plazo en la lista ordenada
    posicion = bisect_left(plazos, plazo_buscado)

    # Si el plazo ya existe, utilizar directamente su factor.
    if plazos[posicion] == plazo_buscado:
        return factores[posicion]

    # Si no buscamos los plazos cercanos por ambos lados al punto
    plazo_anterior = plazos[posicion - 1]
    plazo_siguiente = plazos[posicion]

    factor_anterior = factores[posicion - 1]
    factor_siguiente = factores[posicion]

    # Calculamos los pesos de los puntos previos
    peso_siguiente = (plazo_buscado - plazo_anterior) / (plazo_siguiente - plazo_anterior)

    peso_anterior = 1 - peso_siguiente

    # La interpolación es lineal en los logaritmos de los factores.
    # Forma equivalente con exp y log:
    # exp(peso_anterior * log(factor_anterior) + peso_siguiente * log(factor_siguiente))
    # Aplicando exp(a * log(x)) = x**a se obtiene el producto de potencias.
    factor_interpolado = factor_anterior**peso_anterior * factor_siguiente**peso_siguiente

    return float(factor_interpolado)


def calcular_spread(
    flujos: pd.DataFrame | None,
    precio_mercado: float,
    precio_limpio: float,
    cupon_corrido: float,
) -> tuple[float, float]:
    """Devuelve el z-spread en pb y el precio limpio a partir de los flujos sin spread."""
    if flujos is None:
        return float("nan"), float("nan")

    plazos = flujos["plazo"]
    valores_cero = flujos["valor_actual"]
    valores = valores_cero
    z = 0.0

    # Usamos el método de Newton-Raphson para aproximar el z-spread que hace que el precio limpio coincida con el precio de mercado.
    # Eligimos este método porque nos facilita su aplicación partiendo de los resultados del apartado anterior,
    # y porque converge rápidamente en la mayoría de los casos.
    for _ in range(20):
        # Calculamos la diferencia entre el precio de mercado y el precio limpio calculado con el z-spread actual.
        diferencia = precio_mercado - precio_limpio
        # Fijamos un criterio de convergencia del método basado en el punto anterior y una tolerancia
        if abs(diferencia) < TOLERANCIA_PRECIO:
            return z * 10_000, precio_limpio

        # Calculamos la pendiente del precio respecto al spread, que es la derivada del precio limpio con respecto al z-spread.
        pendiente = float(-(plazos * valores).sum())
        if pendiente == 0:
            mensaje = "La pendiente del precio respecto al spread es cero."
            raise ValueError(mensaje)
        # Actualizamos el z-spread usando la fórmula de Newton-Raphson: z = z - f(z)/f'(z), donde f(z) es la diferencia entre el precio de mercado y el precio limpio.
        z += diferencia / pendiente
        valores = valores_cero * (-z * plazos).map(exp)
        precio_limpio = float(valores.sum()) - cupon_corrido

    mensaje = "Newton-Raphson no ha convergido para el z-spread."
    raise RuntimeError(mensaje)


def calcular_yield(
    flujos: pd.DataFrame | None,
    precio_sucio_mercado: float,
    frecuencia: int,
) -> float:
    """Calcula la yield anual en tanto por uno, con capitalización por cupón."""
    if flujos is None:
        return float("nan")
    # Obtenemos los periodos de pago y sus importes
    periodos = frecuencia * flujos["plazo"]
    importes = flujos["flujo_total"]

    # Calculamos el precio sucio partiendo de la yield anual
    def precio_sucio(yield_anual: float) -> float:
        return float((importes / (1 + yield_anual / frecuencia) ** periodos).sum())

    # Fijamos un límite inferior y superior para buscar la yield
    # Con esto buscaremos una yield que haga que nuestro precio sucio calculado coincida con el sucio del mercado
    inferior, superior = -0.50, 1.00
    if not precio_sucio(inferior) >= precio_sucio_mercado >= precio_sucio(superior):
        mensaje = "La yield está fuera del intervalo de búsqueda."
        raise ValueError(mensaje)

    # Aproximamos la yield mediante el método de la bisección
    for _ in range(50):
        yield_anual = (inferior + superior) / 2
        diferencia = precio_sucio(yield_anual) - precio_sucio_mercado
        if abs(diferencia) < TOLERANCIA_PRECIO:
            return yield_anual
        if diferencia > 0:
            inferior = yield_anual
        else:
            superior = yield_anual

    mensaje = "La bisección no ha convergido para la yield."
    raise RuntimeError(mensaje)


def calcular_duracion(
    flujos: pd.DataFrame | None,
    y: float,
    frecuencia: int,
) -> tuple[float, float]:
    """Devuelve la duración de Macaulay y la modificada en años."""
    duracion, duracion_modificada, _ = calcular_sensibilidad(flujos, y, frecuencia)
    return duracion, duracion_modificada


def calcular_sensibilidad(
    flujos: pd.DataFrame | None,
    y: float,
    frecuencia: int,
) -> tuple[float, float, float]:
    """Devuelve la duración deMacaulay, duración modificada y convexidad a la misma yield."""
    if flujos is None:
        return float("nan"), float("nan"), float("nan")

    plazos = flujos["plazo"]
    # Preparamos la derivada de Macaulay para la DurMod y la convexidad
    base = 1 + y / frecuencia
    # Calculamos el VA de cada flujo a la yield dada
    valores = flujos["flujo_total"] / base ** (frecuencia * plazos)
    precio_sucio = float(valores.sum())

    # Obtenemos la duración de Macaulay, la modificada, y la convexidad
    duracion = float((plazos * valores).sum()) / precio_sucio
    duracion_modificada = duracion / base
    convexidad = float((plazos * (plazos + 1 / frecuencia) * valores).sum()) / (
        precio_sucio * base**2
    )
    return duracion, duracion_modificada, convexidad


# Funcion base para calcular la valoración de un bono
def valoracion_bono(
    bono: pd.Series,
    curva_descuento: pd.DataFrame,
    fecha_valor: str | pd.Timestamp = "2026-10-01",
    nominal: float = 100,
    spread: float = 0,
) -> tuple[float, float, float, pd.DataFrame]:
    """Devuelve los precios limpio y sucio, el cupón corrido y los flujos ACT/365."""
    fecha = pd.Timestamp(fecha_valor)
    # Importes del cupón: nominal por tipo anual.
    primer_cupon = pd.to_datetime(bono["First Coupon Date"], dayfirst=True)
    vencimiento_cupon = pd.to_datetime(bono["Maturity"], dayfirst=True)
    # Si el bono es callable, suponemos que se rescata en el próximo call disponible.
    fecha_call = pd.NaT
    if str(bono.get("Callable")).strip().upper() == "Y":
        fecha_call = pd.to_datetime(bono.get("Next Call Date"), dayfirst=True, errors="coerce")
    # En los bonos perpetuos usamos el próximo call porque no tienen vencimiento.
    hay_rescate = (
        pd.notna(fecha_call)
        and fecha_call > fecha
        and (pd.isna(vencimiento_cupon) or fecha_call < vencimiento_cupon)
    )
    # Si hay rescate de un bono igualamos el vencimiento al call para calcular los flujos.
    if hay_rescate:
        vencimiento_cupon = fecha_call

    # Controlamos bonos sin fecha de vencimiento ni call, no se van a poder valorar.
    if pd.isna(vencimiento_cupon):
        mensaje = f"El bono {bono.get('ISIN')} no tiene vencimiento ni fecha call futura."
        raise ValueError(mensaje)

    # Controlamos la frecuencia del cupón, debe ser positiva y divisible entre 12 para poder calcular los flujos.
    frecuencia = float(bono.get("Coupon Frequency", 1))
    if frecuencia <= 0 or 12 % frecuencia != 0:
        mensaje = f"Frecuencia de cupón no admitida para {bono.get('ISIN')}: {frecuencia}."
        raise ValueError(mensaje)

    interes_cupon = float(bono["Coupon"]) / 100
    cupon_anual = nominal * interes_cupon
    importe_cupon = cupon_anual / frecuencia
    periodo_cupon = pd.DateOffset(months=12 // int(frecuencia))
    # Obtenemos el calendario de cupones y filtramos los cupones ya amortizados
    calendario_cupones = pd.date_range(primer_cupon, vencimiento_cupon, freq=periodo_cupon)
    cupones_faltantes = calendario_cupones[calendario_cupones > fecha]
    # Obtenemos la fecha del último cupón pagado, y si no se usa la fecha de emisión
    ultimo_cupon_pagado = calendario_cupones[calendario_cupones <= fecha].max()
    if pd.isna(ultimo_cupon_pagado):
        ultimo_cupon_pagado = pd.to_datetime(bono.get("Issue date"), dayfirst=True)

    # Si la amortización cae entre cupones, añadimos un último pago prorrateado, calculando los devengos.
    cupon_final = importe_cupon
    if vencimiento_cupon not in calendario_cupones:
        inicio_devengo = calendario_cupones.max()
        if pd.isna(inicio_devengo):
            inicio_devengo = ultimo_cupon_pagado
        cupon_final = cupon_anual * (vencimiento_cupon - inicio_devengo).days / 365
        cupones_faltantes = cupones_faltantes.insert(len(cupones_faltantes), vencimiento_cupon)

    # Flujos de pago y plazos en años con base ACT/365.
    flujos = pd.DataFrame(
        {
            "fecha_pago": pd.to_datetime(cupones_faltantes),
            "cupon": importe_cupon,
            "amortizacion": 0.0,
        }
    )
    flujos["dias_hasta_pago"] = (flujos["fecha_pago"] - fecha).dt.days
    flujos["plazo"] = flujos["dias_hasta_pago"] / 365
    flujos.iloc[-1, cast("int", flujos.columns.get_loc("cupon"))] = cupon_final
    # Si el último pago es un vencimiento, añadimos la amortización del nominal.
    flujos.iloc[-1, cast("int", flujos.columns.get_loc("amortizacion"))] = nominal
    flujos["flujo_total"] = flujos["cupon"] + flujos["amortizacion"]

    # Preparamos estructura de curva con factores incluido en el objeto
    curva = curva_descuento.copy()

    curva["fecha"] = pd.to_datetime(curva["fecha"])
    curva["factor_descuento"] = pd.to_numeric(curva["factor_descuento"])

    curva = (
        curva.dropna(subset=["fecha", "factor_descuento"])
        .sort_values("fecha")
        .reset_index(drop=True)
    )

    # Validamos la curva de descuento para asegurarnos datos válidos
    if curva.empty:
        mensaje = "La curva no contiene datos válidos."
        raise ValueError(mensaje)

    if curva["fecha"].duplicated().any():
        mensaje = "La curva contiene fechas duplicadas."
        raise ValueError(mensaje)

    if (curva["factor_descuento"] <= 0).any():
        mensaje = "Los factores de descuento deben ser positivos."
        raise ValueError(mensaje)

    if curva["fecha"].iloc[0] != fecha:
        mensaje = "La curva no corresponde a la fecha de valoración."
        raise ValueError(mensaje)

    curva["plazo"] = (curva["fecha"] - fecha).dt.days / 365

    plazos_curva = curva["plazo"].tolist()
    factores_curva = curva["factor_descuento"].tolist()

    # Obtenemos por cada pago de bono su factor con la interpolación
    flujos["factor_descuento"] = flujos["plazo"].apply(
        lambda plazo: interpolar_factor(
            plazo,
            plazos_curva,
            factores_curva,
        )
    )

    # Calculamos el factor con el spread aplicado
    # Si es 0 el spread el ajuste mantiene el factor de descuento directamente
    flujos["factor_ajustado"] = flujos["factor_descuento"] * flujos["plazo"].apply(
        lambda plazo: exp(-spread * plazo)
    )

    # Descontamos cada pago
    # Para ello se multiplica el dinero cobrado en la fecha por el factor de descuento ajustado
    flujos["valor_actual"] = flujos["flujo_total"] * flujos["factor_ajustado"]

    # Precio sucio: es el sumatorio del valor de los cobros en cada fecha
    precio_sucio = float(flujos["valor_actual"].sum())

    # Calculo del cupon corrido
    # Primero obtenemos de la fecha con el ultimo cupon pagado los días ya devengados
    dias_devengados = (fecha - ultimo_cupon_pagado).days

    cupon_corrido = cupon_anual * dias_devengados / 365

    # Finalmente obtenemos el precio limpio que es el precio sucio - el cupon corrido
    precio_limpio = precio_sucio - cupon_corrido

    return (precio_limpio, precio_sucio, cupon_corrido, flujos)
