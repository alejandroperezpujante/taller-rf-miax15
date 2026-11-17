"""Funciones auxiliares del taller de renta fija."""

from bisect import bisect_left
from math import exp
from typing import cast

import pandas as pd


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

    # Formula interpolación exponencial
    factor_interpolado = factor_anterior**peso_anterior * factor_siguiente**peso_siguiente

    return float(factor_interpolado)


# Funcion base para calcular la valoración de un bono
def valoracion_bono(
    bono: pd.Series,
    curva_descuento: pd.DataFrame,
    fecha_valor: str | pd.Timestamp = "2026-10-01",
    nominal: float = 100,
    spread: float = 0,
) -> tuple[float, float, float, pd.DataFrame]:
    """Muestra los flujos pendientes del bono anual del ejemplo de clase.

    Avance parcial: calcula cupones, amortización final y plazos ACT/365.
    Todavía no utiliza la curva ni el spread, ni devuelve una valoración.
    La prueba usa nominal 100 y fecha valor 01/10/2026. El calendario supone
    cupones anuales regulares; no contempla otros bonos ni fechas call.
    Se conserva la amortización de 100 del desarrollo original.
    """
    fecha = pd.Timestamp(fecha_valor)
    # Importes del cupón: nominal por tipo anual.
    primer_cupon = pd.to_datetime(bono["First Coupon Date"], dayfirst=True)
    vencimiento_cupon = pd.to_datetime(bono["Maturity"], dayfirst=True)
    interes_cupon = float(bono["Coupon"]) / 100
    importe_cupon = nominal * interes_cupon
    # Calendario de cupones anuales pendientes.
    calendario_cupones = pd.date_range(primer_cupon, vencimiento_cupon, freq=pd.DateOffset(years=1))
    cupones_faltantes = calendario_cupones[calendario_cupones > fecha]
    ultimo_cupon_pagado = calendario_cupones[calendario_cupones <= fecha].max()

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
    flujos.iloc[-1, cast("int", flujos.columns.get_loc("amortizacion"))] = nominal
    flujos["flujo_total"] = flujos["cupon"] + flujos["amortizacion"]

    print(flujos)  # noqa: T201 - se conserva para poder hacer pruebas

    # Preparamos estructura de curva con factores incluido en el objeto
    curva = curva_descuento.copy()

    curva["fecha"] = pd.to_datetime(curva["fecha"])
    curva["factor_descuento"] = pd.to_numeric(curva["factor_descuento"])

    curva = (
        curva.dropna(subset=["fecha", "factor_descuento"])
        .sort_values("fecha")
        .reset_index(drop=True)
    )

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

    cupon_corrido = nominal * interes_cupon * dias_devengados / 365

    # Finalmente obtenemos el precio limpio que es el precio sucio - el cupon corrido
    precio_limpio = precio_sucio - cupon_corrido

    return (precio_limpio, precio_sucio, cupon_corrido, flujos)
