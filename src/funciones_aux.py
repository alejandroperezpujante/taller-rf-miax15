"""Funciones auxiliares del taller de renta fija."""

from typing import cast

import pandas as pd


#Funcion base para calcular la valoración de un bono
def valoracion_bono(
    bono: pd.Series,
    curva_descuento: pd.DataFrame,  # noqa: ARG001 - pendiente del descuento de flujos
    fecha_valor: str | pd.Timestamp = "2026-10-01",
    nominal: float = 100,
    spread: float = 0,  # noqa: ARG001 - pendiente del ajuste por spread (no usado por el momento)
) -> None:
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
    _ultimo_cupon_pagado = calendario_cupones[calendario_cupones <= fecha].max()

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
    flujos.iloc[-1, cast("int", flujos.columns.get_loc("amortizacion"))] = 100
    print(flujos)  # noqa: T201 - se conserva para poder hacer pruebas
