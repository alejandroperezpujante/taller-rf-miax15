"""Cartera equiponderada mensual con rentabilidad total de los bonos."""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING, Literal, NamedTuple, cast

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping


class SalidaBono(NamedTuple):
    """Fecha y efectivo de una salida; en el impago, el importe es la recuperación total."""

    fecha: pd.Timestamp
    tipo: Literal["call", "vencimiento", "impago"]
    importe: float


VENTANA_CALL = pd.Timedelta(days=7)
MINIMO_CIERRES = 2


def _cupones_y_corrido(
    fecha: pd.Timestamp,
    calendario: pd.DatetimeIndex,
    emision: pd.Timestamp,
    cupon_anual: float,
    frecuencia: int,
) -> tuple[float, float]:
    pagados = int(calendario.searchsorted(fecha, side="right"))
    ultimo_pago = calendario[pagados - 1] if pagados else emision
    corrido = cupon_anual * (fecha - ultimo_pago).days / 365
    return pagados * cupon_anual / frecuencia, corrido


def _flujos_hasta_salida(
    salida: SalidaBono,
    calendario: pd.DatetimeIndex,
    emision: pd.Timestamp,
    cupon_anual: float,
    frecuencia: int,
) -> float:
    if salida.tipo == "impago":
        # La recuperación registrada ya es el efectivo final del impago.
        anteriores = int(calendario.searchsorted(salida.fecha, side="left"))
        return salida.importe + anteriores * cupon_anual / frecuencia

    pagados, corrido = _cupones_y_corrido(
        salida.fecha, calendario, emision, cupon_anual, frecuencia
    )
    return salida.importe + pagados + corrido


def _resolver_salida(
    bono: pd.Series,
    confirmada: SalidaBono | None,
) -> SalidaBono | None:
    vencimiento = pd.to_datetime(bono["Maturity"], format="%d/%m/%Y")
    if pd.notna(vencimiento) and (confirmada is None or vencimiento < confirmada.fecha):
        return SalidaBono(vencimiento, "vencimiento", 100.0)
    return confirmada


def _call_sin_cotizacion(
    fecha_call: pd.Timestamp,
    inicio: pd.Timestamp,
    fin: pd.Timestamp,
    ultima_cotizacion: pd.Timestamp,
) -> bool:
    return (
        pd.notna(fecha_call)
        and inicio < fecha_call < fin
        and pd.Timedelta(0) < fecha_call - ultima_cotizacion <= VENTANA_CALL
    )


def _rentabilidades_bono(
    bono: pd.Series,
    precios: pd.Series[float],
    fechas: pd.DatetimeIndex,
    precios_rebalanceo: pd.Series[float],
    confirmada: SalidaBono | None,
) -> pd.Series:
    rentabilidades: dict[pd.Timestamp, float] = {}
    emision = pd.Timestamp(pd.to_datetime(bono["Issue date"], format="%d/%m/%Y"))
    primer_cupon = pd.to_datetime(bono["First Coupon Date"], format="%d/%m/%Y")
    frecuencia = int(bono["Coupon Frequency"])
    cupon_anual = float(bono["Coupon"])
    fin_calendario = max(fechas[-1], primer_cupon)
    calendario = pd.date_range(
        primer_cupon, fin_calendario, freq=pd.DateOffset(months=12 // frecuencia)
    )
    precios = precios.dropna().loc[emision:]
    precios_rebalanceo = precios_rebalanceo.dropna()
    salida = _resolver_salida(bono, confirmada)
    fecha_call = pd.Timestamp(pd.to_datetime(bono["Next Call Date"], format="%d/%m/%Y"))

    for inicio, fin in pairwise(fechas):
        if salida is not None and salida.fecha <= inicio:
            break
        if inicio not in precios_rebalanceo.index:
            continue

        pagados_inicio, corrido_inicio = _cupones_y_corrido(
            inicio, calendario, emision, cupon_anual, frecuencia
        )
        precio_inicio = precios_rebalanceo.loc[inicio] + corrido_inicio
        posicion = int(precios.index.searchsorted(fin, side="right")) - 1
        ultima_cotizacion = precios.index[posicion]
        cierre_reciente = fin in precios_rebalanceo.index

        # Una fecha prevista precisa evidencia de cese de cotización para inferir el call.
        if (
            confirmada is None
            and not cierre_reciente
            and bono["Callable"] == "Y"
            and (salida is None or fecha_call < salida.fecha)
            and _call_sin_cotizacion(fecha_call, inicio, fin, ultima_cotizacion)
        ):
            salida = SalidaBono(fecha_call, "call", 100.0)

        if salida is not None and salida.fecha <= fin:
            valor_final = _flujos_hasta_salida(salida, calendario, emision, cupon_anual, frecuencia)
        else:
            # Con un MID reciente, mantenemos el devengo hasta el rebalanceo.
            fecha_valoracion = fin if cierre_reciente else ultima_cotizacion
            pagados_fin, corrido_fin = _cupones_y_corrido(
                fecha_valoracion, calendario, emision, cupon_anual, frecuencia
            )
            mid_final = (
                precios_rebalanceo.loc[fin] if cierre_reciente else float(precios.iloc[posicion])
            )
            valor_final = mid_final + corrido_fin + pagados_fin

        rentabilidades[fin] = (valor_final - pagados_inicio) / precio_inicio - 1

    return pd.Series(rentabilidades, dtype=float).reindex(fechas[1:])


def preparar_precios_rebalanceo(
    universo: pd.DataFrame,
    precios_mid: pd.DataFrame,
    benchmark: pd.Series,
) -> pd.DataFrame:
    """Prepara los MID mensuales con una sesión de tolerancia y sin precios preemisión."""
    precios_mid = precios_mid.sort_index(axis=1)
    fechas_comunes = pd.DatetimeIndex(
        benchmark.dropna().index.intersection(precios_mid.columns)
    ).sort_values()
    fechas = pd.DatetimeIndex(
        fechas_comunes.to_series().groupby(fechas_comunes.to_period("M")).min()
    )
    if len(fechas) < MINIMO_CIERRES:
        mensaje = "El backtest necesita al menos dos meses de precios comunes."
        raise ValueError(mensaje)

    emisiones = pd.to_datetime(universo.set_index("ISIN")["Issue date"], format="%d/%m/%Y")
    emisiones = emisiones.reindex(precios_mid.index)
    precios_mensuales = {}
    for fecha, posicion in zip(fechas, fechas_comunes.get_indexer(fechas), strict=True):
        anterior = fechas_comunes[max(0, int(posicion) - 1)]
        ventana = precios_mid.loc[:, anterior:fecha]
        # Enmascarar cada fecha antes de rellenar evita usar un MID previo a la emisión.
        validos = pd.DataFrame(
            {
                dia: ventana[dia].where(emisiones.le(dia))
                for dia in pd.DatetimeIndex(ventana.columns)
            }
        )
        precios_mensuales[fecha] = validos.ffill(axis=1).iloc[:, -1]

    return pd.DataFrame(precios_mensuales)


def backtest_equiponderado(
    universo: pd.DataFrame,
    precios_mid: pd.DataFrame,
    benchmark: pd.Series,
    salidas_confirmadas: Mapping[str, SalidaBono] | None = None,
    *,
    precios_rebalanceo: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula rentabilidades mensuales en decimal y pesos al inicio de cada periodo."""
    precios_mid = precios_mid.sort_index(axis=1)
    if precios_rebalanceo is None:
        precios_rebalanceo = preparar_precios_rebalanceo(universo, precios_mid, benchmark)
    fechas = pd.DatetimeIndex(precios_rebalanceo.columns)

    salidas = salidas_confirmadas or {}
    por_bono = {}
    for _, bono in universo.iterrows():
        isin = str(bono["ISIN"])
        cotizaciones = cast("pd.Series[float]", precios_mid.loc[isin])
        cotizaciones_rebalanceo = cast("pd.Series[float]", precios_rebalanceo.loc[isin])
        por_bono[isin] = _rentabilidades_bono(
            bono, cotizaciones, fechas, cotizaciones_rebalanceo, salidas.get(isin)
        )

    rentabilidades = pd.DataFrame(por_bono)
    seleccion = rentabilidades.notna()
    cantidad = seleccion.sum(axis=1).clip(lower=1)
    pesos = seleccion.div(cantidad, axis=0)
    pesos.index = fechas[:-1].rename("Fecha rebalanceo")

    rent_cartera = rentabilidades.mean(axis=1).fillna(0.0)
    rent_benchmark = benchmark.loc[fechas].pct_change().iloc[1:]
    resultados = pd.DataFrame(
        {
            "Fecha Mes": fechas[1:],
            "Rent cartera": rent_cartera.to_numpy(),
            "Rent benchmark": rent_benchmark.to_numpy(),
            "Dif": (rent_cartera - rent_benchmark).to_numpy(),
        }
    )
    return resultados, pesos
