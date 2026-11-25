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
    """Acumula los cupones pagados y calcula el cupón corrido con base de 365 días."""
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
    """Suma los cupones que correspondan al efectivo de la salida."""
    if salida.tipo == "impago":
        # A la recuperación solo se añaden los cupones pagados antes del impago.
        anteriores = int(calendario.searchsorted(salida.fecha, side="left"))
        return salida.importe + anteriores * cupon_anual / frecuencia

    # En una amortización se incluyen los cupones pagados y el corrido hasta la salida.
    pagados, corrido = _cupones_y_corrido(
        salida.fecha, calendario, emision, cupon_anual, frecuencia
    )
    return salida.importe + pagados + corrido


def _resolver_salida(
    bono: pd.Series,
    confirmada: SalidaBono | None,
) -> SalidaBono | None:
    """Elige la primera salida conocida entre el vencimiento y el evento confirmado."""
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
    """Comprueba si la call cae dentro del periodo y en los siete días tras el último MID."""
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
    """Calcula la rentabilidad total del bono entre rebalanceos consecutivos."""
    # Solo podemos abrir una posición en fechas con un MID de rebalanceo.
    precios_rebalanceo = precios_rebalanceo.dropna()
    if precios_rebalanceo.empty:
        return pd.Series(index=fechas[1:], dtype=float)

    # Construimos el calendario de pagos desde el primer cupón según su frecuencia.
    rentabilidades: dict[pd.Timestamp, float] = {}
    emision = pd.Timestamp(pd.to_datetime(bono["Issue date"], format="%d/%m/%Y"))
    primer_cupon = pd.to_datetime(bono["First Coupon Date"], format="%d/%m/%Y")
    frecuencia = int(bono["Coupon Frequency"])
    cupon_anual = float(bono["Coupon"])
    fin_calendario = max(fechas[-1], primer_cupon)
    calendario = pd.date_range(
        primer_cupon, fin_calendario, freq=pd.DateOffset(months=12 // frecuencia)
    )
    # Reunimos las cotizaciones desde la emisión y las posibles fechas de salida.
    precios = precios.dropna().loc[emision:]
    salida = _resolver_salida(bono, confirmada)
    fecha_call = pd.Timestamp(
        pd.to_datetime(bono["Next Call Date"], format="%d/%m/%Y", errors="coerce")
    )

    for inicio, fin in pairwise(fechas):
        # Dejamos de valorar tras la salida y saltamos periodos sin precio de entrada.
        if salida is not None and salida.fecha <= inicio:
            break
        if inicio not in precios_rebalanceo.index:
            continue

        # Sumamos el corrido al MID de entrada para medir el desembolso inicial.
        pagados_inicio, corrido_inicio = _cupones_y_corrido(
            inicio, calendario, emision, cupon_anual, frecuencia
        )
        precio_inicio = precios_rebalanceo.loc[inicio] + corrido_inicio
        # Localizamos el último MID disponible hasta el cierre del periodo.
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
            # Al salir del bono, valoramos la posición por el efectivo acumulado.
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

        # Restamos los cupones anteriores o iguales a la entrada para aislar el rendimiento del mes.
        rentabilidades[fin] = (valor_final - pagados_inicio) / precio_inicio - 1

    return pd.Series(rentabilidades, dtype=float).reindex(fechas[1:])


def limpiar_precios_historicos(
    universo: pd.DataFrame,
    precios_mid: pd.DataFrame,
    salidas_confirmadas: Mapping[str, SalidaBono] | None = None,
) -> tuple[pd.DataFrame, pd.Series[int]]:
    """Descarta fechas inválidas y enmascara MID inválidos o fuera de la vida del bono."""
    # Ordenamos las sesiones con fecha válida y comprobamos que cada una sea única.
    fechas = pd.to_datetime(precios_mid.columns, format="%d/%m/%Y", errors="coerce")
    fechas_validas = fechas.notna()
    historico = precios_mid.loc[:, fechas_validas].set_axis(fechas[fechas_validas], axis=1)
    historico = historico.sort_index(axis=1)
    if historico.columns.has_duplicates:
        mensaje = "El histórico de precios contiene fechas duplicadas."
        raise ValueError(mensaje)

    # Alineamos los bonos por ISIN para validar sus fechas de emisión y vencimiento.
    bonos = universo.set_index("ISIN").reindex(historico.index)
    emisiones = pd.to_datetime(bonos["Issue date"], format="%d/%m/%Y", errors="coerce")
    vencimientos = pd.to_datetime(bonos["Maturity"], format="%d/%m/%Y", errors="coerce")
    vida_valida = emisiones.notna() & (bonos["Maturity"].isna() | vencimientos.gt(emisiones))

    # Una salida confirmada puede adelantar el fin de vida o fijarlo si falta el vencimiento.
    fechas_fin = vencimientos.to_dict()
    for isin, salida in (salidas_confirmadas or {}).items():
        fin = fechas_fin.get(isin, pd.NaT)
        if pd.isna(fin) or salida.fecha < fin:
            fechas_fin[isin] = salida.fecha
    fin_vida = pd.Series(fechas_fin, dtype="datetime64[ns]").reindex(historico.index)

    # Conservamos como candidatos los MID numéricos, positivos y finitos.
    numericos = historico.apply(pd.to_numeric, errors="coerce")
    mid_valido = numericos.gt(0) & numericos.lt(float("inf"))
    limpia = {}
    antes_emision = pd.Series(data=False, index=historico.index)
    desde_salida = pd.Series(data=False, index=historico.index)
    # Filtramos cada sesión y marcamos los bonos con cotizaciones fuera de su vida.
    for fecha in pd.DatetimeIndex(historico.columns):
        antes = emisiones.gt(fecha)
        despues = fin_vida.le(fecha)
        antes_emision |= mid_valido[fecha] & antes
        desde_salida |= mid_valido[fecha] & despues
        limpia[fecha] = numericos[fecha].where(mid_valido[fecha] & vida_valida & ~antes & ~despues)

    # Contamos las incidencias por fecha, cotización o bono según su causa.
    recuentos = {
        "Columnas con fecha inválida": len(fechas) - len(historico.columns),
        "MID no numéricos, no positivos o infinitos": int(
            (historico.notna() & ~mid_valido).sum().sum()
        ),
        "Bonos con fechas de vida inválidas": int((~vida_valida).sum()),
        "Bonos con MID antes de la emisión": int(antes_emision.sum()),
        "Bonos con MID desde el vencimiento o la salida": int(desde_salida.sum()),
    }
    return pd.DataFrame(limpia, index=historico.index), pd.Series(recuentos, name="Recuento")


def preparar_precios_rebalanceo(
    universo: pd.DataFrame,
    precios_mid: pd.DataFrame,
    benchmark: pd.Series,
) -> pd.DataFrame:
    """Prepara los MID mensuales con una sesión de tolerancia y sin precios preemisión."""
    # Elegimos para cada mes la primera sesión compartida con el benchmark.
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
        # Buscamos el MID más reciente desde la sesión común anterior hasta el rebalanceo.
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
    # Usamos los cierres de rebalanceo facilitados o los calculamos a partir del histórico.
    precios_mid = precios_mid.sort_index(axis=1)
    if precios_rebalanceo is None:
        precios_rebalanceo = preparar_precios_rebalanceo(universo, precios_mid, benchmark)
    fechas = pd.DatetimeIndex(precios_rebalanceo.columns)

    # Calculamos la rentabilidad de cada bono con sus cotizaciones y flujos.
    salidas = salidas_confirmadas or {}
    por_bono = {}
    for _, bono in universo.iterrows():
        isin = str(bono["ISIN"])
        cotizaciones = cast("pd.Series[float]", precios_mid.loc[isin])
        cotizaciones_rebalanceo = cast("pd.Series[float]", precios_rebalanceo.loc[isin])
        por_bono[isin] = _rentabilidades_bono(
            bono, cotizaciones, fechas, cotizaciones_rebalanceo, salidas.get(isin)
        )

    # Repartimos el peso por igual entre los bonos con rentabilidad disponible en el periodo.
    rentabilidades = pd.DataFrame(por_bono)
    seleccion = rentabilidades.notna()
    cantidad = seleccion.sum(axis=1).clip(lower=1)
    pesos = seleccion.div(cantidad, axis=0)
    pesos.index = fechas[:-1].rename("Fecha rebalanceo")

    # Comparamos la media de los bonos con la variación del benchmark entre los mismos cierres.
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
