"""Selección de emisiones para el mandato de renta fija."""

import pandas as pd
import pulp  # type: ignore[import-untyped]

RATINGS_IG = frozenset(["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-"])
SENIOR = frozenset({"Sr Unsecured", "Sr Preferred", "Sr Non Preferred", "Secured", "1st lien"})
MAX_BONOS = 20
MAX_DURACION = 3
MAX_EMISION = 0.10
MAX_EMISOR = 0.15
TOLERANCIA_YIELD = 1e-10
TOLERANCIA = 1e-8


def candidatos_mandato(
    universo: pd.DataFrame, sensibilidades: pd.DataFrame, fecha: pd.Timestamp
) -> pd.DataFrame:
    """Aplica los filtros a la foto del universo y a las sensibilidades de esa fecha."""
    datos = universo.merge(sensibilidades[["ISIN", "Yield", "Dur"]], on="ISIN", validate="1:1")
    emision = pd.to_datetime(datos["Issue date"], format="%d/%m/%Y")
    vencimiento = pd.to_datetime(datos["Maturity"], format="%d/%m/%Y")
    elegibles = (
        emision.le(fecha)
        & (vencimiento.isna() | vencimiento.gt(fecha))
        & ~datos["RescatadoAsumido"]
        & datos["Seniority"].isin(SENIOR)
        & datos["Outstanding Amount"].gt(500_000_000)
        & datos["Price"].gt(0)
        & datos["Issuer"].notna()
        & datos["Yield"].notna()
        & datos["Dur"].gt(0)
    )
    return datos.loc[elegibles].set_index("ISIN").sort_index()


def _resolver(modelo: pulp.LpProblem, solver: pulp.LpSolver) -> bool:
    resultado = modelo.solve(solver)
    # PuLP 4 devuelve estadísticas; PuLP 3 devuelve el código de estado.
    if hasattr(resultado, "status"):
        if resultado.status.name == "GapLimit":
            # CBC puede comunicar el óptimo como parada por gap aun con cota idéntica.
            return (
                resultado.has_solution
                and resultado.best_bound is not None
                and abs(resultado.objective - resultado.best_bound) <= TOLERANCIA_YIELD
            )
        return bool(resultado.status.name == "Optimal")
    return bool(resultado == pulp.LpStatusOptimal)


def optimizar_mandato(datos: pd.DataFrame, pesos_previos: pd.Series | None = None) -> pd.Series:
    """Maximiza la yield ponderada y permite desempatar por rotación."""
    modelo = pulp.LpProblem("mandato", pulp.LpMaximize)
    pesos = {
        i: modelo.add_variable(f"peso_{i}", lowBound=0, upBound=MAX_EMISION) for i in datos.index
    }
    seleccion = {i: modelo.add_variable(f"seleccion_{i}", cat="Binary") for i in datos.index}
    yield_cartera = pulp.lpSum(pesos[i] * rendimiento for i, rendimiento in datos["Yield"].items())
    modelo += yield_cartera
    modelo += pulp.lpSum(pesos.values()) == 1
    modelo += pulp.lpSum(seleccion.values()) <= MAX_BONOS
    for i in datos.index:
        modelo += pesos[i] <= MAX_EMISION * seleccion[i]
    modelo += pulp.lpSum(pesos[i] * d for i, d in datos["Dur"].items()) <= MAX_DURACION
    riesgo_alto = datos.index[~datos["Rating"].isin(RATINGS_IG)]
    modelo += pulp.lpSum(pesos[i] for i in riesgo_alto) <= MAX_EMISION
    for emisiones in datos.groupby("Issuer").groups.values():
        modelo += pulp.lpSum(pesos[i] for i in emisiones) <= MAX_EMISOR

    cbc = getattr(pulp, "PULP_CBC_CMD", pulp.COIN_CMD)
    solver = cbc(msg=False)

    if not _resolver(modelo, solver):
        mensaje = "El mandato no tiene una solución óptima con los candidatos disponibles."
        raise ValueError(mensaje)

    if pesos_previos is not None:
        # Las ventas de emisiones que salieron del universo son constantes en este desempate.
        modelo += yield_cartera >= pulp.value(yield_cartera) - TOLERANCIA_YIELD
        cambios = {i: modelo.add_variable(f"cambio_{i}", lowBound=0) for i in datos.index}
        for i in datos.index:
            diferencia = pesos[i] - float(pesos_previos.get(i, 0))
            modelo += cambios[i] >= diferencia
            modelo += cambios[i] >= -diferencia
        modelo.sense = pulp.LpMinimize
        modelo.setObjective(pulp.lpSum(cambios.values()))
        if not _resolver(modelo, solver):
            mensaje = "No se pudo resolver el desempate por rotación."
            raise ValueError(mensaje)

    resultado = pd.Series({i: pesos[i].value() for i in datos.index}, dtype=float, name="Peso")
    resultado = resultado[resultado.gt(TOLERANCIA)]
    resultado = resultado.mask((resultado - 0.10).abs().le(TOLERANCIA), 0.10)
    controles = [
        abs(resultado.sum() - 1) <= TOLERANCIA,
        len(resultado) <= MAX_BONOS,
        resultado.max() <= MAX_EMISION + TOLERANCIA,
        resultado.groupby(datos["Issuer"]).sum().max() <= MAX_EMISOR + TOLERANCIA,
        resultado.reindex(riesgo_alto, fill_value=0).sum() <= MAX_EMISION + TOLERANCIA,
        resultado.dot(datos.loc[resultado.index, "Dur"]) <= MAX_DURACION + TOLERANCIA,
    ]
    if not all(controles):
        mensaje = "La solución incumple alguno de los límites del mandato."
        raise ValueError(mensaje)
    return resultado.sort_values(ascending=False)


def _preparar_cobros(cobros: pd.DataFrame) -> pd.DataFrame:
    pagos = cobros.copy()
    pagos["Fecha"] = pd.to_datetime(pagos["Fecha"])
    terminal = pagos["Tipo"].isin(["call", "vencimiento", "impago"])
    if (
        not (terminal | pagos["Tipo"].eq("cupon")).all()
        or not pagos["Importe"].between(0, float("inf"), inclusive="left").all()
        or pagos.duplicated(["Fecha", "ISIN", "Tipo"]).any()
        or pagos.loc[terminal, "ISIN"].duplicated().any()
    ):
        mensaje = "Revisar tipos, importes o duplicados en los cobros."
        raise ValueError(mensaje)
    return pagos.sort_values("Fecha")


def _liquidar_pagos(
    pagos: pd.DataFrame, unidades: pd.Series, retirados: set[str], motivos: list[str]
) -> tuple[pd.Series, float, list[tuple[str, str, pd.Timestamp, float]]]:
    efectivo = 0.0
    salidas = []
    for pago in pagos.itertuples():
        cantidad = unidades.get(pago.ISIN, 0.0)
        efectivo += cantidad * pago.Importe
        if pago.Tipo != "cupon":
            retirados.add(str(pago.ISIN))
            if cantidad:
                salidas.append(
                    (
                        str(pago.ISIN),
                        str(pago.Tipo),
                        pd.Timestamp(str(pago.Fecha)),
                        float(cantidad * pago.Importe),
                    )
                )
                unidades = unidades.drop(pago.ISIN)
                motivos.append(f"{pago.Tipo} {pago.ISIN}")
    return unidades, efectivo, salidas


def _sesiones_mandato(
    fotos: dict[pd.Timestamp, pd.DataFrame], benchmark: pd.Series
) -> pd.DatetimeIndex:
    fechas = pd.DatetimeIndex(sorted(fotos))
    if fechas.empty or not benchmark.reindex(fechas).gt(0).all():
        mensaje = "Faltan sesiones o niveles válidos del benchmark."
        raise ValueError(mensaje)
    sesiones = benchmark.dropna().sort_index().loc[fechas[0] : fechas[-1]].index
    if not fechas.equals(sesiones):
        mensaje = "Se necesita una foto por sesión del benchmark."
        raise ValueError(mensaje)
    return fechas


def simular_mandato(
    fotos: dict[pd.Timestamp, pd.DataFrame], cobros: pd.DataFrame, benchmark: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simula desde la primera foto. Cada foto contiene el universo de una sesión.

    Las fotos se indexan por ISIN e incluyen PrecioSucioMercado, Yield, Dur y
    Elegible. Los cobros por 100 de nominal llevan Fecha, ISIN, Tipo e Importe.
    Un pago terminal incluye el cupón final que corresponda.
    """
    fechas = _sesiones_mandato(fotos, benchmark)
    pagos = _preparar_cobros(cobros)
    # Si un pago cae en festivo, el efectivo se puede invertir en la sesión siguiente.
    efectivo = 100.0
    unidades = pd.Series(dtype=float)
    retirados = set(pagos.loc[pagos["Tipo"].ne("cupon") & pagos["Fecha"].le(fechas[0]), "ISIN"])
    anterior = None
    base_benchmark = benchmark.loc[fechas[0]]
    diario = []
    historial: list[tuple[pd.Timestamp, str, float, float, str, pd.Timestamp | None, float]] = []
    for fecha in fechas:
        foto = fotos[fecha]
        motivos = []
        salidas: list[tuple[str, str, pd.Timestamp, float]] = []
        if anterior is None:
            motivos.append("inicial")
        else:
            if anterior.to_period("M") != fecha.to_period("M"):
                motivos.append("mensual")
            periodo = pagos["Fecha"].between(anterior, fecha, inclusive="right")
            unidades, cobrado, salidas = _liquidar_pagos(
                pagos.loc[periodo], unidades, retirados, motivos
            )
            efectivo += cobrado

        precios = foto["PrecioSucioMercado"]
        valores = unidades * precios.reindex(unidades.index)
        if not precios.reindex(unidades.index).between(0, float("inf"), inclusive="neither").all():
            mensaje = f"Falta precio sucio para una posición viva el {fecha:%d/%m/%Y}."
            raise ValueError(mensaje)
        patrimonio = efectivo + valores.sum()
        previos = valores / patrimonio
        duraciones = foto["Dur"].reindex(unidades.index)
        if duraciones.isna().any():
            mensaje = f"Falta duración de una posición viva el {fecha:%d/%m/%Y}."
            raise ValueError(mensaje)
        duracion = float((previos * duraciones).sum())
        if duracion > 3 + TOLERANCIA:
            motivos.append("duración")
        pesos = previos
        if motivos:
            elegibles = foto.loc[
                foto["Elegible"]
                & ~foto.index.isin(retirados)
                & precios.between(0, float("inf"), inclusive="neither")
            ]
            pesos = optimizar_mandato(elegibles, previos if anterior is not None else None)
            unidades = patrimonio * pesos / precios.loc[pesos.index]
            efectivo = patrimonio - (unidades * precios.loc[pesos.index]).sum()
            duracion = pesos.dot(foto.loc[pesos.index, "Dur"])
            # Los bonos amortizados ya estaban en efectivo antes de enviar las órdenes.
            historial.extend(
                (fecha, isin, importe / patrimonio, 0.0, tipo, fecha_pago, importe)
                for isin, tipo, fecha_pago, importe in salidas
            )
            for isin in previos.index.union(pesos.index):
                antes, despues = float(previos.get(isin, 0)), float(pesos.get(isin, 0))
                if abs(despues - antes) > TOLERANCIA:
                    operacion = "entrada" if antes == 0 else "salida" if despues == 0 else "ajuste"
                    historial.append((fecha, isin, antes, despues, operacion, None, 0.0))
        diario.append(
            (
                fecha,
                ", ".join(motivos),
                patrimonio,
                efectivo,
                len(pesos),
                duracion,
                100 * benchmark.loc[fecha] / base_benchmark,
            )
        )
        anterior = fecha

    return (
        pd.DataFrame(
            diario, columns=["Fecha", "Motivo", "Cartera", "Efectivo", "Bonos", "Dur", "RECMTREU"]
        ).set_index("Fecha"),
        pd.DataFrame(
            historial,
            columns=[
                "Fecha",
                "ISIN",
                "PesoAntes",
                "PesoDespues",
                "Operacion",
                "FechaCobro",
                "CobroBase100",
            ],
        ),
    )
