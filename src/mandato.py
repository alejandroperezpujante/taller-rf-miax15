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
        & ~datos["CallPrevia"]
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


def optimizar_mandato(datos: pd.DataFrame) -> pd.Series:
    """Maximiza la yield ponderada respetando los límites del mandato."""
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
