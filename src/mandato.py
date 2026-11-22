"""Selección de emisiones para el mandato de renta fija."""

import pandas as pd
import pulp

RATINGS_IG = frozenset("AAA AA+ AA AA- A+ A A- BBB+ BBB BBB-".split())
SENIOR = frozenset({"Sr Unsecured", "Sr Preferred", "Sr Non Preferred", "Secured", "1st lien"})
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


def optimizar_mandato(datos: pd.DataFrame, pesos_previos: pd.Series | None = None) -> pd.Series:
    """Maximiza la yield ponderada y permite desempatar por rotación."""
    modelo = pulp.LpProblem("mandato", pulp.LpMaximize)
    pesos = pulp.LpVariable.dicts("peso", datos.index, lowBound=0, upBound=0.10)
    seleccion = pulp.LpVariable.dicts("seleccion", datos.index, cat="Binary")
    yield_cartera = pulp.lpSum(pesos[i] * rendimiento for i, rendimiento in datos["Yield"].items())
    modelo += yield_cartera
    modelo += pulp.lpSum(pesos.values()) == 1
    modelo += pulp.lpSum(seleccion.values()) <= 20
    for i in datos.index:
        modelo += pesos[i] <= 0.10 * seleccion[i]
    modelo += pulp.lpSum(pesos[i] * d for i, d in datos["Dur"].items()) <= 3
    riesgo_alto = datos.index[~datos["Rating"].isin(RATINGS_IG)]
    modelo += pulp.lpSum(pesos[i] for i in riesgo_alto) <= 0.10
    for emisiones in datos.groupby("Issuer").groups.values():
        modelo += pulp.lpSum(pesos[i] for i in emisiones) <= 0.15

    solver = pulp.PULP_CBC_CMD(msg=False)
    if modelo.solve(solver) != pulp.LpStatusOptimal:
        raise ValueError("El mandato no tiene una solución óptima con los candidatos disponibles.")

    if pesos_previos is not None:
        # Las ventas de emisiones que salieron del universo son constantes en este desempate.
        modelo += yield_cartera >= pulp.value(yield_cartera) - 1e-10
        cambios = pulp.LpVariable.dicts("cambio", datos.index, lowBound=0)
        for i in datos.index:
            diferencia = pesos[i] - float(pesos_previos.get(i, 0))
            modelo += cambios[i] >= diferencia
            modelo += cambios[i] >= -diferencia
        modelo.sense = pulp.LpMinimize
        modelo.setObjective(pulp.lpSum(cambios.values()))
        if modelo.solve(solver) != pulp.LpStatusOptimal:
            raise ValueError("No se pudo resolver el desempate por rotación.")

    resultado = pd.Series({i: pesos[i].value() for i in datos.index}, dtype=float, name="Peso")
    resultado = resultado[resultado.gt(0)]
    controles = [
        abs(resultado.sum() - 1) <= TOLERANCIA,
        len(resultado) <= 20,
        resultado.max() <= 0.10 + TOLERANCIA,
        resultado.groupby(datos["Issuer"]).sum().max() <= 0.15 + TOLERANCIA,
        resultado.reindex(riesgo_alto, fill_value=0).sum() <= 0.10 + TOLERANCIA,
        resultado.dot(datos.loc[resultado.index, "Dur"]) <= 3 + TOLERANCIA,
    ]
    if not all(controles):
        raise ValueError("La solución incumple alguno de los límites del mandato.")
    return resultado.sort_values(ascending=False)
