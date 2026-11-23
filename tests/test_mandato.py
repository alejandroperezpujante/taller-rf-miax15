"""Pruebas de reinversión del mandato en sesiones consecutivas."""

import pandas as pd
import pytest
from src.backtest import SalidaBono, cobros_mandato
from src.mandato import simular_mandato


def test_reinvierte_call_y_reoptimiza_toda_la_cartera() -> None:
    """Un call en domingo y el corte mensual producen una única operación el lunes."""
    fechas = pd.to_datetime(["2026-10-30", "2026-11-02", "2026-11-03"])
    indices = [f"B{i}" for i in range(12)]
    foto = pd.DataFrame(
        {
            "PrecioSucioMercado": 100.0,
            "Yield": [0.06] * 10 + [0.04, 0.03],
            "Dur": 2.0,
            "Elegible": True,
            "Issuer": indices,
            "Rating": "A",
        },
        index=indices,
    )
    lunes = foto.copy()
    lunes.loc["B11", "Yield"] = 0.09
    lunes.loc["B9", "Yield"] = 0.02
    martes = lunes.copy()
    martes.loc["B10", "PrecioSucioMercado"] = 99.0
    pagos = pd.DataFrame(
        [
            ("2026-11-01", "B0", "call", 102.0),
            ("2026-11-03", "B10", "cupon", 1.0),
        ],
        columns=["Fecha", "ISIN", "Tipo", "Importe"],
    )
    diario, cambios = simular_mandato(
        dict(zip(fechas, [foto, lunes, martes], strict=True)),
        pagos,
        pd.Series([200, 202, 202], index=fechas),
    )
    assert diario["Cartera"].to_list() == pytest.approx([100, 100.2, 100.2])
    assert diario["Motivo"].to_list() == ["inicial", "mensual, call B0", ""]
    assert diario["Efectivo"].iloc[-1] == pytest.approx(0.1002)
    assert diario["RECMTREU"].iloc[-1] == pytest.approx(101)
    rescate = cambios.loc[cambios["Operacion"].eq("call")].iloc[0]
    assert rescate.ISIN == "B0"
    assert rescate.FechaCobro == pd.Timestamp("2026-11-01")
    assert rescate.CobroBase100 == pytest.approx(10.2)
    assert cambios.loc[cambios["Operacion"].eq("entrada"), "ISIN"].nunique() == 12
    assert diario.Bonos.iloc[-1] == 10
    tras = cambios[cambios.Fecha == fechas[1]].set_index("ISIN").PesoDespues
    assert "B0" not in tras[tras.gt(0)]
    assert tras["B9"] == 0
    assert tras.loc[["B10", "B11"]].to_list() == pytest.approx([0.1, 0.1])


def test_rebalanceo_intrames_solo_con_salida_confirmada() -> None:
    """Un call previsto no amortiza. Al confirmarse, incluye el devengo una sola vez."""
    fechas = pd.to_datetime(["2026-10-01", "2026-10-02"])
    ids = [f"B{i}" for i in range(11)]
    universo = pd.DataFrame(
        {
            "ISIN": ids,
            "First Coupon Date": "02/10/2025",
            "Issue date": "02/10/2024",
            "Maturity": "02/10/2027",
            "Coupon Frequency": 1,
            "Coupon": 2.0,
            "Next Call Date": "02/10/2026",
            "Callable": "Y",
        }
    )
    foto = pd.DataFrame(
        {
            "Issuer": ids,
            "Rating": "A",
            "Elegible": True,
            "Dur": 1.0,
            "Yield": [0.05] * 10 + [0.04],
            "PrecioSucioMercado": 100.0,
        },
        index=ids,
    )
    for confirmado in [False, True]:
        pagos = cobros_mandato(
            universo, *fechas, {"B0": SalidaBono(fechas[1], "call", 100.0)} if confirmado else {}
        )
        diario, _ = simular_mandato(
            dict.fromkeys(fechas, foto), pagos, pd.Series(100.0, index=fechas)
        )
        assert diario.Motivo.iloc[-1] == ("call B0" if confirmado else "")
        assert diario.Cartera.iloc[-1] == pytest.approx(102.0)
        assert diario.Efectivo.iloc[-1] == pytest.approx(0.0 if confirmado else 2.0)
