"""Regresiones de entradas, salidas y cupones del backtest mensual."""

from math import nan

import pandas as pd
import pytest
from src.backtest import SalidaBono, backtest_equiponderado, limpiar_precios_historicos


def _bono(isin: str, **datos: str | float | None) -> dict[str, str | float | None]:
    return {
        "ISIN": isin,
        "Issue date": "02/01/2023",
        "First Coupon Date": "02/01/2024",
        "Maturity": "02/01/2025",
        "Coupon Frequency": 1,
        "Coupon": 0,
        "Callable": "N",
        "Next Call Date": None,
    } | datos


def test_entradas_y_salidas_conservan_la_rentabilidad_del_mes() -> None:
    """Ignore pre-issue prices and keep the proceeds of bonds held when they exit."""
    universo = pd.DataFrame(
        [
            _bono("nuevo", **{"Issue date": "15/01/2024"}),
            _bono("vence", Maturity="15/01/2024"),
            _bono("call", Callable="Y", **{"Next Call Date": "28/01/2024"}),
            _bono("continua", Callable="Y", **{"Next Call Date": "28/01/2024"}),
            _bono("impago"),
            _bono("hueco"),
        ]
    )
    fechas = pd.to_datetime(["2024-01-02", "2024-01-25", "2024-01-31", "2024-02-01", "2024-03-01"])
    precios = pd.DataFrame(
        {
            "nuevo": [100, 102, 102, 103, 104],
            "vence": [98, nan, nan, nan, nan],
            "call": [99, 99, nan, nan, nan],
            "continua": [100, 103, 104, 104, 105],
            "impago": [100, 80, nan, nan, nan],
            "hueco": [100, 96, nan, nan, 101],
        },
        index=fechas,
    ).T
    benchmark = pd.Series([100.0, 101.0, 101.5, 102.0, 104.0], index=fechas)
    impago = SalidaBono(pd.Timestamp("2024-01-29"), "impago", 30)
    backtest, pesos = backtest_equiponderado(universo, precios, benchmark, {"impago": impago})

    assert pesos.iloc[0].tolist() == pytest.approx([0, 0.2, 0.2, 0.2, 0.2, 0.2])
    assert pesos.iloc[1].tolist() == pytest.approx([0.5, 0, 0, 0.5, 0, 0])
    rent_primera = ((100 / 98 - 1) + (100 / 99 - 1) + 0.04 - 0.70 - 0.04) / 5
    rent_segunda = ((104 / 103 - 1) + (105 / 104 - 1)) / 2
    assert backtest["Rent cartera"].tolist() == pytest.approx([rent_primera, rent_segunda])
    assert backtest["Rent benchmark"].tolist() == pytest.approx([0.02, 104 / 102 - 1])


def test_total_return_incluye_cupon_y_corrido_semestral() -> None:
    """Include the cash coupon and the dirty-price change over the same month."""
    universo = pd.DataFrame(
        [
            _bono(
                "semestral",
                **{
                    "First Coupon Date": "15/07/2023",
                    "Coupon Frequency": 2,
                    "Coupon": 4,
                },
            ),
        ]
    )
    fechas = pd.to_datetime(["2024-01-02", "2024-02-01"])
    precios = pd.DataFrame([[100.0, 100.0]], index=["semestral"], columns=fechas)
    benchmark = pd.Series([100.0, 101.0], index=fechas)

    backtest, _ = backtest_equiponderado(universo, precios, benchmark)

    precio_inicio = 100 + 4 * 171 / 365
    valor_fin = 100 + 4 * 17 / 365 + 2
    assert backtest["Rent cartera"].iloc[0] == pytest.approx(valor_fin / precio_inicio - 1)


def test_rebalanceo_con_el_mid_de_la_sesion_anterior() -> None:
    """Keep a bond through a one-session price gap without using the next quote."""
    universo = pd.DataFrame(
        [
            _bono("festivo", Coupon=3.65),
            _bono("no_emitido", **{"Issue date": "01/02/2024"}),
        ]
    )
    fechas = pd.to_datetime(["2024-01-02", "2024-01-31", "2024-02-01", "2024-02-02", "2024-03-01"])
    precios = pd.DataFrame(
        {"festivo": [100, 101, nan, 200, 105], "no_emitido": [100, 100, nan, 112, 113]},
        index=fechas,
    ).T
    benchmark = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=fechas)

    backtest, pesos = backtest_equiponderado(universo, precios, benchmark)

    assert pesos["festivo"].tolist() == [1, 1]
    assert pesos["no_emitido"].tolist() == [0, 0]
    cierre_febrero = 101 + 0.30
    cierre_marzo = 105 + 0.59
    assert backtest["Rent cartera"].tolist() == pytest.approx(
        [
            cierre_febrero / 100 - 1,
            cierre_marzo / cierre_febrero - 1,
        ]
    )


def test_limpieza_fechas_mid_y_vida_de_emision() -> None:
    """Clean quotes without treating an indicative call as a confirmed exit."""
    universo = pd.DataFrame(
        [
            _bono("nueva", **{"Issue date": "02/01/2024"}),
            _bono("vence", Maturity="03/01/2024"),
            _bono("sale"),
            _bono("prevista", Callable="Y", **{"Next Call Date": "03/01/2024"}),
            _bono("fecha_invalida", **{"Issue date": "31/02/2024"}),
        ]
    )
    bruto = pd.DataFrame(
        [
            [99, "101.5", 0, "texto", "inf", 100],
            [103, 102, 101, 100, 99, 80],
            [101, 102, 103, 104, 105, 80],
            [100, 101, 102, 103, 104, 80],
            [100, 101, 102, 103, 104, 80],
        ],
        index=universo["ISIN"],
        columns=[
            "01/01/2024",
            "02/01/2024",
            "03/01/2024",
            "04/01/2024",
            "05/01/2024",
            "31/02/2024",
        ],
    )
    call = SalidaBono(pd.Timestamp("2024-01-04"), "call", 100)

    limpio, resumen = limpiar_precios_historicos(universo, bruto, {"sale": call})

    assert limpio.shape == (5, 5)
    assert limpio.loc["nueva"].dropna().tolist() == [101.5]
    assert limpio.loc["vence"].dropna().tolist() == [103, 102]
    assert limpio.loc["sale"].dropna().tolist() == [101, 102, 103]
    assert limpio.loc["prevista"].notna().sum() == 5
    assert limpio.loc["fecha_invalida"].isna().all()
    assert resumen.tolist() == [1, 3, 1, 1, 2]
    assert bruto.shape == (5, 6)
