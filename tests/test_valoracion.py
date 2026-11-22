"""Regression checks for semiannual coupons in the valuation engine."""

import pandas as pd
import pytest
from src.funciones_aux import calcular_sensibilidad, valoracion_bono


@pytest.mark.parametrize(
    ("vencimiento", "call", "fecha_amortizacion", "cupon_final"),
    [
        ("15/05/2027", None, "2027-05-15", 2.0),
        ("15/04/2027", None, "2027-04-15", 4 * 151 / 365),
        ("15/05/2027", "15/02/2027", "2027-02-15", 4 * 92 / 365),
    ],
)
def test_cupon_semestral(
    vencimiento: str, call: str | None, fecha_amortizacion: str, cupon_final: float
) -> None:
    """Pay half the annual coupon each semester and accrue irregular periods."""
    bono = pd.Series(
        {
            "First Coupon Date": "15/11/2025",
            "Maturity": vencimiento,
            "Callable": "Y" if call is not None else "N",
            "Next Call Date": call,
            "Coupon": 4.0,
            "Coupon Frequency": 2,
        }
    )
    curva = pd.DataFrame(
        {
            "fecha": ["2026-10-01", "2027-05-15"],
            "factor_descuento": [1.0, 1.0],
        }
    )

    limpio, sucio, corrido, flujos = valoracion_bono(bono, curva)

    assert flujos["fecha_pago"].tolist() == [
        pd.Timestamp("2026-11-15"),
        pd.Timestamp(fecha_amortizacion),
    ]
    assert flujos["cupon"].tolist() == pytest.approx([2.0, cupon_final])
    assert flujos["amortizacion"].tolist() == [0.0, 100.0]
    assert corrido == pytest.approx(4 * 139 / 365)
    assert sucio == pytest.approx(102 + cupon_final)
    assert limpio == pytest.approx(sucio - corrido)


@pytest.mark.parametrize("frecuencia", [1, 2])
def test_convexidad_es_segunda_derivada_del_precio(frecuencia: int) -> None:
    """Check the curvature for both annual and semiannual compounding."""
    flujos = pd.DataFrame({"plazo": [0.5, 1.5, 2.0], "flujo_total": [2.0, 2.0, 102.0]})
    y, paso = 0.04, 1e-4

    def precio(tipo: float) -> float:
        return float(
            sum(
                flujo / (1 + tipo / frecuencia) ** (frecuencia * plazo)
                for plazo, flujo in flujos.itertuples(index=False, name=None)
            )
        )

    _, _, convexidad = calcular_sensibilidad(flujos, y, frecuencia)
    precio_central = precio(y)
    derivada = (precio(y + paso) - 2 * precio_central + precio(y - paso)) / paso**2

    assert convexidad == pytest.approx(derivada / precio_central, rel=1e-6)
