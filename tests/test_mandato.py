"""Pruebas de la optimización utilizada en el apartado 6."""

import pandas as pd
import pytest
from src.mandato import optimizar_mandato


def test_selecciona_las_diez_mayores_yields() -> None:
    """Con límites holgados, invierte el máximo permitido en los diez mejores bonos."""
    indices = [f"B{i}" for i in range(12)]
    datos = pd.DataFrame(
        {
            "Issuer": indices,
            "Rating": "A",
            "Dur": 2.0,
            "Yield": [0.02 + i * 0.001 for i in range(12)],
        },
        index=indices,
    )
    pesos = optimizar_mandato(datos)
    assert set(pesos.index) == set(indices[2:])
    assert pesos.sum() == pytest.approx(1)
    assert pesos.to_list() == pytest.approx([0.1] * 10)


def test_rechaza_cartera_inviable_por_duracion() -> None:
    """Un universo con duración superior a tres en todos los bonos es inviable."""
    indices = [f"B{i}" for i in range(10)]
    datos = pd.DataFrame(
        {"Issuer": indices, "Rating": "A", "Dur": 3.1, "Yield": 0.04},
        index=indices,
    )
    with pytest.raises(ValueError, match="solución óptima"):
        optimizar_mandato(datos)
