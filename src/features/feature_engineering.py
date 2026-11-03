import pandas as pd


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["data_dt"] = pd.to_datetime(result["data_dt"], dayfirst=True, errors="coerce")
    result["mes_num"] = result["data_dt"].dt.month
    result["dia_ano"] = result["data_dt"].dt.dayofyear
    result["fim_de_semana"] = result["data_dt"].dt.dayofweek.ge(5).astype(int)
    result["estacao"] = result["mes_num"].map(
        {
            12: "verao",
            1: "verao",
            2: "verao",
            3: "outono",
            4: "outono",
            5: "outono",
            6: "inverno",
            7: "inverno",
            8: "inverno",
            9: "primavera",
            10: "primavera",
            11: "primavera",
        }
    )
    return result
