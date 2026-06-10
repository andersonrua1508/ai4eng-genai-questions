import pandas as pd

def calcular_variacion_por_grupo(df, col_grupo, col_valor):
    """
    Ordena el DataFrame por col_grupo, calcula la diferencia de col_valor 
    dentro de cada grupo entre filas consecutivas y reemplaza el primer valor (NaN) por 0.
    """
    df_out = df.copy()
    
    # Ordenar por grupo (manteniendo el orden interno)
    df_out = df_out.sort_values(by=[col_grupo])
    
    # Calcular diferencia dentro de cada grupo
    df_out['variacion'] = df_out.groupby(col_grupo)[col_valor].diff()
    
    # Reemplazar NaN (primer valor de cada grupo) por 0
    df_out['variacion'] = df_out['variacion'].fillna(0)
    
    return df_out
