import pandas as pd

def analizar_churn(df):
    """
    Calcula el pago promedio, filtra clientes valiosos (> promedio y > 12 meses)
    y calcula la proporción de clientes que cancelaron el servicio ("Si").
    """
    # Calcular promedio general
    promedio_pago = df['pago_mensual'].mean()
    
    # Filtrar clientes valiosos
    clientes_valiosos = df[(df['pago_mensual'] > promedio_pago) & (df['antiguedad'] > 12)]
    
    # Si no hay clientes valiosos, retornar 0.0
    if len(clientes_valiosos) == 0:
        return 0.0
    
    # Calcular tasa de cancelación en el grupo
    cancelaciones = len(clientes_valiosos[clientes_valiosos['cancelo'] == "Si"])
    tasa_churn = cancelaciones / len(clientes_valiosos)
    
    return round(float(tasa_churn), 4)
