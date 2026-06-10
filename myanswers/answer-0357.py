from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preparar_datos(df, target_col):
    """
    Separa las características de la variable objetivo, imputa los valores faltantes 
    con el promedio y escala las características. Retorna X e y como arrays de numpy.
    """
    # Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()
    
    # Imputar valores faltantes (promedio)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    return X_scaled, y
