import os
import sys
import importlib.util

def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    generators = [
        ("question_0001", "myquestions/question-0001-usecase-generator.py", "generar_caso_de_uso_expandir_json_y_resumir"),
        ("question_0002", "myquestions/question-0002-usecase-generator.py", "generar_caso_de_uso_alinear_sensor_eventos"),
        ("question_0003", "myquestions/question-0003-usecase-generator.py", "generar_caso_de_uso_clusterizar_dbscan"),
        ("question_0004", "myquestions/question-0004-usecase-generator.py", "generar_caso_de_uso_top_features_permutation_importance")
    ]

    all_passed = True

    for mod_name, path, func_name in generators:
        print(f"--- Probando {func_name} del archivo {path} ---")
        try:
            mod = load_module_from_file(mod_name, path)
            func = getattr(mod, func_name)
            for i in range(3):
                input_data, output_data = func()
                print(f"  Ejecución {i+1}:")
                if isinstance(input_data, dict):
                    print(f"    - Input keys: {list(input_data.keys())}")
                else:
                    print(f"    - Input type: {type(input_data)}")
                print(f"    - Output type: {type(output_data)}")
                
                if hasattr(output_data, "shape"):
                    print(f"    - Output shape: {output_data.shape}")
                elif hasattr(output_data, "__len__"):
                    print(f"    - Output len: {len(output_data)}")
        except Exception as e:
            print(f"Error en {func_name}: {e}")
            all_passed = False
            import traceback
            traceback.print_exc()
            
    if all_passed:
        print("\n¡Todas las pruebas pasaron exitosamente!")
    else:
        print("\nHubo errores durante las pruebas.")
        sys.exit(1)

if __name__ == "__main__":
    main()
