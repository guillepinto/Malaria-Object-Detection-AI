import os
import argparse
import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from datetime import datetime

def main():
    # Parsear los argumentos de la línea de comandos
    parser = argparse.ArgumentParser(description="Generate object detection submission")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model on Hugging Face or path to a local folder")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test images directory")
    args = parser.parse_args()

    # Check if the test path exists
    if not os.path.exists(args.test_path):
        print(f"Error: The test path '{args.test_path}' does not exist.")
        return

    submission = pd.DataFrame()  # Initialize submission variable

    try:
        # Cargar el modelo preentrenado
        image_processor = AutoImageProcessor.from_pretrained(args.model_name)
        model = AutoModelForObjectDetection.from_pretrained(args.model_name)

        # Obtener una lista de todos los archivos de imagen en el directorio
        image_files = os.listdir(args.test_path)

        # Inicializar una lista vacía para almacenar los resultados de todas las imágenes
        all_data = []

        # Iterar a través de cada imagen en el directorio
        for image_file in tqdm(image_files):
            # Ruta completa a la imagen
            img_path = os.path.join(args.test_path, image_file)
            image = Image.open(img_path)

            # Preparar la imagen para el modelo
            inputs = image_processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)

            # Post-procesar las predicciones del modelo
            width, height = image.size
            target_sizes = torch.tensor([height, width]).unsqueeze(0)
            results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

            if not results["boxes"].tolist():
                # Si no se detectan objetos, agregar una entrada 'NEG'
                all_data.append({
                    'Image_ID': image_file,
                    'class': 'NEG',
                    'confidence': 1.0,
                    'ymin': 0,
                    'xmin': 0,
                    'ymax': 0,
                    'xmax': 0
                })
            else:
                # Iterar a través de los resultados de esta imagen
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    box = [round(i, 2) for i in box.tolist()]
                    detected_class = 'Trophozoite' if model.config.id2label[label.item()] == 0 else 'WBC'

                    all_data.append({
                        'Image_ID': image_file,
                        'class': detected_class,
                        'confidence': round(score.item(), 3),
                        'ymin': box[1],
                        'xmin': box[0],
                        'ymax': box[3],
                        'xmax': box[2]
                    })

        # Convertir la lista a un DataFrame de Pandas
        submission = pd.DataFrame(all_data)

    except Exception as e:
        print(f"Ocurrió un error durante el procesamiento: {e}")
    finally:
        # Obtener la fecha y hora actual
        now = datetime.now()
        timestamp = now.strftime("%m%d_%H%M")

        # Sanity checks y asignación de output_dir
        safe_model_name = args.model_name.replace('/', '_')
        output_dir = f"submissions/{safe_model_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Nombre del archivo de salida
        output_file = os.path.join(output_dir, f'{safe_model_name}_submission_{timestamp}.csv')

        # Guardar el DataFrame como un archivo CSV
        submission.to_csv(output_file, index=False)
        print(f"Archivo de submission guardado como {output_file}")

if __name__ == "__main__":
    main()