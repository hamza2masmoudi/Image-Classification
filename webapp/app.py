import streamlit as st
import requests
from PIL import Image
import numpy as np

st.title("Classification d'Images CIFAR-10")
st.write("Téléchargez une image pour effectuer une prédiction.")

# Chargement de l'image
uploaded_file = st.file_uploader("Choisissez une image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Afficher l'image
    image = Image.open(uploaded_file)
    st.image(image, caption="Image chargée.", use_column_width=True)

    # Prétraitement de l'image
    image_array = np.array(image.resize((32, 32))) / 255.0  # Resizing and normalizing
    
    # Extraction de 4 caractéristiques : moyennes des canaux R, G, B et moyenne générale
    mean_red = np.mean(image_array[:, :, 0])  # Moyenne canal rouge
    mean_green = np.mean(image_array[:, :, 1])  # Moyenne canal vert
    mean_blue = np.mean(image_array[:, :, 2])  # Moyenne canal bleu
    overall_mean = np.mean(image_array)  # Moyenne générale

    data = [mean_red, mean_green, mean_blue, overall_mean]

    # Bouton pour effectuer une prédiction
    if st.button("Prédire"):
        try:
            # Appel à l'API
            response = requests.post(
                "http://api:8000/predict",  # Docker Compose service name for the API
                json={"data": data}
            )

            if response.status_code == 200:
                predicted_class = response.json().get("prediction", ["Unknown"])[0]
                st.success(f"Classe prédite : {predicted_class}")

                # Champ pour entrer la cible réelle
                real_target = st.number_input("Entrez la cible réelle (target)", min_value=0, max_value=9, step=1)

                # Bouton pour envoyer un feedback
                if st.button("Envoyer un feedback"):
                    try:
                        feedback_response = requests.post(
                            "http://api:8000/feedback",  # Endpoint de feedback
                            json={
                                "image": data,
                                "predicted_class": predicted_class,
                                "actual_class": real_target
                            }
                        )
                        if feedback_response.status_code == 200:
                            st.success("Feedback envoyé avec succès !")
                        else:
                            st.error(f"Erreur lors de l'envoi du feedback : {feedback_response.text}")
                    except Exception as e:
                        st.error(f"Erreur : {str(e)}")
            else:
                st.error(f"Erreur lors de la prédiction : {response.text}")
        except Exception as e:
            st.error(f"Erreur : {str(e)}")