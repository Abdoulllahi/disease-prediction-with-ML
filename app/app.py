from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

def load_model(model_path):
    """
    Load the saved SVM model.
    """
    return joblib.load(model_path)


def predict_disease(model, user_inputs, label_encoder):
    """
    Predict the disease based on user inputs.
    """
    prediction = model.predict(user_inputs)
    predicted_class = label_encoder.inverse_transform(prediction)[0]
    return predicted_class

# Load the SVM model and label encoder
svm_model = load_model('app/models/svm_model.joblib')
label_encoder = load_model('app/models/label_encoder.joblib')

# Dictionary mapping symptoms to image filenames
symptom_images = {
    'itching': 'itching.png',
    'skin_rash': 'skin_rash.png',
    # 'nodal_skin_eruptions': 'nodal_skin_eruptions.png',
    'continuous_sneezing': 'continuous_sneezing.png',
    'shivering': 'shivering.png',
    'chills': 'chills.png',
    'joint_pain': 'joint_pain.png',
    'stomach_pain': 'stomach_pain.png',
    'acidity': 'acidity.png',
    # 'ulcers_on_tongue': 'ulcers_on_tongue.png',
    'muscle_wasting': 'muscle_wasting.png',
    'vomiting': 'vomiting.png',
    'burning_micturition': 'burning_micturition.png',
    # 'spotting_ urination': 'spotting_ urination.png',
    'fatigue': 'fatigue.png',
    'weight_gain': 'weight_gain.png',
    'anxiety': 'anxiety.png',
    # 'cold_hands_and_feets': 'cold_hands_and_feets.png',
    'mood_swings': 'mood_swings.png',
    'weight_loss': 'weight_loss.png',
    'restlessness': 'restlessness.png',
    'lethargy': 'lethargy.png',
    'patches_in_throat': 'patches_in_throat.png',
    'irregular_sugar_level': 'irregular_sugar_level.png',
    'cough': 'cough.png',
    'high_fever': 'high_fever.png',
    'sunken_eyes': 'sunken_eyes.png',
    'breathlessness': 'breathlessness.png',
    'sweating': 'sweating.png',
    # 'dehydration': 'dehydration.png',
    'indigestion': 'indigestion.png',
    'headache': 'headache.png',
    'yellowish_skin': 'yellowish_skin.png',
    'dark_urine': 'dark_urine.png',
    'nausea': 'nausea.png',
    'loss_of_appetite': 'loss_of_appetite.png',
    # 'pain_behind_the_eyes': 'pain_behind_the_eyes.png',
    'back_pain': 'back_pain.png',
    'constipation': 'constipation.png',
    'abdominal_pain': 'abdominal_pain.png',
    'diarrhoea': 'diarrhoea.png',
    'mild_fever': 'mild_fever.png',
    'yellow_urine': 'yellow_urine.png',
    'yellowing_of_eyes': 'yellowing_of_eyes.png',
    'acute_liver_failure': 'acute_liver_failure.png',
    # 'fluid_overload': 'fluid_overload.png',
    'swelling_of_stomach': 'swelling_of_stomach.png',
    'swelled_lymph_nodes': 'swelled_lymph_nodes.png',
    'malaise': 'malaise.png',
    'blurred_and_distorted_vision': 'blurred_and_distorted_vision.png',
    'phlegm': 'phlegm.png',
    'throat_irritation': 'throat_irritation.png',
    'redness_of_eyes': 'redness_of_eyes.png',
    'sinus_pressure': 'sinus_pressure.png',
    'runny_nose': 'runny_nose.png',
    'congestion': 'congestion.png',
    'chest_pain': 'chest_pain.png',
    # 'weakness_in_limbs': 'weakness_in_limbs.png',
    'fast_heart_rate': 'fast_heart_rate.png',
    # 'pain_during_bowel_movements': 'pain_during_bowel_movements.png',
    # 'pain_in_anal_region': 'pain_in_anal_region.png',
    # 'bloody_stool': 'bloody_stool.png',
    # 'irritation_in_anus': 'irritation_in_anus.png',
    'neck_pain': 'neck_pain.png',
    'dizziness': 'dizziness.png',
    'cramps': 'cramps.png',
    'bruising': 'bruising.png',
    'obesity': 'obesity.png',
    # 'swollen_legs': 'swollen_legs.png',
    'swollen_blood_vessels': 'swollen_blood_vessels.png',
    # 'puffy_face_and_eyes': 'puffy_face_and_eyes.png',
    'enlarged_thyroid': 'enlarged_thyroid.png',
    # 'brittle_nails': 'brittle_nails.png',
    # 'swollen_extremities': 'swollen_extremities.png',
    # 'excessive_hunger': 'excessive_hunger.png',
    # 'extra_marital_contacts': 'extra_marital_contacts.png',
    # 'drying_and_tingling_lips': 'drying_and_tingling_lips.png',
    'slurred_speech': 'slurred_speech.png',
    'knee_pain': 'knee_pain.png',
    'hip_joint_pain': 'hip_joint_pain.png',
    'muscle_weakness': 'muscle_weakness.png',
    # 'stiff_neck': 'stiff_neck.png',
    # 'swelling_joints': 'swelling_joints.png',
    # 'movement_stiffness': 'movement_stiffness.png',
    # 'spinning_movements': 'spinning_movements.png',
    'loss_of_balance': 'loss_of_balance.png',
    # 'unsteadiness': 'unsteadiness.png',
    # 'weakness_of_one_body_side': 'weakness_of_one_body_side.png',
    # 'loss_of_smell': 'loss_of_smell.png',
    # 'bladder_discomfort': 'bladder_discomfort.png',
    # 'foul_smell_of urine': 'foul_smell_of urine.png',
    # 'continuous_feel_of_urine': 'continuous_feel_of_urine.png',
    'passage_of_gases': 'passage_of_gases.png',
    # 'internal_itching': 'internal_itching.png',
    # 'toxic_look_(typhos)': 'toxic_look_(typhos).png',
    'depression': 'depression.png',
    # 'irritability': 'irritability.png',
    # 'muscle_pain': 'muscle_pain.png',
    # 'altered_sensorium': 'altered_sensorium.png',
    # 'red_spots_over_body': 'red_spots_over_body.png',
    # 'belly_pain': 'belly_pain.png',
    # 'abnormal_menstruation': 'abnormal_menstruation.png',
    # 'dischromic _patches': 'dischromic _patches.png',
    'watering_from_eyes': 'watering_from_eyes.png',
    # 'increased_appetite': 'increased_appetite.png',
    # 'polyuria': 'polyuria.png',
    # 'family_history': 'family_history.png',
    # 'mucoid_sputum': 'mucoid_sputum.png',
    # 'rusty_sputum': 'rusty_sputum.png',
    # 'lack_of_concentration': 'lack_of_concentration.png',
    # 'visual_disturbances': 'visual_disturbances.png',
    # 'receiving_blood_transfusion': 'receiving_blood_transfusion.png',
    # 'receiving_unsterile_injections': 'receiving_unsterile_injections.png',
    # 'coma': 'coma.png',
    # 'stomach_bleeding': 'stomach_bleeding.png',
    # 'distention_of_abdomen': 'distention_of_abdomen.png',
    # 'history_of_alcohol_consumption': 'history_of_alcohol_consumption.png',
    # 'fluid_overload.1': 'fluid_overload.1.png',
    # 'blood_in_sputum': 'blood_in_sputum.png',
    # 'prominent_veins_on_calf': 'prominent_veins_on_calf.png',
    'palpitations': 'palpitations.png',
    # 'painful_walking': 'painful_walking.png',
    'pus_filled_pimples': 'pus_filled_pimples.png',
    # 'blackheads': 'blackheads.png',
    # 'scurring': 'scurring.png',
    'skin_peeling': 'skin_peeling.png',
    # 'silver_like_dusting': 'silver_like_dusting.png',
    # 'small_dents_in_nails': 'small_dents_in_nails.png',
    # 'inflammatory_nails': 'inflammatory_nails.png',
    # 'blister': 'blister.png',
    # 'red_sore_around_nose': 'red_sore_around_nose.png',
    # 'yellow_crust_ooze': 'yellow_crust_ooze.png',
    'default': 'not_found.png'
}


@app.route('/', methods=['GET', 'POST'])
def index():
    symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain',
            'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition',
            'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings',
            'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
            'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache',
            'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain',
            'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes',
            'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
            'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure',
            'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
            'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain',
            'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes',
            'enlarged_thyroid', 'brittle_nails', 'swollen_extremities', 'excessive_hunger', 'extra_marital_contacts',
            'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck',
            'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
            'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
            'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression',
            'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
            'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite',
            'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration',
            'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma',
            'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1',
            'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples',
            'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
            'blister', 'red_sore_around_nose', 'yellow_crust_ooze']
    
    if request.method == 'POST':
        selected_symptoms = request.form.get('selected_symptoms', '').split(',')
        user_inputs = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]
        user_inputs = np.array(user_inputs).reshape(1, -1)

        # Predict the disease
        predicted_disease = predict_disease(svm_model, user_inputs, label_encoder)

        return render_template('index.html', symptoms=symptoms, result=predicted_disease, symptom_images=symptom_images)

    return render_template('index.html', symptoms=symptoms, result=None, symptom_images=symptom_images)

if __name__ == '__main__':
    app.run(debug=True)